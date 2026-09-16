"""Load, resolve, execute, and summarize UMA catalysis experiments."""

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

from uma_catalysis.artifacts import ArtifactStore
from uma_catalysis.calculations import load_predictor
from uma_catalysis.structs.config import (
    ComputeConfig,
    ExperimentConfig,
    Facet,
    MaterialConfig,
    ModelConfig,
    RunConfig,
    TrackingConfig,
)
from uma_catalysis.tracking import track_emissions
from uma_catalysis.workflows import (
    run_bulk_optimization,
    run_co_reaction_study,
    run_coverage_study,
    run_hydrogen_adsorption,
    run_surface_energy_study,
    run_wulff_construction,
)

WORKFLOW_DEPENDENCIES: Mapping[str, tuple[str, ...]] = {
    "bulk": (),
    "surface_energies": ("bulk",),
    "wulff": ("surface_energies",),
    "h_adsorption": ("bulk",),
    "coverage": ("bulk",),
    "co_reaction": ("bulk",),
}


@dataclass(frozen=True, slots=True)
class ExperimentRun:
    """Store results and top-level artifacts from one executed experiment."""

    requested_workflows: tuple[str, ...]
    executed_workflows: tuple[str, ...]
    results: Mapping[str, Any]
    summary_path: Path
    emissions_path: Path | None = None


def _section(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Return a required TOML table as a string-keyed mapping."""
    section = data.get(name)
    if not isinstance(section, Mapping):
        raise ValueError(f"Configuration must define a [{name}] table.")
    return section


def _string(value: Any, name: str) -> str:
    """Validate and return a TOML string value."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _boolean(value: Any, name: str) -> bool:
    """Validate and return a TOML boolean value."""
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean.")
    return value


def _integer(value: Any, name: str) -> int:
    """Validate and return a TOML integer value."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")
    return value


def _number(value: Any, name: str) -> float:
    """Validate and return a TOML numeric value as a float."""
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number.")
    return float(value)


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    """Validate and return a TOML array of strings."""
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be an array of strings.")
    return tuple(value)


def _integer_tuple(value: Any, name: str) -> tuple[int, ...]:
    """Validate and return a TOML array of integers."""
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array of integers.")
    return tuple(_integer(item, name) for item in value)


def _facet(value: Any, name: str) -> Facet:
    """Validate and return one three-index Miller facet."""
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must be an array of three integers.")
    return tuple(_integer(index, name) for index in value)


def _facet_tuple(value: Any, name: str) -> tuple[Facet, ...]:
    """Validate and return a TOML array of Miller facets."""
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array of Miller facets.")
    return tuple(_facet(facet, name) for facet in value)


def load_experiment_config(path: Path) -> ExperimentConfig:
    """
    Load an UMA catalysis experiment configuration from TOML.

    Parameters
    ----------
    path : pathlib.Path
        TOML configuration file to load.

    Returns
    -------
    uma_catalysis.structs.config.ExperimentConfig
        Validated configuration with paths resolved relative to the TOML file.

    Raises
    ------
    ValueError
        If the file is missing, malformed, or does not match the configuration
        schema.
    """
    if not path.is_file():
        raise ValueError(f"Configuration file does not exist: {path}")
    try:
        with path.open("rb") as file:
            raw = tomllib.load(file)
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"Invalid TOML configuration: {path}") from error

    run = _section(raw, "run")
    model = _section(raw, "model")
    compute = _section(raw, "compute")
    material = _section(raw, "material")
    tracking = _section(raw, "tracking")

    output_directory = Path(
        _string(run.get("output_directory"), "run.output_directory")
    )
    if not output_directory.is_absolute():
        output_directory = (path.parent / output_directory).resolve()

    return ExperimentConfig(
        run=RunConfig(
            workflows=_string_tuple(run.get("workflows"), "run.workflows"),
            output_directory=output_directory,
            show_figures=_boolean(run.get("show_figures"), "run.show_figures"),
        ),
        model=ModelConfig(
            model_name=_string(model.get("model_name"), "model.model_name"),
            d3_device=_string(model.get("d3_device"), "model.d3_device"),
            d3_damping=_string(model.get("d3_damping"), "model.d3_damping"),
        ),
        compute=ComputeConfig(
            random_seed=_integer(compute.get("random_seed"), "compute.random_seed"),
            fast_mode=_boolean(compute.get("fast_mode"), "compute.fast_mode"),
            relaxation_force_threshold=_number(
                compute.get("relaxation_force_threshold"),
                "compute.relaxation_force_threshold",
            ),
            relaxation_steps=_integer(
                compute.get("relaxation_steps"), "compute.relaxation_steps"
            ),
            adsorption_candidate_count=_integer(
                compute.get("adsorption_candidate_count"),
                "compute.adsorption_candidate_count",
            ),
            calculate_zpe=_boolean(
                compute.get("calculate_zpe"), "compute.calculate_zpe"
            ),
            run_neb=_boolean(compute.get("run_neb"), "compute.run_neb"),
            neb_force_threshold=_number(
                compute.get("neb_force_threshold"), "compute.neb_force_threshold"
            ),
            neb_steps=_integer(compute.get("neb_steps"), "compute.neb_steps"),
            neb_intermediate_images=_integer(
                compute.get("neb_intermediate_images"),
                "compute.neb_intermediate_images",
            ),
        ),
        material=MaterialConfig(
            element=_string(material.get("element"), "material.element"),
            crystal_structure=_string(
                material.get("crystal_structure"), "material.crystal_structure"
            ),
            initial_lattice_constant=_number(
                material.get("initial_lattice_constant"),
                "material.initial_lattice_constant",
            ),
            experimental_lattice_constant=_number(
                material.get("experimental_lattice_constant"),
                "material.experimental_lattice_constant",
            ),
            surface_facets=_facet_tuple(
                material.get("surface_facets"), "material.surface_facets"
            ),
            surface_thicknesses=_integer_tuple(
                material.get("surface_thicknesses"), "material.surface_thicknesses"
            ),
            adsorption_facet=_facet(
                material.get("adsorption_facet"), "material.adsorption_facet"
            ),
            vacuum_size=_number(material.get("vacuum_size"), "material.vacuum_size"),
        ),
        tracking=TrackingConfig(
            enabled=_boolean(tracking.get("enabled"), "tracking.enabled"),
            project_name=_string(tracking.get("project_name"), "tracking.project_name"),
            output_file=_string(tracking.get("output_file"), "tracking.output_file"),
        ),
    )


def resolve_workflows(requested_workflows: tuple[str, ...]) -> tuple[str, ...]:
    """Return requested workflows with all prerequisites in execution order."""
    resolved: list[str] = []
    visiting: set[str] = set()

    def visit(workflow: str) -> None:
        if workflow in resolved:
            return
        if workflow in visiting:
            raise ValueError(f"Workflow dependency cycle includes {workflow!r}.")
        try:
            dependencies = WORKFLOW_DEPENDENCIES[workflow]
        except KeyError as error:
            raise ValueError(f"Unknown workflow dependency: {workflow!r}.") from error
        visiting.add(workflow)
        for dependency in dependencies:
            visit(dependency)
        visiting.remove(workflow)
        resolved.append(workflow)

    for requested_workflow in requested_workflows:
        visit(requested_workflow)
    return tuple(resolved)


def _seed_generators(seed: int) -> None:
    """Set random seeds at the experiment boundary for reproducible sampling."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


def _serialize(value: Any) -> Any:
    """Convert typed result data to JSON-compatible values without ASE objects."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_serialize(item) for item in value]
    if is_dataclass(value):
        return {
            field.name: _serialize(getattr(value, field.name))
            for field in fields(value)
            if not field.name.endswith("_atoms")
        }
    raise TypeError(f"Cannot serialize result value of type {type(value).__name__}.")


def _show_result_figures(results: Mapping[str, Any]) -> None:
    """Display saved workflow figures when interactive presentation is requested."""
    figure_paths = []
    for result in results.values():
        for attribute in ("figure_path", "visualization_path", "neb_figure_path"):
            path = getattr(result, attribute, None)
            if isinstance(path, Path) and path.is_file():
                figure_paths.append(path)
    if not figure_paths:
        return

    import matplotlib.pyplot as plt

    for path in figure_paths:
        figure, axis = plt.subplots()
        axis.imshow(plt.imread(path))
        axis.set_axis_off()
        axis.set_title(path.stem)
    plt.show()


def run_experiment(config: ExperimentConfig) -> ExperimentRun:
    """
    Execute configured workflows once their typed prerequisites are available.

    UMA is loaded once, inside the optional emissions-tracking boundary, and
    shared by every selected workflow. A JSON summary is written beneath the
    configured output root after all stages have completed.
    """
    requested_workflows = config.run.workflows
    executed_workflows = resolve_workflows(requested_workflows)
    artifacts = ArtifactStore(config.run.output_directory)
    _seed_generators(config.compute.random_seed)
    results: dict[str, Any] = {}

    with track_emissions(config.tracking, artifacts) as emissions_path:
        predictor = load_predictor(config.model)
        for workflow in executed_workflows:
            if workflow == "bulk":
                results[workflow] = run_bulk_optimization(
                    predictor,
                    config.material,
                    config.compute,
                    artifacts,
                )
            elif workflow == "surface_energies":
                results[workflow] = run_surface_energy_study(
                    predictor,
                    results["bulk"],
                    config.material,
                    config.compute,
                    artifacts,
                )
            elif workflow == "wulff":
                results[workflow] = run_wulff_construction(
                    results["bulk"],
                    results["surface_energies"],
                    config.material,
                    artifacts,
                )
            elif workflow == "h_adsorption":
                results[workflow] = run_hydrogen_adsorption(
                    predictor,
                    results["bulk"],
                    config.model,
                    config.material,
                    config.compute,
                    artifacts,
                )
            elif workflow == "coverage":
                results[workflow] = run_coverage_study(
                    predictor,
                    results["bulk"],
                    config.model,
                    config.material,
                    config.compute,
                    artifacts,
                )
            elif workflow == "co_reaction":
                results[workflow] = run_co_reaction_study(
                    predictor,
                    results["bulk"],
                    config.model,
                    config.material,
                    config.compute,
                    artifacts,
                )
            else:
                raise RuntimeError(f"No handler was registered for {workflow!r}.")

    summary_path = artifacts.write_json(
        {
            "requested_workflows": list(requested_workflows),
            "executed_workflows": list(executed_workflows),
            "model": config.model.model_name,
            "random_seed": config.compute.random_seed,
            "emissions": None if emissions_path is None else str(emissions_path),
            "results": _serialize(results),
        },
        Path("run_summary.json"),
    )
    if config.run.show_figures:
        _show_result_figures(results)
    return ExperimentRun(
        requested_workflows=requested_workflows,
        executed_workflows=executed_workflows,
        results=results,
        summary_path=summary_path,
        emissions_path=emissions_path,
    )
