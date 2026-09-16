"""Define validated configuration for UMA catalysis workflows."""

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path

Facet = tuple[int, int, int]

VALID_WORKFLOWS = frozenset(
    {
        "bulk",
        "surface_energies",
        "wulff",
        "h_adsorption",
        "coverage",
        "co_reaction",
    }
)


def _validate_non_empty(value: str, name: str) -> None:
    """Raise a ValueError when a required string is empty."""
    if not value.strip():
        raise ValueError(f"{name} must not be empty.")


def _validate_positive(value: float | int, name: str) -> None:
    """Raise a ValueError when a required numeric value is not positive."""
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive, finite value.")


def _validate_facet(facet: Facet, name: str) -> None:
    """Raise a ValueError when a Miller-index tuple is malformed."""
    if len(facet) != 3 or not all(isinstance(index, int) for index in facet):
        raise ValueError(f"{name} must contain exactly three integer Miller indices.")
    if not any(facet):
        raise ValueError(f"{name} must not be the zero Miller index.")


@dataclass(frozen=True, slots=True)
class RunConfig:
    """
    Select workflows and presentation behavior for one execution.

    Parameters
    ----------
    workflows : tuple[str, ...], optional
        Ordered workflow names to execute. Default is ``("bulk",)``.
    output_directory : pathlib.Path, optional
        Root directory for generated artifacts. Default is ``outputs``.
    show_figures : bool, optional
        Whether the eventual runner should display generated figures. Default
        is False.
    """

    workflows: tuple[str, ...] = ("bulk",)
    output_directory: Path = Path("outputs")
    show_figures: bool = False

    def __post_init__(self) -> None:
        """Validate selected workflows and the output destination."""
        if not self.workflows:
            raise ValueError("workflows must not be empty.")
        if len(set(self.workflows)) != len(self.workflows):
            raise ValueError("workflows must not contain duplicates.")
        invalid_workflows = set(self.workflows).difference(VALID_WORKFLOWS)
        if invalid_workflows:
            raise ValueError(
                f"Unknown workflows: {', '.join(sorted(invalid_workflows))}."
            )
        if not str(self.output_directory):
            raise ValueError("output_directory must not be empty.")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """
    Define UMA and D3 calculator settings.

    Parameters
    ----------
    model_name : str, optional
        FAIR Chemistry pretrained model identifier. Default is ``"uma-s-1p2"``.
    d3_device : str, optional
        Device requested for the D3 endpoint calculator. Default is ``"cpu"``.
    d3_damping : str, optional
        Dispersion damping model. Default is ``"bj"``.
    """

    model_name: str = "uma-s-1p2"
    d3_device: str = "cpu"
    d3_damping: str = "bj"

    def __post_init__(self) -> None:
        """Validate calculator identifiers."""
        _validate_non_empty(self.model_name, "model_name")
        _validate_non_empty(self.d3_device, "d3_device")
        _validate_non_empty(self.d3_damping, "d3_damping")


@dataclass(frozen=True, slots=True)
class ComputeConfig:
    """
    Define reproducibility, relaxation, ZPE, and NEB work budgets.

    Parameters
    ----------
    random_seed : int, optional
        Seed used for stochastic candidate generation. Default is 42.
    fast_mode : bool, optional
        Whether a workflow should use its reduced classroom settings. Default
        is False.
    relaxation_force_threshold : float, optional
        Maximum force in eV/angstrom for geometry relaxation. Default is 0.05.
    relaxation_steps : int, optional
        Maximum geometry-optimization steps. Default is 300.
    adsorption_candidate_count : int, optional
        Number of adsorbate-placement candidates to generate. Default is 5.
    calculate_zpe : bool, optional
        Whether workflows should calculate zero-point-energy corrections.
        Default is True.
    run_neb : bool, optional
        Whether the CO-reaction workflow should run its expensive NEB branch.
        Default is False.
    neb_force_threshold : float, optional
        Maximum force in eV/angstrom for NEB optimization. Default is 0.1.
    neb_steps : int, optional
        Maximum NEB optimization steps. Default is 300.
    neb_intermediate_images : int, optional
        Number of intermediate NEB images. Default is 10.
    """

    random_seed: int = 42
    fast_mode: bool = False
    relaxation_force_threshold: float = 0.05
    relaxation_steps: int = 300
    adsorption_candidate_count: int = 5
    calculate_zpe: bool = True
    run_neb: bool = False
    neb_force_threshold: float = 0.1
    neb_steps: int = 300
    neb_intermediate_images: int = 10

    def __post_init__(self) -> None:
        """Validate reproducibility and numerical work-budget settings."""
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative.")
        _validate_positive(
            self.relaxation_force_threshold, "relaxation_force_threshold"
        )
        _validate_positive(self.relaxation_steps, "relaxation_steps")
        _validate_positive(
            self.adsorption_candidate_count, "adsorption_candidate_count"
        )
        _validate_positive(self.neb_force_threshold, "neb_force_threshold")
        _validate_positive(self.neb_steps, "neb_steps")
        _validate_positive(self.neb_intermediate_images, "neb_intermediate_images")


@dataclass(frozen=True, slots=True)
class MaterialConfig:
    """
    Define the bulk and surface system studied by tutorial workflows.

    Parameters
    ----------
    element : str, optional
        Chemical element used to construct the bulk reference. Default is
        ``"Ni"``.
    crystal_structure : str, optional
        ASE bulk crystal-structure identifier. Default is ``"fcc"``.
    initial_lattice_constant : float, optional
        Initial bulk lattice constant in angstrom. Default is 3.52.
    experimental_lattice_constant : float, optional
        Experimental reference lattice constant in angstrom. Default is 3.524.
    surface_facets : tuple[Facet, ...], optional
        Miller indices used for surface-energy calculations. Default is the
        four facets used by the tutorial.
    surface_thicknesses : tuple[int, ...], optional
        Slab layer counts used for surface-energy fitting. Default is
        ``(4, 6, 8)``.
    adsorption_facet : Facet, optional
        Miller index used for adsorption workflows. Default is ``(1, 1, 1)``.
    vacuum_size : float, optional
        Vacuum spacing in angstrom. Default is 10.0.
    """

    element: str = "Ni"
    crystal_structure: str = "fcc"
    initial_lattice_constant: float = 3.52
    experimental_lattice_constant: float = 3.524
    surface_facets: tuple[Facet, ...] = (
        (1, 1, 1),
        (1, 0, 0),
        (1, 1, 0),
        (2, 1, 1),
    )
    surface_thicknesses: tuple[int, ...] = (4, 6, 8)
    adsorption_facet: Facet = (1, 1, 1)
    vacuum_size: float = 10.0

    def __post_init__(self) -> None:
        """Validate bulk and surface configuration values."""
        _validate_non_empty(self.element, "element")
        _validate_non_empty(self.crystal_structure, "crystal_structure")
        _validate_positive(self.initial_lattice_constant, "initial_lattice_constant")
        _validate_positive(
            self.experimental_lattice_constant, "experimental_lattice_constant"
        )
        if not self.surface_facets:
            raise ValueError("surface_facets must not be empty.")
        for facet in self.surface_facets:
            _validate_facet(facet, "surface_facets")
        if not self.surface_thicknesses:
            raise ValueError("surface_thicknesses must not be empty.")
        if any(thickness < 1 for thickness in self.surface_thicknesses):
            raise ValueError("surface_thicknesses must contain positive values.")
        _validate_facet(self.adsorption_facet, "adsorption_facet")
        _validate_positive(self.vacuum_size, "vacuum_size")


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    """
    Define optional CodeCarbon tracking settings.

    Parameters
    ----------
    enabled : bool, optional
        Whether CodeCarbon tracking is enabled. Default is False.
    project_name : str, optional
        CodeCarbon project identifier. Default is
        ``"ccai-uma-catalysis-tutorial"``.
    output_file : str, optional
        File name written beneath the run output directory. Default is
        ``"emissions.csv"``.
    """

    enabled: bool = False
    project_name: str = "ccai-uma-catalysis-tutorial"
    output_file: str = "emissions.csv"

    def __post_init__(self) -> None:
        """Validate CodeCarbon identifiers and output filename."""
        _validate_non_empty(self.project_name, "project_name")
        _validate_non_empty(self.output_file, "output_file")
        if Path(self.output_file).name != self.output_file:
            raise ValueError("output_file must be a filename, not a path.")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """
    Collect the complete typed configuration for one tutorial execution.

    Parameters
    ----------
    run : RunConfig, optional
        Workflow selection and output settings. Default is ``RunConfig()``.
    model : ModelConfig, optional
        UMA and D3 calculator settings. Default is ``ModelConfig()``.
    compute : ComputeConfig, optional
        Reproducibility and computational-budget settings. Default is
        ``ComputeConfig()``.
    material : MaterialConfig, optional
        Bulk and surface system settings. Default is ``MaterialConfig()``.
    tracking : TrackingConfig, optional
        Optional CodeCarbon settings. Default is ``TrackingConfig()``.
    """

    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    material: MaterialConfig = field(default_factory=MaterialConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
