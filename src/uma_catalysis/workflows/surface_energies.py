"""Calculate surface energies from relaxed slabs and linear energy fits."""

from pathlib import Path
from typing import Any

from uma_catalysis.artifacts import ArtifactStore
from uma_catalysis.calculations import (
    build_bulk_structure,
    build_surface_energy_slab,
    build_uma_calculator,
    linear_fit,
    relax_positions,
    surface_energy_from_intercept,
)
from uma_catalysis.evaluation import plot_surface_energy_fits
from uma_catalysis.structs.config import ComputeConfig, MaterialConfig
from uma_catalysis.structs.results import (
    BulkOptimizationResult,
    SurfaceEnergyResult,
    SurfaceEnergyStudyResult,
)

PART_DIRECTORY = Path("part2-surface-energies")


def _facet_name(facet: tuple[int, int, int]) -> str:
    """Return the tutorial's compact facet label, such as ``ni111``."""
    return "ni" + "".join(str(index) for index in facet)


def _reference_bulk_atoms(
    bulk_result: BulkOptimizationResult, material: MaterialConfig
) -> Any:
    """Return a relaxed bulk structure or reconstruct a reproducible reference."""
    if bulk_result.relaxed_atoms is not None:
        return bulk_result.relaxed_atoms.copy()
    return build_bulk_structure(material, bulk_result.optimized_lattice_constant)


def run_surface_energy_study(
    predictor: Any,
    bulk_result: BulkOptimizationResult,
    material: MaterialConfig,
    compute: ComputeConfig,
    artifacts: ArtifactStore,
) -> SurfaceEnergyStudyResult:
    """
    Relax slabs, fit slab energy against atom count, and calculate surface energy.

    Parameters
    ----------
    predictor : Any
        FAIR Chemistry prediction unit used to construct ``omat`` calculators.
    bulk_result : uma_catalysis.structs.results.BulkOptimizationResult
        Bulk optimization result that supplies the lattice constant and, when
        available, the relaxed bulk reference structure.
    material : uma_catalysis.structs.config.MaterialConfig
        Facets, slab thicknesses, and vacuum settings for the study.
    compute : uma_catalysis.structs.config.ComputeConfig
        Slab-relaxation force threshold and step budget.
    artifacts : uma_catalysis.artifacts.ArtifactStore
        Controlled destination for structures, trajectories, logs, plots, and
        result metadata.

    Returns
    -------
    uma_catalysis.structs.results.SurfaceEnergyStudyResult
        Shared bulk reference energy and one fitted surface-energy result per
        selected facet.
    """
    artifacts.directory(PART_DIRECTORY)
    bulk_atoms = _reference_bulk_atoms(bulk_result, material)
    bulk_atoms.calc = build_uma_calculator(predictor, task_name="omat")
    bulk_energy_per_atom = float(bulk_atoms.get_potential_energy()) / len(bulk_atoms)
    facet_results = []

    for facet in material.surface_facets:
        facet_directory = PART_DIRECTORY / _facet_name(facet)
        artifacts.directory(facet_directory)
        atom_counts = []
        slab_energies = []
        surface_area = None

        for layer_count in material.surface_thicknesses:
            slab = build_surface_energy_slab(
                bulk_atoms=bulk_atoms,
                facet=facet,
                layer_count=layer_count,
                lattice_constant=bulk_result.optimized_lattice_constant,
                vacuum_size=material.vacuum_size,
            )
            slab.calc = build_uma_calculator(predictor, task_name="omat")
            artifact_stem = f"{_facet_name(facet)}_{layer_count}_layers"
            outcome = relax_positions(
                slab,
                force_threshold=compute.relaxation_force_threshold,
                steps=compute.relaxation_steps,
                trajectory_path=artifacts.path(
                    facet_directory / f"{artifact_stem}.traj"
                ),
                logfile_path=artifacts.path(facet_directory / f"{artifact_stem}.log"),
            )
            structure_path = artifacts.write_atoms(
                outcome.atoms,
                facet_directory / f"{artifact_stem}_relaxed.cif",
            )
            atom_counts.append(len(outcome.atoms))
            slab_energies.append(float(outcome.atoms.get_potential_energy()))
            cell = outcome.atoms.get_cell()
            surface_area = float(
                (cell[0][1] * cell[1][2] - cell[0][2] * cell[1][1]) ** 2
                + (cell[0][2] * cell[1][0] - cell[0][0] * cell[1][2]) ** 2
                + (cell[0][0] * cell[1][1] - cell[0][1] * cell[1][0]) ** 2
            ) ** 0.5
            artifacts.write_json(
                {
                    "facet": list(facet),
                    "layer_count": layer_count,
                    "atom_count": len(outcome.atoms),
                    "energy_ev": slab_energies[-1],
                    "converged": outcome.converged,
                    "structure": str(structure_path),
                },
                facet_directory / f"{artifact_stem}.json",
            )

        if surface_area is None:
            raise RuntimeError("No slabs were generated for the configured facet.")
        fitted_slope, fitted_intercept = linear_fit(
            tuple(atom_counts), tuple(slab_energies)
        )
        surface_energy_ev, surface_energy_si = surface_energy_from_intercept(
            fitted_intercept, surface_area
        )
        facet_result = SurfaceEnergyResult(
            facet=facet,
            atom_counts=tuple(atom_counts),
            slab_energies=tuple(slab_energies),
            fit_slope=fitted_slope,
            fit_intercept=fitted_intercept,
            surface_energy_ev_per_angstrom_squared=surface_energy_ev,
            surface_energy_j_per_m2=surface_energy_si,
        )
        facet_results.append(facet_result)
        artifacts.write_json(
            {
                "facet": list(facet),
                "atom_counts": atom_counts,
                "slab_energies_ev": slab_energies,
                "fit_slope_ev_per_atom": fitted_slope,
                "fit_intercept_ev": fitted_intercept,
                "surface_energy_ev_per_angstrom_squared": surface_energy_ev,
                "surface_energy_j_per_m2": surface_energy_si,
            },
            facet_directory / "result.json",
        )

    figure = plot_surface_energy_fits(facet_results, material.element)
    figure_path = artifacts.write_figure(
        figure,
        PART_DIRECTORY / "surface_energy_fits.png",
    )
    import matplotlib.pyplot as plt

    plt.close(figure)
    study = SurfaceEnergyStudyResult(
        bulk_energy_per_atom=bulk_energy_per_atom,
        facet_results=tuple(facet_results),
        figure_path=figure_path,
    )
    artifacts.write_json(
        {
            "bulk_energy_per_atom_ev": study.bulk_energy_per_atom,
            "facets": [
                {
                    "facet": list(result.facet),
                    "surface_energy_j_per_m2": result.surface_energy_j_per_m2,
                }
                for result in study.facet_results
            ],
            "figure": str(figure_path),
        },
        PART_DIRECTORY / "result.json",
    )
    return study
