"""Calculate coverage-dependent H adsorption and lateral interactions."""

from pathlib import Path
from typing import Any

from uma_catalysis.artifacts import ArtifactStore
from uma_catalysis.calculations import (
    adsorption_energy,
    build_adsorption_slab,
    build_d3_calculator,
    build_diatomic_reference,
    evaluate_energy_components,
    generate_multiple_adsorbate_candidates,
    linear_fit,
    relax_with_uma,
)
from uma_catalysis.evaluation import plot_coverage_dependence
from uma_catalysis.structs.config import ComputeConfig, MaterialConfig, ModelConfig
from uma_catalysis.structs.results import (
    BulkOptimizationResult,
    CoveragePoint,
    CoverageResult,
    EnergyComponents,
)

PART_DIRECTORY = Path("part5-coverage-dependence")
FAST_HYDROGEN_COUNTS = (1, 4, 8)
FULL_HYDROGEN_COUNTS = (1, 4, 8, 12, 16)


def _hydrogen_counts(compute: ComputeConfig) -> tuple[int, ...]:
    """Return the notebook's fast or full H-count series."""
    return FAST_HYDROGEN_COUNTS if compute.fast_mode else FULL_HYDROGEN_COUNTS


def _surface_site_count(atoms: Any) -> int:
    """Count FAIR Chemistry slab surface atoms identified by tag 1."""
    count = sum(tag == 1 for tag in atoms.get_tags())
    if count < 1:
        raise ValueError("The adsorption slab has no surface-site tags.")
    return count


def _relax_oc20(
    atoms: Any,
    predictor: Any,
    compute: ComputeConfig,
    trajectory_path: Path | None,
    logfile_path: Path | None,
) -> Any:
    """Relax one coverage-study structure using the shared OC20 helper."""
    return relax_with_uma(
        atoms,
        predictor,
        task_name="oc20",
        force_threshold=compute.relaxation_force_threshold,
        steps=compute.relaxation_steps,
        trajectory_path=trajectory_path,
        logfile_path=logfile_path,
    )


def run_coverage_study(
    predictor: Any,
    bulk_result: BulkOptimizationResult,
    model: ModelConfig,
    material: MaterialConfig,
    compute: ComputeConfig,
    artifacts: ArtifactStore,
) -> CoverageResult:
    """
    Calculate H adsorption at multiple coverages and fit lateral interactions.

    The average adsorption energy uses the tutorial convention:
    ``(E(nH*) - E(*) - n E(H2)/2) / n``. The resulting fit has the form
    ``E_ads(theta) = intercept + interaction_parameter * theta``.
    """
    part_directory = artifacts.directory(PART_DIRECTORY)
    d3_calculator = build_d3_calculator(model)
    reference_slab = build_adsorption_slab(
        material,
        lattice_constant=bulk_result.optimized_lattice_constant,
    )
    clean_outcome = _relax_oc20(
        reference_slab.atoms.copy(),
        predictor,
        compute,
        trajectory_path=part_directory / "ni111_clean.traj",
        logfile_path=part_directory / "ni111_clean.log",
    )
    clean_energy = evaluate_energy_components(clean_outcome.atoms, d3_calculator)
    clean_slab_path = artifacts.write_atoms(
        clean_outcome.atoms,
        PART_DIRECTORY / "ni111_clean.xyz",
    )
    surface_site_count = _surface_site_count(clean_outcome.atoms)

    h2 = build_diatomic_reference(
        symbols=("H", "H"),
        bond_length=0.74,
        vacuum_size=material.vacuum_size,
    )
    h2_outcome = _relax_oc20(
        h2,
        predictor,
        compute,
        trajectory_path=part_directory / "h2.traj",
        logfile_path=part_directory / "h2.log",
    )
    h2_energy = evaluate_energy_components(h2_outcome.atoms, d3_calculator)
    hydrogen_reference_path = artifacts.write_atoms(
        h2_outcome.atoms,
        PART_DIRECTORY / "h2_optimized.xyz",
    )

    points: list[CoveragePoint] = []
    for hydrogen_count in _hydrogen_counts(compute):
        coverage_slab = build_adsorption_slab(
            material,
            lattice_constant=bulk_result.optimized_lattice_constant,
        )
        coverage_slab.atoms = clean_outcome.atoms.copy()
        candidates = generate_multiple_adsorbate_candidates(
            coverage_slab,
            adsorbate_smiles=("*H",) * hydrogen_count,
            count=compute.adsorption_candidate_count,
        )
        if not candidates:
            raise RuntimeError(
                f"No configurations were generated for {hydrogen_count} H adsorbates."
            )

        candidate_energies: list[EnergyComponents] = []
        candidate_structures: list[Any] = []
        for candidate_number, candidate in enumerate(candidates, start=1):
            candidate.set_pbc([True, True, True])
            outcome = _relax_oc20(
                candidate.copy(),
                predictor,
                compute,
                trajectory_path=(
                    part_directory
                    / f"coverage_{hydrogen_count:02d}h_{candidate_number}.traj"
                ),
                logfile_path=None,
            )
            energy = evaluate_energy_components(outcome.atoms, d3_calculator)
            candidate_energies.append(energy)
            candidate_structures.append(outcome.atoms)

        best_candidate_index = min(
            range(len(candidate_energies)),
            key=lambda index: candidate_energies[index].total_energy,
        )
        best_structure_path = artifacts.write_atoms(
            candidate_structures[best_candidate_index],
            PART_DIRECTORY / f"coverage_{hydrogen_count:02d}h_best.xyz",
        )
        adsorption_energy_per_hydrogen = (
            adsorption_energy(
                candidate_energies[best_candidate_index],
                clean_energy,
                h2_energy,
                reference_multiplier=hydrogen_count * 0.5,
            )
            / hydrogen_count
        )
        points.append(
            CoveragePoint(
                coverage=hydrogen_count / surface_site_count,
                adsorption_energy_per_adsorbate=adsorption_energy_per_hydrogen,
                hydrogen_count=hydrogen_count,
                candidate_energies=tuple(candidate_energies),
                best_structure_path=best_structure_path,
            )
        )

    interaction_parameter, intercept = linear_fit(
        tuple(point.coverage for point in points),
        tuple(point.adsorption_energy_per_adsorbate for point in points),
    )
    figure = plot_coverage_dependence(points, intercept, interaction_parameter)
    figure_path = artifacts.write_figure(
        figure,
        PART_DIRECTORY / "coverage_dependence.png",
    )
    import matplotlib.pyplot as plt

    plt.close(figure)
    result = CoverageResult(
        points=tuple(points),
        intercept=intercept,
        interaction_parameter=interaction_parameter,
        clean_slab_path=clean_slab_path,
        hydrogen_reference_path=hydrogen_reference_path,
        figure_path=figure_path,
    )
    artifacts.write_json(
        {
            "clean_slab_energy_ev": clean_energy.total_energy,
            "h2_energy_ev": h2_energy.total_energy,
            "intercept_ev_per_hydrogen": result.intercept,
            "interaction_parameter_ev_per_ml": result.interaction_parameter,
            "points": [
                {
                    "hydrogen_count": point.hydrogen_count,
                    "coverage_ml": point.coverage,
                    "adsorption_energy_ev_per_hydrogen": (
                        point.adsorption_energy_per_adsorbate
                    ),
                    "candidate_energies_ev": [
                        energy.total_energy for energy in point.candidate_energies
                    ],
                    "best_structure": str(point.best_structure_path),
                }
                for point in result.points
            ],
            "figure": str(figure_path),
        },
        PART_DIRECTORY / "result.json",
    )
    return result
