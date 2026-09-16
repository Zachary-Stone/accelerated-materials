"""Calculate low-coverage H adsorption on the tutorial Ni surface."""

from pathlib import Path
from typing import Any

from uma_catalysis.artifacts import ArtifactStore
from uma_catalysis.calculations import (
    build_adsorption_slab,
    build_d3_calculator,
    build_diatomic_reference,
    calculate_vibrational_zpe,
    evaluate_energy_components,
    generate_single_adsorbate_candidates,
    relax_with_uma,
)
from uma_catalysis.evaluation import plot_adsorption_structure
from uma_catalysis.structs.config import ComputeConfig, MaterialConfig, ModelConfig
from uma_catalysis.structs.results import (
    AdsorptionResult,
    BulkOptimizationResult,
    EnergyComponents,
    HydrogenAdsorptionResult,
)

PART_DIRECTORY = Path("part4-h-adsorption")


def _adsorbate_indices(atoms: Any) -> list[int]:
    """Return FAIR Chemistry adsorbate atom indices identified by tag 2."""
    return [index for index, tag in enumerate(atoms.get_tags()) if tag == 2]


def _hydrogen_zpe_correction(
    adsorbed_atoms: Any,
    h2_atoms: Any,
    vibration_directory: Path,
) -> tuple[float, float, float]:
    """Calculate H* and H2 ZPE values plus their adsorption correction."""
    hydrogen_indices = _adsorbate_indices(adsorbed_atoms)
    if len(hydrogen_indices) != 1:
        raise ValueError("Expected exactly one tagged H adsorbate atom.")
    adsorbate_zpe = calculate_vibrational_zpe(
        adsorbed_atoms,
        indices=hydrogen_indices,
        delta=0.02,
        name=vibration_directory / "h_adsorbate",
    )
    reference_zpe = calculate_vibrational_zpe(
        h2_atoms,
        indices=(0, 1),
        delta=0.02,
        name=vibration_directory / "h2",
    )
    return adsorbate_zpe, reference_zpe, adsorbate_zpe - 0.5 * reference_zpe


def run_hydrogen_adsorption(
    predictor: Any,
    bulk_result: BulkOptimizationResult,
    model: ModelConfig,
    material: MaterialConfig,
    compute: ComputeConfig,
    artifacts: ArtifactStore,
) -> HydrogenAdsorptionResult:
    """
    Calculate the lowest-energy H adsorption configuration and optional ZPE.

    Parameters
    ----------
    predictor : Any
        FAIR Chemistry prediction unit used to construct ``oc20`` calculators.
    bulk_result : uma_catalysis.structs.results.BulkOptimizationResult
        Bulk optimization result that supplies the adsorption-slab lattice.
    model : uma_catalysis.structs.config.ModelConfig
        D3 endpoint-correction settings.
    material : uma_catalysis.structs.config.MaterialConfig
        Adsorption facet and vacuum settings.
    compute : uma_catalysis.structs.config.ComputeConfig
        Candidate count, relaxation budget, and ZPE selection.
    artifacts : uma_catalysis.artifacts.ArtifactStore
        Controlled destination for all generated structures and figures.

    Returns
    -------
    uma_catalysis.structs.results.HydrogenAdsorptionResult
        Best adsorption result, candidate energies, reference artifacts, and
        optional ZPE values.
    """
    part_directory = artifacts.directory(PART_DIRECTORY)
    d3_calculator = build_d3_calculator(model)
    adsorption_slab = build_adsorption_slab(
        material,
        lattice_constant=bulk_result.optimized_lattice_constant,
    )
    clean_outcome = relax_with_uma(
        adsorption_slab.atoms.copy(),
        predictor,
        task_name="oc20",
        force_threshold=compute.relaxation_force_threshold,
        steps=compute.relaxation_steps,
        trajectory_path=part_directory / "ni111_clean.traj",
        logfile_path=part_directory / "ni111_clean.log",
    )
    clean_energy = evaluate_energy_components(clean_outcome.atoms, d3_calculator)
    clean_slab_path = artifacts.write_atoms(
        clean_outcome.atoms,
        PART_DIRECTORY / "ni111_clean.xyz",
    )
    adsorption_slab.atoms = clean_outcome.atoms.copy()

    candidates = generate_single_adsorbate_candidates(
        adsorption_slab,
        adsorbate_smiles="*H",
        count=compute.adsorption_candidate_count,
    )
    if not candidates:
        raise RuntimeError("No H adsorption candidates were generated.")

    candidate_energies: list[EnergyComponents] = []
    candidate_structures = []
    for candidate_number, candidate in enumerate(candidates, start=1):
        candidate.set_pbc([True, True, True])
        artifact_stem = f"h_site_{candidate_number}"
        outcome = relax_with_uma(
            candidate.copy(),
            predictor,
            task_name="oc20",
            force_threshold=compute.relaxation_force_threshold,
            steps=compute.relaxation_steps,
            trajectory_path=part_directory / f"{artifact_stem}.traj",
            logfile_path=part_directory / f"{artifact_stem}.log",
        )
        energy = evaluate_energy_components(outcome.atoms, d3_calculator)
        candidate_energies.append(energy)
        candidate_structures.append(outcome.atoms)
        artifacts.write_atoms(outcome.atoms, PART_DIRECTORY / f"{artifact_stem}.xyz")

    best_candidate_index = min(
        range(len(candidate_energies)),
        key=lambda index: candidate_energies[index].total_energy,
    )
    best_atoms = candidate_structures[best_candidate_index]
    best_energy = candidate_energies[best_candidate_index]
    best_structure_path = artifacts.write_atoms(
        best_atoms,
        PART_DIRECTORY / "h_best.xyz",
    )

    h2 = build_diatomic_reference(
        symbols=("H", "H"),
        bond_length=0.74,
        vacuum_size=material.vacuum_size,
    )
    h2_outcome = relax_with_uma(
        h2,
        predictor,
        task_name="oc20",
        force_threshold=compute.relaxation_force_threshold,
        steps=compute.relaxation_steps,
        trajectory_path=part_directory / "h2.traj",
        logfile_path=part_directory / "h2.log",
    )
    h2_energy = evaluate_energy_components(h2_outcome.atoms, d3_calculator)
    hydrogen_reference_path = artifacts.write_atoms(
        h2_outcome.atoms,
        PART_DIRECTORY / "h2_optimized.xyz",
    )

    adsorbate_zpe = None
    reference_zpe = None
    zpe_correction = None
    if compute.calculate_zpe:
        vibration_directory = artifacts.directory(PART_DIRECTORY / "vibrations")
        adsorbate_zpe, reference_zpe, zpe_correction = _hydrogen_zpe_correction(
            best_atoms,
            h2_outcome.atoms,
            vibration_directory,
        )

    adsorption = AdsorptionResult(
        adsorbate="H*",
        adsorbed_energy=best_energy,
        clean_slab_energy=clean_energy,
        reference_energy=h2_energy,
        reference_multiplier=0.5,
        zero_point_correction=zpe_correction,
        structure_path=best_structure_path,
    )
    figure = plot_adsorption_structure(best_atoms, "Lowest-Energy H* Configuration")
    visualization_path = artifacts.write_figure(
        figure,
        PART_DIRECTORY / "h_best.png",
    )
    import matplotlib.pyplot as plt

    plt.close(figure)
    result = HydrogenAdsorptionResult(
        adsorption=adsorption,
        candidate_energies=tuple(candidate_energies),
        best_candidate_number=best_candidate_index + 1,
        clean_slab_path=clean_slab_path,
        hydrogen_reference_path=hydrogen_reference_path,
        visualization_path=visualization_path,
        adsorbate_zpe=adsorbate_zpe,
        reference_zpe=reference_zpe,
    )
    artifacts.write_json(
        {
            "best_candidate_number": result.best_candidate_number,
            "candidate_energies_ev": [
                energy.total_energy for energy in result.candidate_energies
            ],
            "clean_slab_energy_ev": clean_energy.total_energy,
            "h2_energy_ev": h2_energy.total_energy,
            "electronic_adsorption_energy_ev": adsorption.electronic_adsorption_energy,
            "zpe_correction_ev": zpe_correction,
            "total_adsorption_energy_ev": adsorption.total_adsorption_energy,
            "adsorbate_zpe_ev": adsorbate_zpe,
            "reference_zpe_ev": reference_zpe,
            "best_structure": str(best_structure_path),
            "visualization": str(visualization_path),
        },
        PART_DIRECTORY / "result.json",
    )
    return result
