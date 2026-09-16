"""Calculate CO thermochemistry and the optional C--O recombination NEB path."""

from dataclasses import replace
from pathlib import Path
from typing import Any

from uma_catalysis.artifacts import ArtifactStore
from uma_catalysis.calculations import (
    build_adsorption_slab,
    build_d3_calculator,
    build_diatomic_reference,
    calculate_vibrational_zpe,
    evaluate_energy_components,
    generate_multiple_adsorbate_candidates,
    prepare_stretched_co_guess,
    relax_with_uma,
    run_dyneb,
)
from uma_catalysis.evaluation import plot_neb_path
from uma_catalysis.structs.config import ComputeConfig, MaterialConfig, ModelConfig
from uma_catalysis.structs.results import (
    AdsorptionResult,
    BulkOptimizationResult,
    COReactionStudyResult,
    EnergyComponents,
    ReactionResult,
)

PART_DIRECTORY = Path("part6-co-reaction")
CO_RECOMBINATION_DISTANCE = 1.5


def _relax_oc20(
    atoms: Any,
    predictor: Any,
    compute: ComputeConfig,
    trajectory_path: Path | None,
    logfile_path: Path | None,
) -> Any:
    """Relax one CO-reaction structure using the shared OC20 helper."""
    return relax_with_uma(
        atoms,
        predictor,
        task_name="oc20",
        force_threshold=compute.relaxation_force_threshold,
        steps=compute.relaxation_steps,
        trajectory_path=trajectory_path,
        logfile_path=logfile_path,
    )


def _is_recombined_co(atoms: Any) -> bool:
    """Return whether two tagged adsorbates are closer than the CO cutoff."""
    indices = [index for index, tag in enumerate(atoms.get_tags()) if tag == 2]
    if len(indices) != 2:
        raise ValueError("Expected exactly two tagged C and O adsorbate atoms.")
    return (
        float(atoms.get_distance(indices[0], indices[1], mic=True))
        < CO_RECOMBINATION_DISTANCE
    )


def _best_adsorbate_configuration(
    adsorption_slab: Any,
    adsorbate_smiles: tuple[str, ...],
    label: str,
    predictor: Any,
    compute: ComputeConfig,
    d3_calculator: Any,
    artifacts: ArtifactStore,
    reject_recombined_co: bool = False,
) -> tuple[EnergyComponents, Any, Path]:
    """Relax generated configurations and return the lowest valid structure."""
    candidates = generate_multiple_adsorbate_candidates(
        adsorption_slab,
        adsorbate_smiles=adsorbate_smiles,
        count=compute.adsorption_candidate_count,
    )
    if not candidates:
        raise RuntimeError(f"No {label} adsorption configurations were generated.")

    energies: list[EnergyComponents] = []
    structures: list[Any] = []
    for candidate_number, candidate in enumerate(candidates, start=1):
        candidate.set_pbc([True, True, True])
        outcome = _relax_oc20(
            candidate.copy(),
            predictor,
            compute,
            trajectory_path=(
                artifacts.directory(PART_DIRECTORY)
                / f"{label}_{candidate_number:02d}.traj"
            ),
            logfile_path=None,
        )
        if reject_recombined_co and _is_recombined_co(outcome.atoms):
            continue
        energy = evaluate_energy_components(outcome.atoms, d3_calculator)
        energies.append(energy)
        structures.append(outcome.atoms)

    if not energies:
        raise RuntimeError(
            "No valid C* + O* configurations remained after rejecting recombined CO."
        )
    best_index = min(
        range(len(energies)), key=lambda index: energies[index].total_energy
    )
    best_path = artifacts.write_atoms(
        structures[best_index],
        PART_DIRECTORY / f"{label}_best.xyz",
    )
    return energies[best_index], structures[best_index], best_path


def _adsorbate_indices(atoms: Any, expected_count: int) -> list[int]:
    """Return tagged adsorbate indices and validate their expected count."""
    indices = [index for index, tag in enumerate(atoms.get_tags()) if tag == 2]
    if len(indices) != expected_count:
        raise ValueError(f"Expected {expected_count} tagged adsorbate atoms.")
    return indices


def run_co_reaction_study(
    predictor: Any,
    bulk_result: BulkOptimizationResult,
    model: ModelConfig,
    material: MaterialConfig,
    compute: ComputeConfig,
    artifacts: ArtifactStore,
) -> COReactionStudyResult:
    """Calculate CO/C/O thermochemistry and optionally a C--O recombination NEB."""
    part_directory = artifacts.directory(PART_DIRECTORY)
    d3_calculator = build_d3_calculator(model)
    base_slab = build_adsorption_slab(
        material,
        lattice_constant=bulk_result.optimized_lattice_constant,
    )
    clean_outcome = _relax_oc20(
        base_slab.atoms.copy(),
        predictor,
        compute,
        trajectory_path=part_directory / "ni111_clean.traj",
        logfile_path=part_directory / "ni111_clean.log",
    )
    clean_energy = evaluate_energy_components(clean_outcome.atoms, d3_calculator)
    artifacts.write_atoms(clean_outcome.atoms, PART_DIRECTORY / "ni111_clean.xyz")

    def adsorption_slab() -> Any:
        slab = build_adsorption_slab(
            material,
            lattice_constant=bulk_result.optimized_lattice_constant,
        )
        slab.atoms = base_slab.atoms.copy()
        return slab

    co_energy, final_co, co_structure_path = _best_adsorbate_configuration(
        adsorption_slab(),
        ("*CO",),
        "co_final",
        predictor,
        compute,
        d3_calculator,
        artifacts,
    )
    c_o_energy, initial_c_o, c_o_structure_path = _best_adsorbate_configuration(
        adsorption_slab(),
        ("*C", "*O"),
        "c_o_initial",
        predictor,
        compute,
        d3_calculator,
        artifacts,
        reject_recombined_co=True,
    )
    carbon_energy, _, carbon_structure_path = _best_adsorbate_configuration(
        adsorption_slab(),
        ("*C",),
        "c",
        predictor,
        compute,
        d3_calculator,
        artifacts,
    )
    oxygen_energy, _, oxygen_structure_path = _best_adsorbate_configuration(
        adsorption_slab(),
        ("*O",),
        "o",
        predictor,
        compute,
        d3_calculator,
        artifacts,
    )

    co_gas = build_diatomic_reference(
        symbols=("C", "O"),
        bond_length=1.15,
        vacuum_size=material.vacuum_size,
    )
    co_gas_outcome = _relax_oc20(
        co_gas,
        predictor,
        compute,
        trajectory_path=part_directory / "co_gas.traj",
        logfile_path=None,
    )
    co_gas_energy = evaluate_energy_components(co_gas_outcome.atoms, d3_calculator)
    co_gas_path = artifacts.write_atoms(
        co_gas_outcome.atoms,
        PART_DIRECTORY / "co_gas.xyz",
    )

    reaction_zpe = None
    co_adsorption_zpe = None
    if compute.calculate_zpe:
        vibration_directory = artifacts.directory(PART_DIRECTORY / "vibrations")
        zpe_final_co = calculate_vibrational_zpe(
            final_co,
            indices=_adsorbate_indices(final_co, expected_count=2),
            delta=0.02,
            name=vibration_directory / "co_adsorbed",
        )
        zpe_initial_c_o = calculate_vibrational_zpe(
            initial_c_o,
            indices=_adsorbate_indices(initial_c_o, expected_count=2),
            delta=0.02,
            name=vibration_directory / "c_o_adsorbed",
        )
        zpe_co_gas = calculate_vibrational_zpe(
            co_gas_outcome.atoms,
            indices=(0, 1),
            delta=0.01,
            name=vibration_directory / "co_gas",
        )
        reaction_zpe = zpe_final_co - zpe_initial_c_o
        co_adsorption_zpe = zpe_final_co - zpe_co_gas

    reaction = ReactionResult(
        reaction_name="C* + O* -> CO*",
        initial_energy=c_o_energy,
        final_energy=co_energy,
        zero_point_correction=reaction_zpe,
    )
    co_adsorption = AdsorptionResult(
        adsorbate="CO*",
        adsorbed_energy=co_energy,
        clean_slab_energy=clean_energy,
        reference_energy=co_gas_energy,
        reference_multiplier=1.0,
        zero_point_correction=co_adsorption_zpe,
        structure_path=co_structure_path,
    )

    neb_figure_path = None
    if compute.run_neb:
        initial_guess = prepare_stretched_co_guess(
            final_co,
            predictor,
            force_threshold=compute.relaxation_force_threshold,
            steps=compute.relaxation_steps,
            constrained_trajectory_path=part_directory
            / "initial_guess_constrained.traj",
            unconstrained_trajectory_path=(
                part_directory / "initial_guess_unconstrained.traj"
            ),
        )
        neb = run_dyneb(
            initial_guess,
            final_co,
            predictor,
            intermediate_images=compute.neb_intermediate_images,
            force_threshold=compute.neb_force_threshold,
            steps=compute.neb_steps,
            trajectory_path=part_directory / "neb_optimization.traj",
            logfile_path=part_directory / "neb.log",
        )
        neb_path = artifacts.write_atoms(neb.images, PART_DIRECTORY / "neb_images.traj")
        figure = plot_neb_path(neb.relative_energies)
        neb_figure_path = artifacts.write_figure(
            figure, PART_DIRECTORY / "neb_path.png"
        )
        import matplotlib.pyplot as plt

        plt.close(figure)
        reaction = replace(
            reaction,
            forward_barrier=neb.forward_barrier,
            reverse_barrier=neb.reverse_barrier,
            path_artifact_path=neb_path,
        )

    result = COReactionStudyResult(
        reaction=reaction,
        co_adsorption=co_adsorption,
        clean_slab_energy=clean_energy,
        separate_carbon_energy=carbon_energy,
        separate_oxygen_energy=oxygen_energy,
        co_structure_path=co_structure_path,
        c_o_structure_path=c_o_structure_path,
        carbon_structure_path=carbon_structure_path,
        oxygen_structure_path=oxygen_structure_path,
        co_gas_path=co_gas_path,
        neb_figure_path=neb_figure_path,
    )
    artifacts.write_json(
        {
            "reaction_energy_electronic_ev": result.reaction.electronic_reaction_energy,
            "reaction_zpe_correction_ev": result.reaction.zero_point_correction,
            "reaction_energy_total_ev": result.reaction.total_reaction_energy,
            "co_adsorption_energy_electronic_ev": (
                result.co_adsorption.electronic_adsorption_energy
            ),
            "co_adsorption_zpe_correction_ev": (
                result.co_adsorption.zero_point_correction
            ),
            "co_adsorption_energy_total_ev": (
                result.co_adsorption.total_adsorption_energy
            ),
            "separate_carbon_energy_ev": result.separate_carbon_energy.total_energy,
            "separate_oxygen_energy_ev": result.separate_oxygen_energy.total_energy,
            "forward_barrier_ev": result.reaction.forward_barrier,
            "reverse_barrier_ev": result.reaction.reverse_barrier,
            "neb_path": (
                None
                if result.reaction.path_artifact_path is None
                else str(result.reaction.path_artifact_path)
            ),
            "neb_figure": None if neb_figure_path is None else str(neb_figure_path),
        },
        PART_DIRECTORY / "result.json",
    )
    return result
