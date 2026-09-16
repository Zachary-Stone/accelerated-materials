"""Optimize the tutorial bulk crystal and preserve reusable artifacts."""

from pathlib import Path
from typing import Any

from uma_catalysis.artifacts import ArtifactStore
from uma_catalysis.calculations import (
    build_bulk_structure,
    build_uma_calculator,
    relax_cell_and_positions,
)
from uma_catalysis.structs.config import ComputeConfig, MaterialConfig
from uma_catalysis.structs.results import BulkOptimizationResult

PART_DIRECTORY = Path("part1-bulk-optimization")


def run_bulk_optimization(
    predictor: Any,
    material: MaterialConfig,
    compute: ComputeConfig,
    artifacts: ArtifactStore,
) -> BulkOptimizationResult:
    """
    Relax the tutorial bulk crystal with UMA's bulk-material task.

    Parameters
    ----------
    predictor : Any
        FAIR Chemistry prediction unit used to construct the calculator.
    material : uma_catalysis.structs.config.MaterialConfig
        Bulk element, structure, and lattice-reference settings.
    compute : uma_catalysis.structs.config.ComputeConfig
        Relaxation force threshold and step budget.
    artifacts : uma_catalysis.artifacts.ArtifactStore
        Controlled destination for trajectory, log, structure, and metadata.

    Returns
    -------
    uma_catalysis.structs.results.BulkOptimizationResult
        Relaxed lattice constant, comparison data, artifact path, and in-memory
        structure for downstream workflows.
    """
    part_directory = artifacts.directory(PART_DIRECTORY)
    bulk_atoms = build_bulk_structure(material)
    bulk_atoms.calc = build_uma_calculator(predictor, task_name="omat")
    outcome = relax_cell_and_positions(
        bulk_atoms,
        force_threshold=compute.relaxation_force_threshold,
        steps=compute.relaxation_steps,
        trajectory_path=part_directory / "ni_bulk_opt.traj",
        logfile_path=part_directory / "ni_bulk_opt.log",
    )
    optimized_lattice_constant = float(outcome.atoms.get_cell()[0, 0])
    structure_path = artifacts.write_atoms(
        outcome.atoms,
        PART_DIRECTORY / "ni_bulk_relaxed.cif",
    )
    result = BulkOptimizationResult(
        initial_lattice_constant=material.initial_lattice_constant,
        optimized_lattice_constant=optimized_lattice_constant,
        experimental_lattice_constant=material.experimental_lattice_constant,
        converged=outcome.converged,
        structure_path=structure_path,
        relaxed_atoms=outcome.atoms,
    )
    artifacts.write_json(
        {
            "initial_lattice_constant_angstrom": result.initial_lattice_constant,
            "optimized_lattice_constant_angstrom": result.optimized_lattice_constant,
            "experimental_lattice_constant_angstrom": (
                result.experimental_lattice_constant
            ),
            "relative_error_percent": result.relative_error_percent,
            "converged": result.converged,
        },
        PART_DIRECTORY / "result.json",
    )
    return result
