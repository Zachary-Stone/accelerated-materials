"""Relax ASE structures and calculate optional vibrational ZPE corrections."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RelaxationOutcome:
    """
    Store a relaxed structure and optimizer completion details.

    Parameters
    ----------
    atoms : Any
        ASE-compatible atoms object after optimizer execution.
    converged : bool
        Whether the optimizer reported convergence at the requested threshold.
    steps : int
        Number of optimizer steps taken.
    """

    atoms: Any
    converged: bool
    steps: int


def _validate_relaxation_inputs(force_threshold: float, steps: int) -> None:
    """Validate common geometry-optimization settings."""
    if force_threshold <= 0:
        raise ValueError("force_threshold must be positive.")
    if steps < 1:
        raise ValueError("steps must be at least 1.")


def _run_lbfgs(
    optimizable: Any,
    force_threshold: float,
    steps: int,
    trajectory_path: Path | None,
    logfile_path: Path | None,
) -> tuple[bool, int]:
    """Run ASE LBFGS and return its convergence flag and step count."""
    from ase.optimize import LBFGS

    optimizer = LBFGS(
        optimizable,
        trajectory=None if trajectory_path is None else str(trajectory_path),
        logfile=None if logfile_path is None else str(logfile_path),
    )
    converged = bool(optimizer.run(fmax=force_threshold, steps=steps))
    return converged, optimizer.get_number_of_steps()


def relax_positions(
    atoms: Any,
    force_threshold: float,
    steps: int,
    trajectory_path: Path | None = None,
    logfile_path: Path | None = None,
) -> RelaxationOutcome:
    """
    Relax atom positions with ASE LBFGS while preserving the cell.

    Parameters
    ----------
    atoms : Any
        ASE-compatible atoms object with a calculator assigned.
    force_threshold : float
        Maximum force in eV/angstrom.
    steps : int
        Maximum optimizer steps.
    trajectory_path : pathlib.Path or None, optional
        Optional ASE trajectory destination. Default is None.
    logfile_path : pathlib.Path or None, optional
        Optional optimizer log destination. Default is None.

    Returns
    -------
    RelaxationOutcome
        Relaxed atoms and optimizer completion details.
    """
    _validate_relaxation_inputs(force_threshold, steps)
    converged, completed_steps = _run_lbfgs(
        atoms, force_threshold, steps, trajectory_path, logfile_path
    )
    return RelaxationOutcome(atoms=atoms, converged=converged, steps=completed_steps)


def relax_cell_and_positions(
    atoms: Any,
    force_threshold: float,
    steps: int,
    trajectory_path: Path | None = None,
    logfile_path: Path | None = None,
) -> RelaxationOutcome:
    """
    Relax a bulk structure's cell and atomic positions with ASE LBFGS.

    Parameters
    ----------
    atoms : Any
        ASE-compatible bulk atoms object with a calculator assigned.
    force_threshold : float
        Maximum force in eV/angstrom.
    steps : int
        Maximum optimizer steps.
    trajectory_path : pathlib.Path or None, optional
        Optional ASE trajectory destination. Default is None.
    logfile_path : pathlib.Path or None, optional
        Optional optimizer log destination. Default is None.

    Returns
    -------
    RelaxationOutcome
        Relaxed bulk atoms and optimizer completion details.
    """
    _validate_relaxation_inputs(force_threshold, steps)

    from ase.filters import ExpCellFilter

    cell_filter = ExpCellFilter(atoms)
    converged, completed_steps = _run_lbfgs(
        cell_filter, force_threshold, steps, trajectory_path, logfile_path
    )
    return RelaxationOutcome(atoms=atoms, converged=converged, steps=completed_steps)


def calculate_vibrational_zpe(
    atoms: Any,
    indices: Sequence[int],
    delta: float,
    name: Path,
    nfree: int = 2,
    cleanup: bool = True,
) -> float:
    """
    Calculate the positive-mode zero-point energy for selected atoms.

    Parameters
    ----------
    atoms : Any
        ASE-compatible atoms object with a calculator assigned.
    indices : collections.abc.Sequence[int]
        Atom indices included in the finite-displacement calculation.
    delta : float
        Cartesian displacement in angstrom.
    name : pathlib.Path
        Prefix for ASE vibrational-analysis files.
    nfree : int, optional
        Number of finite-displacement points per coordinate. Default is 2.
    cleanup : bool, optional
        Whether temporary vibrational files are removed after calculation.
        Default is True.

    Returns
    -------
    float
        Positive-mode zero-point energy in eV.

    Raises
    ------
    ValueError
        If the selected indices, displacement, or finite-difference setting is
        invalid.
    """
    if not indices:
        raise ValueError("indices must not be empty.")
    if delta <= 0:
        raise ValueError("delta must be positive.")
    if nfree not in {2, 4}:
        raise ValueError("nfree must be 2 or 4.")

    import numpy as np
    from ase.vibrations import Vibrations

    vibration = Vibrations(
        atoms,
        indices=list(indices),
        delta=delta,
        name=str(name),
        nfree=nfree,
    )
    try:
        vibration.run()
        energies = np.asarray(vibration.get_energies())
        return float(np.sum(energies[energies > 0]) / 2.0)
    finally:
        if cleanup:
            vibration.clean()
