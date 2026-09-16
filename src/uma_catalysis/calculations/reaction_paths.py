"""Prepare and optimize optional reaction paths with ASE DyNEB."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from uma_catalysis.calculations.calculators import build_uma_calculator


@dataclass(slots=True)
class NEBOutcome:
    """Store an optimized NEB image chain and its relative ML energies."""

    images: list[Any]
    converged: bool
    steps: int
    relative_energies: tuple[float, ...]

    @property
    def forward_barrier(self) -> float:
        """Return the barrier relative to the first image in eV."""
        return max(self.relative_energies)

    @property
    def reverse_barrier(self) -> float:
        """Return the barrier relative to the final image in eV."""
        return self.forward_barrier - self.relative_energies[-1]


def prepare_stretched_co_guess(
    final_co: Any,
    predictor: Any,
    force_threshold: float,
    steps: int,
    constrained_trajectory_path: Path | None = None,
    unconstrained_trajectory_path: Path | None = None,
) -> Any:
    """Create the notebook's stretched-C--O local-minimum starting structure."""
    from ase.constraints import FixBondLengths

    adsorbate_indices = [
        index for index, tag in enumerate(final_co.get_tags()) if tag == 2
    ]
    if len(adsorbate_indices) != 2:
        raise ValueError("Expected exactly two tagged CO adsorbate atoms.")

    stretched = final_co.copy()
    slab = stretched[[tag != 2 for tag in stretched.get_tags()]]
    co_adsorbate = stretched[[tag == 2 for tag in stretched.get_tags()]]
    co_adsorbate.rotate(30.0, "x", center=co_adsorbate.positions[0])
    stretched = slab + co_adsorbate
    stretched.set_pbc([True, True, True])
    stretched_indices = [
        index for index, tag in enumerate(stretched.get_tags()) if tag == 2
    ]
    if len(stretched_indices) != 2:
        raise RuntimeError("Could not preserve tagged CO atoms in the NEB guess.")

    stretched.calc = build_uma_calculator(predictor, task_name="oc20")
    stretched.set_constraint(
        FixBondLengths(
            [stretched_indices],
            tolerance=1e-2,
            iterations=5000,
            bondlengths=[2.0],
        )
    )
    from ase.optimize import LBFGS

    optimizer = LBFGS(
        stretched,
        trajectory=(
            None
            if constrained_trajectory_path is None
            else str(constrained_trajectory_path)
        ),
        logfile=None,
    )
    try:
        optimizer.run(fmax=force_threshold, steps=steps)
    except RuntimeError:
        pass

    stretched.set_constraint()
    stretched.calc = build_uma_calculator(predictor, task_name="oc20")
    optimizer = LBFGS(
        stretched,
        trajectory=(
            None
            if unconstrained_trajectory_path is None
            else str(unconstrained_trajectory_path)
        ),
        logfile=None,
    )
    optimizer.run(fmax=force_threshold, steps=steps)
    return stretched


def run_dyneb(
    initial: Any,
    final: Any,
    predictor: Any,
    intermediate_images: int,
    force_threshold: float,
    steps: int,
    trajectory_path: Path | None = None,
    logfile_path: Path | None = None,
) -> NEBOutcome:
    """Interpolate and optimize a climbing-image DyNEB path using OC20."""
    if intermediate_images < 1:
        raise ValueError("intermediate_images must be at least 1.")
    if force_threshold <= 0:
        raise ValueError("force_threshold must be positive.")
    if steps < 1:
        raise ValueError("steps must be at least 1.")

    from ase.mep.dyneb import DyNEB
    from ase.optimize import FIRE

    images = [initial.copy()]
    images.extend(initial.copy() for _ in range(intermediate_images))
    images.append(final.copy())
    for image in images:
        image.calc = build_uma_calculator(predictor, task_name="oc20")

    neb = DyNEB(images, climb=True, fmax=force_threshold)
    neb.interpolate("idpp", mic=True)
    optimizer = FIRE(
        neb,
        trajectory=None if trajectory_path is None else str(trajectory_path),
        logfile=None if logfile_path is None else str(logfile_path),
    )
    converged = bool(optimizer.run(fmax=force_threshold, steps=steps))
    energies = [float(image.get_potential_energy()) for image in images]
    initial_energy = energies[0]
    return NEBOutcome(
        images=images,
        converged=converged,
        steps=optimizer.get_number_of_steps(),
        relative_energies=tuple(energy - initial_energy for energy in energies),
    )
