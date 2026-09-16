"""Generate figures for UMA catalysis workflow results."""

from collections.abc import Sequence
from math import ceil
from typing import Any

from uma_catalysis.structs.results import CoveragePoint, SurfaceEnergyResult


def plot_wulff_shape(wulff_shape: Any, element: str) -> Any:
    """
    Plot a Pymatgen Wulff shape and return its Matplotlib figure.

    Parameters
    ----------
    wulff_shape : object
        Pymatgen ``WulffShape`` object exposing ``get_plot``.
    element : str
        Chemical-symbol label displayed in the figure title.

    Returns
    -------
    matplotlib.figure.Figure
        Wulff morphology figure ready to save or display.

    Raises
    ------
    ValueError
        If the element label is empty.
    """
    if not element.strip():
        raise ValueError("element must not be empty.")

    axis = wulff_shape.get_plot()
    axis.set_title(f"Wulff Construction: {element} Nanoparticle")
    figure = axis.get_figure()
    figure.tight_layout()
    return figure


def plot_adsorption_structure(atoms: Any, title: str) -> Any:
    """
    Plot an ASE structure using the tutorial's isometric viewing orientation.

    Parameters
    ----------
    atoms : Any
        ASE-compatible structure to visualize.
    title : str
        Figure title describing the optimized adsorption configuration.

    Returns
    -------
    matplotlib.figure.Figure
        Structure figure ready to save or display.

    Raises
    ------
    ValueError
        If the title is empty.
    """
    if not title.strip():
        raise ValueError("title must not be empty.")

    import matplotlib.pyplot as plt
    from ase.visualize.plot import plot_atoms

    figure, axis = plt.subplots(figsize=(7, 6))
    plot_atoms(atoms, axis, rotation="-45x,0y,0z")
    axis.set_title(title)
    axis.set_axis_off()
    figure.tight_layout()
    return figure


def plot_coverage_dependence(
    points: Sequence[CoveragePoint], intercept: float, interaction_parameter: float
) -> Any:
    """
    Plot coverage-resolved adsorption energies and their linear fit.

    Parameters
    ----------
    points : collections.abc.Sequence[CoveragePoint]
        Calculated coverages and average adsorption energies.
    intercept : float
        Extrapolated zero-coverage adsorption energy in eV.
    interaction_parameter : float
        Linear coverage coefficient in eV per monolayer.

    Returns
    -------
    matplotlib.figure.Figure
        Coverage-dependence figure ready to save or display.

    Raises
    ------
    ValueError
        If fewer than two calculated coverage points are provided.
    """
    if len(points) < 2:
        raise ValueError("points must contain at least two coverage values.")

    import matplotlib.pyplot as plt

    coverages = [point.coverage for point in points]
    energies = [point.adsorption_energy_per_adsorbate for point in points]
    maximum_coverage = max(coverages)
    fit_coverages = [maximum_coverage * index / 99 for index in range(100)]
    fit_energies = [
        intercept + interaction_parameter * coverage for coverage in fit_coverages
    ]
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(coverages, energies, s=100, label="Calculated")
    axis.plot(fit_coverages, fit_energies, "--", label="Linear fit")
    axis.set_xlabel("H coverage (ML)")
    axis.set_ylabel("Adsorption energy (eV/H)")
    axis.set_title("Coverage-dependent H adsorption on Ni(111)")
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    return figure


def plot_surface_energy_fits(
    results: Sequence[SurfaceEnergyResult], element: str
) -> Any:
    """
    Plot relaxed slab energies and linear fits for each calculated facet.

    Parameters
    ----------
    results : collections.abc.Sequence[SurfaceEnergyResult]
        Surface-energy results to visualize.
    element : str
        Chemical-symbol label displayed in facet titles.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the calculated points and linear fit for each facet.

    Raises
    ------
    ValueError
        If no results or no element label is supplied.
    """
    if not results:
        raise ValueError("results must not be empty.")
    if not element.strip():
        raise ValueError("element must not be empty.")

    import matplotlib.pyplot as plt

    column_count = min(2, len(results))
    row_count = ceil(len(results) / column_count)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(6 * column_count, 5 * row_count),
        squeeze=False,
    )
    flat_axes = axes.flatten()

    for axis, result in zip(flat_axes, results):
        atom_counts = result.atom_counts
        slab_energies = result.slab_energies
        axis.scatter(atom_counts, slab_energies, s=80, label="Calculated")
        lower_bound = min(atom_counts) - 2
        upper_bound = max(atom_counts) + 2
        fit_x = [
            lower_bound + (upper_bound - lower_bound) * index / 99
            for index in range(100)
        ]
        fit_y = [result.fit_slope * value + result.fit_intercept for value in fit_x]
        axis.plot(fit_x, fit_y, "--", linewidth=2, label="Linear fit")
        facet = "".join(str(index) for index in result.facet)
        axis.set_xlabel("Number of atoms")
        axis.set_ylabel("Slab energy (eV)")
        axis.set_title(
            f"{element}({facet}): gamma = {result.surface_energy_j_per_m2:.2f} J/m²"
        )
        axis.legend()
        axis.grid(True, alpha=0.3)

    for axis in flat_axes[len(results) :]:
        axis.remove()
    figure.tight_layout()
    return figure
