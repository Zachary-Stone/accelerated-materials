"""Calculate reusable energy quantities with explicit units and inputs."""

from math import isfinite
from typing import Any

from uma_catalysis.structs.results import EnergyComponents

EV_PER_ANGSTROM_SQUARED_TO_J_PER_M2 = 16.0218


def _finite(value: float, name: str) -> float:
    """Validate and return a finite numerical input."""
    if not isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def evaluate_energy_components(atoms: Any, d3_calculator: Any) -> EnergyComponents:
    """
    Evaluate an ML energy and a D3 endpoint correction for one structure.

    The structure must already have a UMA calculator assigned. Its calculator
    is restored after the temporary D3 evaluation.

    Parameters
    ----------
    atoms : Any
        ASE-compatible atoms object with an ML calculator already assigned.
    d3_calculator : Any
        ASE-compatible D3 calculator used only for endpoint evaluation.

    Returns
    -------
    uma_catalysis.structs.results.EnergyComponents
        ML energy and D3 correction in eV.

    Raises
    ------
    ValueError
        If the structure has no calculator assigned.
    """
    ml_calculator = atoms.calc
    if ml_calculator is None:
        raise ValueError("atoms must have an ML calculator assigned.")

    ml_energy = float(atoms.get_potential_energy())
    atoms.calc = d3_calculator
    try:
        d3_energy = float(atoms.get_potential_energy())
    finally:
        atoms.calc = ml_calculator
    return EnergyComponents(ml_energy=ml_energy, d3_energy=d3_energy)


def surface_energy_from_intercept(
    fit_intercept: float, surface_area: float
) -> tuple[float, float]:
    """
    Calculate surface energy from a slab-energy linear-fit intercept.

    Parameters
    ----------
    fit_intercept : float
        Intercept of total slab energy versus atom count in eV. The intercept
        includes the energy of the slab's two exposed surfaces.
    surface_area : float
        Area of one slab face in angstrom-squared.

    Returns
    -------
    tuple[float, float]
        Surface energy in eV/angstrom-squared and J/m-squared, respectively.

    Raises
    ------
    ValueError
        If the surface area is not positive and finite, or the intercept is
        not finite.
    """
    _finite(fit_intercept, "fit_intercept")
    _finite(surface_area, "surface_area")
    if surface_area <= 0:
        raise ValueError("surface_area must be positive.")
    energy_ev_per_angstrom_squared = fit_intercept / (2.0 * surface_area)
    return (
        energy_ev_per_angstrom_squared,
        energy_ev_per_angstrom_squared * EV_PER_ANGSTROM_SQUARED_TO_J_PER_M2,
    )


def linear_fit(
    x_values: tuple[int, ...], y_values: tuple[float, ...]
) -> tuple[float, float]:
    """
    Fit a straight line by ordinary least squares without global state.

    Parameters
    ----------
    x_values : tuple[int, ...]
        Independent-variable values.
    y_values : tuple[float, ...]
        Dependent-variable values corresponding to ``x_values``.

    Returns
    -------
    tuple[float, float]
        Fitted slope and intercept, respectively.

    Raises
    ------
    ValueError
        If fewer than two points are supplied, the lengths differ, any value is
        non-finite, or all x values are identical.
    """
    if len(x_values) < 2:
        raise ValueError("x_values must contain at least two points.")
    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have equal length.")

    x_data = tuple(_finite(float(value), "x value") for value in x_values)
    y_data = tuple(_finite(value, "y value") for value in y_values)
    x_mean = sum(x_data) / len(x_data)
    y_mean = sum(y_data) / len(y_data)
    denominator = sum((value - x_mean) ** 2 for value in x_data)
    if denominator == 0.0:
        raise ValueError("x_values must not all be identical.")
    numerator = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_data, y_data, strict=True)
    )
    slope = numerator / denominator
    return slope, y_mean - slope * x_mean


def adsorption_energy(
    adsorbed_energy: EnergyComponents,
    clean_slab_energy: EnergyComponents,
    reference_energy: EnergyComponents,
    reference_multiplier: float,
) -> float:
    """
    Calculate an adsorption energy from explicitly supplied references.

    Parameters
    ----------
    adsorbed_energy : uma_catalysis.structs.results.EnergyComponents
        Energy of the relaxed slab-plus-adsorbate system.
    clean_slab_energy : uma_catalysis.structs.results.EnergyComponents
        Energy of the clean-slab reference.
    reference_energy : uma_catalysis.structs.results.EnergyComponents
        Energy of the gas-phase or elemental reference.
    reference_multiplier : float
        Stoichiometric multiplier applied to ``reference_energy``.

    Returns
    -------
    float
        Adsorption energy in eV, using the notebook's negative-is-binding
        convention.

    Raises
    ------
    ValueError
        If ``reference_multiplier`` is not positive and finite.
    """
    _finite(reference_multiplier, "reference_multiplier")
    if reference_multiplier <= 0:
        raise ValueError("reference_multiplier must be positive.")
    return (
        adsorbed_energy.total_energy
        - clean_slab_energy.total_energy
        - reference_multiplier * reference_energy.total_energy
    )


def reaction_energy(
    initial_energy: EnergyComponents, final_energy: EnergyComponents
) -> float:
    """
    Calculate the final-minus-initial electronic reaction energy.

    Parameters
    ----------
    initial_energy : uma_catalysis.structs.results.EnergyComponents
        Energy of the reaction initial state.
    final_energy : uma_catalysis.structs.results.EnergyComponents
        Energy of the reaction final state.

    Returns
    -------
    float
        Electronic reaction energy in eV.
    """
    return final_energy.total_energy - initial_energy.total_energy


def apply_zero_point_correction(electronic_energy: float, correction: float) -> float:
    """
    Add a zero-point-energy correction to an electronic energy difference.

    Parameters
    ----------
    electronic_energy : float
        Electronic energy difference in eV.
    correction : float
        Additive zero-point-energy correction in eV.

    Returns
    -------
    float
        Energy difference including the zero-point-energy correction in eV.
    """
    return _finite(electronic_energy, "electronic_energy") + _finite(
        correction, "correction"
    )
