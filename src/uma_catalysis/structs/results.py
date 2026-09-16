"""Define typed scientific results produced by UMA catalysis workflows."""

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any

from uma_catalysis.structs.config import Facet


def _validate_finite(value: float, name: str) -> None:
    """Raise a ValueError when a numerical result is not finite."""
    if not isfinite(value):
        raise ValueError(f"{name} must be finite.")


def _validate_facet(facet: Facet, name: str) -> None:
    """Raise a ValueError when a Miller-index tuple is malformed."""
    if len(facet) != 3 or not all(isinstance(index, int) for index in facet):
        raise ValueError(f"{name} must contain exactly three integer Miller indices.")
    if not any(facet):
        raise ValueError(f"{name} must not be the zero Miller index.")


@dataclass(frozen=True, slots=True)
class EnergyComponents:
    """
    Store ML and D3 endpoint energy components in eV.

    Parameters
    ----------
    ml_energy : float
        Energy evaluated with the UMA calculator in eV.
    d3_energy : float, optional
        Endpoint D3 correction in eV. Default is 0.0.
    """

    ml_energy: float
    d3_energy: float = 0.0

    def __post_init__(self) -> None:
        """Validate energy components."""
        _validate_finite(self.ml_energy, "ml_energy")
        _validate_finite(self.d3_energy, "d3_energy")

    @property
    def total_energy(self) -> float:
        """
        Return the combined ML and D3 energy in eV.

        Returns
        -------
        float
            Sum of the ML and D3 energy components.
        """
        return self.ml_energy + self.d3_energy


@dataclass(frozen=True, slots=True)
class BulkOptimizationResult:
    """
    Store the result of bulk lattice optimization.

    Parameters
    ----------
    initial_lattice_constant : float
        Starting lattice constant in angstrom.
    optimized_lattice_constant : float
        Relaxed lattice constant in angstrom.
    experimental_lattice_constant : float
        Experimental comparison value in angstrom.
    converged : bool
        Whether the optimizer reported convergence.
    structure_path : pathlib.Path or None, optional
        Saved relaxed bulk structure, if written. Default is None.
    relaxed_atoms : Any or None, optional
        In-memory ASE-compatible relaxed structure for downstream workflows.
        Default is None. This field is excluded from value comparisons.
    """

    initial_lattice_constant: float
    optimized_lattice_constant: float
    experimental_lattice_constant: float
    converged: bool
    structure_path: Path | None = None
    relaxed_atoms: Any | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate lattice constants."""
        for name, value in (
            ("initial_lattice_constant", self.initial_lattice_constant),
            ("optimized_lattice_constant", self.optimized_lattice_constant),
            ("experimental_lattice_constant", self.experimental_lattice_constant),
        ):
            _validate_finite(value, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive.")

    @property
    def relative_error_percent(self) -> float:
        """
        Return the optimized lattice-constant error relative to experiment.

        Returns
        -------
        float
            Absolute percentage error relative to the experimental reference.
        """
        return (
            abs(self.optimized_lattice_constant - self.experimental_lattice_constant)
            / self.experimental_lattice_constant
            * 100.0
        )


@dataclass(frozen=True, slots=True)
class SurfaceEnergyResult:
    """
    Store one facet's slab-energy fit and surface energy.

    Parameters
    ----------
    facet : Facet
        Surface Miller index.
    atom_counts : tuple[int, ...]
        Atom counts used in the linear fit.
    slab_energies : tuple[float, ...]
        Relaxed slab energies in eV corresponding to ``atom_counts``.
    fit_slope : float
        Fitted bulk-like energy per atom in eV/atom.
    fit_intercept : float
        Fitted two-surface intercept in eV.
    surface_energy_ev_per_angstrom_squared : float
        Surface energy in eV/angstrom-squared.
    surface_energy_j_per_m2 : float
        Surface energy in J/m-squared.
    """

    facet: Facet
    atom_counts: tuple[int, ...]
    slab_energies: tuple[float, ...]
    fit_slope: float
    fit_intercept: float
    surface_energy_ev_per_angstrom_squared: float
    surface_energy_j_per_m2: float

    def __post_init__(self) -> None:
        """Validate linear-fit and surface-energy data."""
        _validate_facet(self.facet, "facet")
        if len(self.atom_counts) < 2:
            raise ValueError("atom_counts must contain at least two entries.")
        if len(self.atom_counts) != len(self.slab_energies):
            raise ValueError("atom_counts and slab_energies must have equal length.")
        if any(count < 1 for count in self.atom_counts):
            raise ValueError("atom_counts must contain positive values.")
        for name, value in (
            ("fit_slope", self.fit_slope),
            ("fit_intercept", self.fit_intercept),
            (
                "surface_energy_ev_per_angstrom_squared",
                self.surface_energy_ev_per_angstrom_squared,
            ),
            ("surface_energy_j_per_m2", self.surface_energy_j_per_m2),
            *[("slab_energy", energy) for energy in self.slab_energies],
        ):
            _validate_finite(value, name)


@dataclass(frozen=True, slots=True)
class SurfaceEnergyStudyResult:
    """
    Store all facet results produced by one surface-energy workflow.

    Parameters
    ----------
    bulk_energy_per_atom : float
        Reference bulk energy in eV per atom.
    facet_results : tuple[SurfaceEnergyResult, ...]
        Surface-energy fit and energy result for each selected facet.
    figure_path : pathlib.Path or None, optional
        Saved linear-fit comparison figure, if written. Default is None.
    """

    bulk_energy_per_atom: float
    facet_results: tuple[SurfaceEnergyResult, ...]
    figure_path: Path | None = None

    def __post_init__(self) -> None:
        """Validate the common bulk reference and unique facet results."""
        _validate_finite(self.bulk_energy_per_atom, "bulk_energy_per_atom")
        if not self.facet_results:
            raise ValueError("facet_results must not be empty.")
        facets = tuple(result.facet for result in self.facet_results)
        if len(set(facets)) != len(facets):
            raise ValueError("facet_results must not contain duplicate facets.")


@dataclass(frozen=True, slots=True)
class WulffResult:
    """
    Store Wulff morphology metrics and facet-area fractions.

    Parameters
    ----------
    volume : float
        Wulff volume in angstrom-cubed.
    surface_area : float
        Wulff surface area in angstrom-squared.
    effective_radius : float
        Effective particle radius in angstrom.
    weighted_surface_energy : float
        Area-weighted surface energy in J/m-squared.
    facet_area_fractions : tuple[tuple[Facet, float], ...]
        Per-facet area fractions as immutable facet/fraction pairs.
    """

    volume: float
    surface_area: float
    effective_radius: float
    weighted_surface_energy: float
    facet_area_fractions: tuple[tuple[Facet, float], ...]

    def __post_init__(self) -> None:
        """Validate morphology metrics and fractional areas."""
        for name, value in (
            ("volume", self.volume),
            ("surface_area", self.surface_area),
            ("effective_radius", self.effective_radius),
            ("weighted_surface_energy", self.weighted_surface_energy),
        ):
            _validate_finite(value, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if not self.facet_area_fractions:
            raise ValueError("facet_area_fractions must not be empty.")
        for facet, fraction in self.facet_area_fractions:
            _validate_facet(facet, "facet_area_fractions")
            _validate_finite(fraction, "facet area fraction")
            if not 0.0 <= fraction <= 1.0:
                raise ValueError("facet area fractions must be between 0 and 1.")


@dataclass(frozen=True, slots=True)
class AdsorptionResult:
    """
    Store an adsorption-energy calculation and its optional ZPE correction.

    Parameters
    ----------
    adsorbate : str
        Label for the adsorbate or adsorbate state.
    adsorbed_energy : EnergyComponents
        Relaxed adsorbed-system energy.
    clean_slab_energy : EnergyComponents
        Clean-slab reference energy.
    reference_energy : EnergyComponents
        Gas-phase or elemental reference energy.
    reference_multiplier : float
        Stoichiometric multiplier applied to the reference energy.
    zero_point_correction : float or None, optional
        Additive ZPE correction in eV. Default is None.
    structure_path : pathlib.Path or None, optional
        Saved optimized adsorbate structure, if written. Default is None.
    """

    adsorbate: str
    adsorbed_energy: EnergyComponents
    clean_slab_energy: EnergyComponents
    reference_energy: EnergyComponents
    reference_multiplier: float
    zero_point_correction: float | None = None
    structure_path: Path | None = None

    def __post_init__(self) -> None:
        """Validate adsorption labels and energy multipliers."""
        if not self.adsorbate.strip():
            raise ValueError("adsorbate must not be empty.")
        _validate_finite(self.reference_multiplier, "reference_multiplier")
        if self.reference_multiplier <= 0:
            raise ValueError("reference_multiplier must be positive.")
        if self.zero_point_correction is not None:
            _validate_finite(self.zero_point_correction, "zero_point_correction")

    @property
    def electronic_adsorption_energy(self) -> float:
        """
        Return the electronic adsorption energy in eV.

        Returns
        -------
        float
            Adsorbed-system energy minus the clean slab and reference energies.
        """
        return (
            self.adsorbed_energy.total_energy
            - self.clean_slab_energy.total_energy
            - self.reference_multiplier * self.reference_energy.total_energy
        )

    @property
    def total_adsorption_energy(self) -> float:
        """
        Return the adsorption energy with its optional ZPE correction.

        Returns
        -------
        float
            Electronic adsorption energy plus the supplied ZPE correction.
        """
        return self.electronic_adsorption_energy + (self.zero_point_correction or 0.0)


@dataclass(frozen=True, slots=True)
class CoveragePoint:
    """
    Store one coverage and its average adsorption energy.

    Parameters
    ----------
    coverage : float
        Adsorbate coverage in monolayers.
    adsorption_energy_per_adsorbate : float
        Average adsorption energy in eV per adsorbate.
    """

    coverage: float
    adsorption_energy_per_adsorbate: float

    def __post_init__(self) -> None:
        """Validate coverage-study data."""
        _validate_finite(self.coverage, "coverage")
        _validate_finite(
            self.adsorption_energy_per_adsorbate, "adsorption_energy_per_adsorbate"
        )
        if self.coverage <= 0:
            raise ValueError("coverage must be positive.")


@dataclass(frozen=True, slots=True)
class CoverageResult:
    """
    Store a coverage series and its linear adsorption-energy fit.

    Parameters
    ----------
    points : tuple[CoveragePoint, ...]
        Calculated coverage points used in the fit.
    intercept : float
        Extrapolated zero-coverage adsorption energy in eV.
    interaction_parameter : float
        Linear coverage coefficient in eV per monolayer.
    """

    points: tuple[CoveragePoint, ...]
    intercept: float
    interaction_parameter: float

    def __post_init__(self) -> None:
        """Validate the fit inputs and outputs."""
        if len(self.points) < 2:
            raise ValueError("points must contain at least two coverage values.")
        _validate_finite(self.intercept, "intercept")
        _validate_finite(self.interaction_parameter, "interaction_parameter")


@dataclass(frozen=True, slots=True)
class ReactionResult:
    """
    Store reaction thermochemistry and optional NEB barrier data.

    Parameters
    ----------
    reaction_name : str
        Human-readable reaction identifier.
    initial_energy : EnergyComponents
        Energy of the initial state.
    final_energy : EnergyComponents
        Energy of the final state.
    zero_point_correction : float or None, optional
        Additive final-minus-initial ZPE correction in eV. Default is None.
    forward_barrier : float or None, optional
        NEB barrier from the initial state in eV. Default is None.
    reverse_barrier : float or None, optional
        NEB barrier from the final state in eV. Default is None.
    path_artifact_path : pathlib.Path or None, optional
        Saved NEB path artifact, if written. Default is None.
    """

    reaction_name: str
    initial_energy: EnergyComponents
    final_energy: EnergyComponents
    zero_point_correction: float | None = None
    forward_barrier: float | None = None
    reverse_barrier: float | None = None
    path_artifact_path: Path | None = None

    def __post_init__(self) -> None:
        """Validate reaction labels and optional energy values."""
        if not self.reaction_name.strip():
            raise ValueError("reaction_name must not be empty.")
        for name, value in (
            ("zero_point_correction", self.zero_point_correction),
            ("forward_barrier", self.forward_barrier),
            ("reverse_barrier", self.reverse_barrier),
        ):
            if value is not None:
                _validate_finite(value, name)
                if name.endswith("barrier") and value < 0:
                    raise ValueError(f"{name} must not be negative.")

    @property
    def electronic_reaction_energy(self) -> float:
        """
        Return the final-minus-initial electronic reaction energy in eV.

        Returns
        -------
        float
            Final-state energy minus initial-state energy.
        """
        return self.final_energy.total_energy - self.initial_energy.total_energy

    @property
    def total_reaction_energy(self) -> float:
        """
        Return the reaction energy with its optional ZPE correction.

        Returns
        -------
        float
            Electronic reaction energy plus the supplied ZPE correction.
        """
        return self.electronic_reaction_energy + (self.zero_point_correction or 0.0)
