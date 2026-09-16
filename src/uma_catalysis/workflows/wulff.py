"""Construct Wulff morphologies from typed bulk and surface-energy results."""

from pathlib import Path
from typing import Any

from uma_catalysis.artifacts import ArtifactStore
from uma_catalysis.calculations import build_bulk_structure
from uma_catalysis.evaluation import plot_wulff_shape
from uma_catalysis.structs.config import MaterialConfig
from uma_catalysis.structs.results import (
    BulkOptimizationResult,
    SurfaceEnergyStudyResult,
    WulffResult,
)

PART_DIRECTORY = Path("part3-wulff-construction")


def _wulff_lattice(
    bulk_result: BulkOptimizationResult, material: MaterialConfig
) -> Any:
    """Return a Pymatgen lattice based on the optimized bulk result."""
    from pymatgen.io.ase import AseAtomsAdaptor

    bulk_atoms = bulk_result.relaxed_atoms
    if bulk_atoms is None:
        bulk_atoms = build_bulk_structure(
            material,
            lattice_constant=bulk_result.optimized_lattice_constant,
        )
    return AseAtomsAdaptor().get_structure(bulk_atoms).lattice


def run_wulff_construction(
    bulk_result: BulkOptimizationResult,
    surface_study: SurfaceEnergyStudyResult,
    material: MaterialConfig,
    artifacts: ArtifactStore,
) -> WulffResult:
    """
    Construct and persist a Wulff morphology from calculated surface energies.

    Parameters
    ----------
    bulk_result : uma_catalysis.structs.results.BulkOptimizationResult
        Bulk result that supplies the optimized crystal lattice.
    surface_study : uma_catalysis.structs.results.SurfaceEnergyStudyResult
        Typed facet energies used to construct the Wulff morphology.
    material : uma_catalysis.structs.config.MaterialConfig
        Chemical element label used in the output figure.
    artifacts : uma_catalysis.artifacts.ArtifactStore
        Controlled destination for the morphology figure and metadata.

    Returns
    -------
    uma_catalysis.structs.results.WulffResult
        Morphology metrics, per-facet area fractions, and saved figure path.
    """
    from pymatgen.analysis.wulff import WulffShape

    facet_results = surface_study.facet_results
    facets = [result.facet for result in facet_results]
    energies = [result.surface_energy_j_per_m2 for result in facet_results]
    wulff_shape = WulffShape(_wulff_lattice(bulk_result, material), facets, energies)
    area_fractions = tuple(
        (facet, float(wulff_shape.area_fraction_dict.get(facet, 0.0)))
        for facet in facets
    )
    figure = plot_wulff_shape(wulff_shape, material.element)
    figure_path = artifacts.write_figure(
        figure,
        PART_DIRECTORY / "wulff_shape.png",
    )
    import matplotlib.pyplot as plt

    plt.close(figure)
    result = WulffResult(
        volume=float(wulff_shape.volume),
        surface_area=float(wulff_shape.surface_area),
        effective_radius=float(wulff_shape.effective_radius),
        weighted_surface_energy=float(wulff_shape.weighted_surface_energy),
        facet_area_fractions=area_fractions,
        figure_path=figure_path,
    )
    artifacts.write_json(
        {
            "volume_angstrom_cubed": result.volume,
            "surface_area_angstrom_squared": result.surface_area,
            "effective_radius_angstrom": result.effective_radius,
            "weighted_surface_energy_j_per_m2": result.weighted_surface_energy,
            "facet_area_fractions": [
                {"facet": list(facet), "fraction": fraction}
                for facet, fraction in result.facet_area_fractions
            ],
            "figure": str(figure_path),
        },
        PART_DIRECTORY / "result.json",
    )
    return result
