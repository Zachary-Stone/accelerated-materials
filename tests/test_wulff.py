"""Fast integration test for Wulff construction without model inference."""

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

from uma_catalysis.artifacts import ArtifactStore
from uma_catalysis.structs import (
    BulkOptimizationResult,
    MaterialConfig,
    SurfaceEnergyResult,
    SurfaceEnergyStudyResult,
)
from uma_catalysis.workflows.wulff import run_wulff_construction


def _surface_energy_result(
    facet: tuple[int, int, int], energy: float
) -> SurfaceEnergyResult:
    """Return a minimal valid surface-energy fit result for testing."""
    return SurfaceEnergyResult(
        facet=facet,
        atom_counts=(4, 6, 8),
        slab_energies=(-10.0, -14.0, -18.0),
        fit_slope=-2.0,
        fit_intercept=2.0,
        surface_energy_ev_per_angstrom_squared=energy / 16.0218,
        surface_energy_j_per_m2=energy,
    )


class WulffWorkflowTests(unittest.TestCase):
    """Validate Wulff construction from typed precursor results."""

    def test_wulff_workflow_writes_a_figure_and_metadata(self) -> None:
        """Construct a morphology without FAIR Chemistry model inference."""
        material = MaterialConfig()
        bulk_result = BulkOptimizationResult(3.52, 3.52, 3.524, converged=True)
        surface_study = SurfaceEnergyStudyResult(
            bulk_energy_per_atom=-5.0,
            facet_results=(
                _surface_energy_result((1, 1, 1), 1.92),
                _surface_energy_result((1, 0, 0), 2.21),
                _surface_energy_result((1, 1, 0), 2.29),
                _surface_energy_result((2, 1, 1), 2.24),
            ),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            result = run_wulff_construction(
                bulk_result,
                surface_study,
                material,
                ArtifactStore(Path(temporary_directory)),
            )

            self.assertIsNotNone(result.figure_path)
            self.assertTrue(result.figure_path.is_file())
            self.assertEqual(len(result.facet_area_fractions), 4)
            self.assertAlmostEqual(
                sum(fraction for _, fraction in result.facet_area_fractions),
                1.0,
            )
