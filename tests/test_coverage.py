"""Tests for coverage-dependent H adsorption without model inference."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from uma_catalysis.artifacts import ArtifactStore
from uma_catalysis.calculations.relaxation import RelaxationOutcome
from uma_catalysis.structs.config import ComputeConfig, MaterialConfig, ModelConfig
from uma_catalysis.structs.results import BulkOptimizationResult, EnergyComponents
from uma_catalysis.workflows.coverage import run_coverage_study


class FakeAtoms:
    """Small ASE-like object for exercising workflow orchestration."""

    def __init__(self, label: str, tags: tuple[int, ...] = ()) -> None:
        self.label = label
        self._tags = tags
        self.pbc = None

    def copy(self) -> FakeAtoms:
        atoms_copy = FakeAtoms(self.label, self._tags)
        atoms_copy.pbc = self.pbc
        return atoms_copy

    def get_tags(self) -> tuple[int, ...]:
        return self._tags

    def set_pbc(self, value: object) -> None:
        self.pbc = value


class FakeSlab:
    def __init__(self, atoms: FakeAtoms) -> None:
        self.atoms = atoms


class FakeFigure:
    def savefig(self, filename: Path, **_: object) -> None:
        Path(filename).touch()


class CoverageWorkflowTest(unittest.TestCase):
    def test_fits_fast_mode_coverage_series(self) -> None:
        clean = FakeAtoms("clean", tags=(1, 1, 1, 1))
        reference = FakeAtoms("h2")
        energies = {
            "clean": EnergyComponents(ml_energy=-8.0),
            "h2": EnergyComponents(ml_energy=-2.0),
            "n1-low": EnergyComponents(ml_energy=-9.875),
            "n1-high": EnergyComponents(ml_energy=-9.5),
            "n4-low": EnergyComponents(ml_energy=-14.0),
            "n4-high": EnergyComponents(ml_energy=-13.5),
            "n8-low": EnergyComponents(ml_energy=-16.0),
            "n8-high": EnergyComponents(ml_energy=-15.5),
        }
        bulk_result = BulkOptimizationResult(
            initial_lattice_constant=3.5,
            optimized_lattice_constant=3.5,
            experimental_lattice_constant=3.5,
            converged=True,
        )

        def relaxed(atoms: FakeAtoms, *_: object, **__: object) -> RelaxationOutcome:
            if atoms.label == "template":
                atoms = clean.copy()
            return RelaxationOutcome(atoms=atoms, converged=True, steps=1)

        def candidates(
            _: FakeSlab, adsorbate_smiles: tuple[str, ...], **__: object
        ) -> list[FakeAtoms]:
            hydrogen_count = len(adsorbate_smiles)
            return [
                FakeAtoms(f"n{hydrogen_count}-high"),
                FakeAtoms(f"n{hydrogen_count}-low"),
            ]

        def endpoint_energy(
            atoms: FakeAtoms, *_: object, **__: object
        ) -> EnergyComponents:
            return energies[atoms.label]

        def fake_write_atoms(
            store: ArtifactStore, _: FakeAtoms, relative_path: Path
        ) -> Path:
            return store.path(relative_path)

        with tempfile.TemporaryDirectory() as temporary_directory:
            artifacts = ArtifactStore(Path(temporary_directory))
            with (
                patch(
                    "uma_catalysis.workflows.coverage.build_d3_calculator",
                    return_value=object(),
                ),
                patch(
                    "uma_catalysis.workflows.coverage.build_adsorption_slab",
                    side_effect=lambda *_args, **_kwargs: FakeSlab(
                        FakeAtoms("template")
                    ),
                ),
                patch(
                    "uma_catalysis.workflows.coverage.build_diatomic_reference",
                    return_value=reference,
                ),
                patch(
                    "uma_catalysis.workflows.coverage.generate_multiple_adsorbate_candidates",
                    side_effect=candidates,
                ),
                patch(
                    "uma_catalysis.workflows.coverage._relax_oc20",
                    side_effect=relaxed,
                ),
                patch(
                    "uma_catalysis.workflows.coverage.evaluate_energy_components",
                    side_effect=endpoint_energy,
                ),
                patch(
                    "uma_catalysis.workflows.coverage.plot_coverage_dependence",
                    return_value=FakeFigure(),
                ),
                patch.object(ArtifactStore, "write_atoms", new=fake_write_atoms),
                patch("matplotlib.pyplot.close"),
            ):
                result = run_coverage_study(
                    predictor=object(),
                    bulk_result=bulk_result,
                    model=ModelConfig(),
                    material=MaterialConfig(),
                    compute=ComputeConfig(fast_mode=True),
                    artifacts=artifacts,
                )

            self.assertEqual(
                [point.hydrogen_count for point in result.points], [1, 4, 8]
            )
            self.assertEqual(
                [point.coverage for point in result.points], [0.25, 1.0, 2.0]
            )
            self.assertAlmostEqual(result.intercept, -1.0)
            self.assertAlmostEqual(result.interaction_parameter, 0.5)
            self.assertTrue(result.figure_path.is_file())


if __name__ == "__main__":
    unittest.main()
