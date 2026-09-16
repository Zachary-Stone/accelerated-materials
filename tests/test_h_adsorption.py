"""Tests for the single-hydrogen adsorption workflow without model inference."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from uma_catalysis.artifacts import ArtifactStore
from uma_catalysis.calculations.relaxation import RelaxationOutcome
from uma_catalysis.structs.config import ComputeConfig, MaterialConfig, ModelConfig
from uma_catalysis.structs.results import BulkOptimizationResult, EnergyComponents
from uma_catalysis.workflows.h_adsorption import run_hydrogen_adsorption


class FakeAtoms:
    """Small ASE-like object for exercising workflow orchestration."""

    def __init__(self, label: str, tags: tuple[int, ...] = ()) -> None:
        self.label = label
        self._tags = tags
        self.calc = None
        self.pbc = None

    def copy(self) -> FakeAtoms:
        atoms_copy = FakeAtoms(self.label, self._tags)
        atoms_copy.calc = self.calc
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


class HydrogenAdsorptionWorkflowTest(unittest.TestCase):
    def test_selects_lowest_energy_adsorbate(self) -> None:
        clean = FakeAtoms("clean")
        candidate_one = FakeAtoms("candidate-one", tags=(0, 0, 2))
        candidate_two = FakeAtoms("candidate-two", tags=(0, 0, 2))
        reference = FakeAtoms("h2")
        energies = {
            "clean": EnergyComponents(ml_energy=-8.0, d3_energy=0.0),
            "candidate-one": EnergyComponents(ml_energy=-9.0, d3_energy=0.0),
            "candidate-two": EnergyComponents(ml_energy=-10.0, d3_energy=0.0),
            "h2": EnergyComponents(ml_energy=-2.0, d3_energy=0.0),
        }
        bulk_result = BulkOptimizationResult(
            initial_lattice_constant=3.5,
            optimized_lattice_constant=3.5,
            experimental_lattice_constant=3.5,
            converged=True,
        )

        def relax(atoms: FakeAtoms, *_: object, **__: object) -> RelaxationOutcome:
            return RelaxationOutcome(atoms=atoms, converged=True, steps=1)

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
                    "uma_catalysis.workflows.h_adsorption.build_d3_calculator",
                    return_value=object(),
                ),
                patch(
                    "uma_catalysis.workflows.h_adsorption.build_adsorption_slab",
                    return_value=FakeSlab(clean),
                ),
                patch(
                    "uma_catalysis.workflows.h_adsorption.generate_single_adsorbate_candidates",
                    return_value=[candidate_one, candidate_two],
                ),
                patch(
                    "uma_catalysis.workflows.h_adsorption.build_diatomic_reference",
                    return_value=reference,
                ),
                patch(
                    "uma_catalysis.workflows.h_adsorption._relax_with_oc20",
                    side_effect=relax,
                ),
                patch(
                    "uma_catalysis.workflows.h_adsorption.evaluate_energy_components",
                    side_effect=endpoint_energy,
                ),
                patch(
                    "uma_catalysis.workflows.h_adsorption.plot_adsorption_structure",
                    return_value=FakeFigure(),
                ),
                patch.object(ArtifactStore, "write_atoms", new=fake_write_atoms),
                patch("matplotlib.pyplot.close"),
            ):
                result = run_hydrogen_adsorption(
                    predictor=object(),
                    bulk_result=bulk_result,
                    model=ModelConfig(),
                    material=MaterialConfig(),
                    compute=ComputeConfig(calculate_zpe=False),
                    artifacts=artifacts,
                )

            self.assertEqual(result.best_candidate_number, 2)
            self.assertEqual(result.adsorption.electronic_adsorption_energy, -1.0)
            self.assertIsNone(result.adsorption.zero_point_correction)
            self.assertTrue(result.visualization_path.is_file())


if __name__ == "__main__":
    unittest.main()
