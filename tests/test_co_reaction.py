"""Tests for CO reaction thermochemistry without model inference."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from uma_catalysis.artifacts import ArtifactStore
from uma_catalysis.calculations.reaction_paths import NEBOutcome
from uma_catalysis.calculations.relaxation import RelaxationOutcome
from uma_catalysis.structs.config import ComputeConfig, MaterialConfig, ModelConfig
from uma_catalysis.structs.results import BulkOptimizationResult, EnergyComponents
from uma_catalysis.workflows.co_reaction import run_co_reaction_study


class FakeAtoms:
    """Small ASE-like object for exercising CO workflow orchestration."""

    def __init__(
        self, label: str, tags: tuple[int, ...] = (), distance: float = 2.0
    ) -> None:
        self.label = label
        self._tags = tags
        self.distance = distance
        self.pbc = None

    def copy(self) -> FakeAtoms:
        atoms_copy = FakeAtoms(self.label, self._tags, self.distance)
        atoms_copy.pbc = self.pbc
        return atoms_copy

    def get_tags(self) -> tuple[int, ...]:
        return self._tags

    def get_distance(self, *_: object, **__: object) -> float:
        return self.distance

    def set_pbc(self, value: object) -> None:
        self.pbc = value


class FakeSlab:
    def __init__(self, atoms: FakeAtoms) -> None:
        self.atoms = atoms


class FakeFigure:
    def savefig(self, filename: Path, **_: object) -> None:
        Path(filename).touch()


class COReactionWorkflowTest(unittest.TestCase):
    def test_calculates_co_and_reaction_thermochemistry(self) -> None:
        clean = FakeAtoms("clean")
        co_gas = FakeAtoms("co-gas")
        energies = {
            "clean": EnergyComponents(-8.0),
            "co-gas": EnergyComponents(-3.0),
            "co-high": EnergyComponents(-12.0),
            "co-low": EnergyComponents(-13.0),
            "c-o-valid": EnergyComponents(-12.0),
            "c-high": EnergyComponents(-8.5),
            "c-low": EnergyComponents(-9.0),
            "o-high": EnergyComponents(-9.5),
            "o-low": EnergyComponents(-10.0),
        }
        bulk_result = BulkOptimizationResult(3.5, 3.5, 3.5, True)

        def relaxed(atoms: FakeAtoms, *_: object, **__: object) -> RelaxationOutcome:
            if atoms.label == "template":
                atoms = clean.copy()
            return RelaxationOutcome(atoms=atoms, converged=True, steps=1)

        def candidates(
            _: FakeSlab, adsorbate_smiles: tuple[str, ...], **__: object
        ) -> list[FakeAtoms]:
            match adsorbate_smiles:
                case ("*CO",):
                    return [FakeAtoms("co-high", (2, 2)), FakeAtoms("co-low", (2, 2))]
                case ("*C", "*O"):
                    return [
                        FakeAtoms("c-o-recombined", (2, 2), distance=1.0),
                        FakeAtoms("c-o-valid", (2, 2), distance=2.0),
                    ]
                case ("*C",):
                    return [FakeAtoms("c-high", (2,)), FakeAtoms("c-low", (2,))]
                case ("*O",):
                    return [FakeAtoms("o-high", (2,)), FakeAtoms("o-low", (2,))]
            raise AssertionError(f"Unexpected adsorbates: {adsorbate_smiles}")

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
                    "uma_catalysis.workflows.co_reaction.build_d3_calculator",
                    return_value=object(),
                ),
                patch(
                    "uma_catalysis.workflows.co_reaction.build_adsorption_slab",
                    side_effect=lambda *_args, **_kwargs: FakeSlab(
                        FakeAtoms("template")
                    ),
                ),
                patch(
                    "uma_catalysis.workflows.co_reaction.build_diatomic_reference",
                    return_value=co_gas,
                ),
                patch(
                    "uma_catalysis.workflows.co_reaction.generate_multiple_adsorbate_candidates",
                    side_effect=candidates,
                ),
                patch(
                    "uma_catalysis.workflows.co_reaction._relax_oc20",
                    side_effect=relaxed,
                ),
                patch(
                    "uma_catalysis.workflows.co_reaction.evaluate_energy_components",
                    side_effect=endpoint_energy,
                ),
                patch(
                    "uma_catalysis.workflows.co_reaction.prepare_stretched_co_guess",
                    return_value=FakeAtoms("neb-initial", (2, 2)),
                ),
                patch(
                    "uma_catalysis.workflows.co_reaction.run_dyneb",
                    return_value=NEBOutcome(
                        images=[FakeAtoms("neb-initial"), FakeAtoms("neb-final")],
                        converged=True,
                        steps=1,
                        relative_energies=(0.0, 1.5),
                    ),
                ),
                patch(
                    "uma_catalysis.workflows.co_reaction.plot_neb_path",
                    return_value=FakeFigure(),
                ),
                patch.object(ArtifactStore, "write_atoms", new=fake_write_atoms),
                patch("matplotlib.pyplot.close"),
            ):
                result = run_co_reaction_study(
                    predictor=object(),
                    bulk_result=bulk_result,
                    model=ModelConfig(),
                    material=MaterialConfig(),
                    compute=ComputeConfig(calculate_zpe=False, run_neb=False),
                    artifacts=artifacts,
                )
                neb_result = run_co_reaction_study(
                    predictor=object(),
                    bulk_result=bulk_result,
                    model=ModelConfig(),
                    material=MaterialConfig(),
                    compute=ComputeConfig(calculate_zpe=False, run_neb=True),
                    artifacts=artifacts,
                )

            self.assertEqual(result.reaction.electronic_reaction_energy, -1.0)
            self.assertEqual(result.co_adsorption.electronic_adsorption_energy, -2.0)
            self.assertEqual(result.separate_carbon_energy.total_energy, -9.0)
            self.assertEqual(result.separate_oxygen_energy.total_energy, -10.0)
            self.assertIsNone(result.reaction.forward_barrier)
            self.assertEqual(neb_result.reaction.forward_barrier, 1.5)
            self.assertEqual(neb_result.reaction.reverse_barrier, 0.0)
            self.assertIsNotNone(neb_result.reaction.path_artifact_path)
            self.assertTrue(neb_result.neb_figure_path.is_file())


if __name__ == "__main__":
    unittest.main()
