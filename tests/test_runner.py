"""Tests for dependency-aware experiment orchestration."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from uma_catalysis.runner import resolve_workflows, run_experiment
from uma_catalysis.structs.config import ExperimentConfig, RunConfig, TrackingConfig
from uma_catalysis.structs.results import BulkOptimizationResult


class RunnerTests(unittest.TestCase):
    """Validate workflow dependencies and no-model orchestration behavior."""

    def test_resolve_workflows_includes_prerequisites_once(self) -> None:
        """Run shared bulk and surface prerequisites before selected workflows."""
        resolved = resolve_workflows(("wulff", "coverage", "co_reaction"))

        self.assertEqual(
            resolved,
            ("bulk", "surface_energies", "wulff", "coverage", "co_reaction"),
        )

    def test_run_experiment_loads_one_predictor_and_writes_summary(self) -> None:
        """Share the model predictor and serialize all executed result values."""
        bulk_result = BulkOptimizationResult(3.5, 3.5, 3.5, True)
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = ExperimentConfig(
                run=RunConfig(
                    workflows=("wulff", "coverage", "co_reaction"),
                    output_directory=Path(temporary_directory),
                ),
                tracking=TrackingConfig(enabled=False),
            )
            predictor = object()
            with (
                patch(
                    "uma_catalysis.runner.load_predictor", return_value=predictor
                ) as load,
                patch(
                    "uma_catalysis.runner.run_bulk_optimization",
                    return_value=bulk_result,
                ) as bulk,
                patch(
                    "uma_catalysis.runner.run_surface_energy_study",
                    return_value={"stage": "surface"},
                ) as surface,
                patch(
                    "uma_catalysis.runner.run_wulff_construction",
                    return_value={"stage": "wulff"},
                ) as wulff,
                patch(
                    "uma_catalysis.runner.run_coverage_study",
                    return_value={"stage": "coverage"},
                ) as coverage,
                patch(
                    "uma_catalysis.runner.run_co_reaction_study",
                    return_value={"stage": "co-reaction"},
                ) as co_reaction,
            ):
                experiment = run_experiment(config)

            self.assertEqual(load.call_count, 1)
            self.assertEqual(bulk.call_count, 1)
            self.assertEqual(surface.call_count, 1)
            self.assertEqual(wulff.call_count, 1)
            self.assertEqual(coverage.call_count, 1)
            self.assertEqual(co_reaction.call_count, 1)
            self.assertEqual(
                experiment.executed_workflows,
                ("bulk", "surface_energies", "wulff", "coverage", "co_reaction"),
            )
            with experiment.summary_path.open(encoding="utf-8") as file:
                summary = json.load(file)

        self.assertEqual(summary["results"]["bulk"]["optimized_lattice_constant"], 3.5)
        self.assertEqual(summary["results"]["co_reaction"], {"stage": "co-reaction"})


if __name__ == "__main__":
    unittest.main()
