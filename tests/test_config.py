"""Unit tests for UMA catalysis configuration parsing and validation."""

import unittest
from pathlib import Path

from uma_catalysis.runner import load_experiment_config
from uma_catalysis.structs import ComputeConfig, MaterialConfig, RunConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ConfigurationTests(unittest.TestCase):
    """Validate tutorial configuration defaults and TOML loading."""

    def test_default_configs_match_the_full_notebook_settings(self) -> None:
        """Preserve the source tutorial's full-workflow defaults."""
        compute = ComputeConfig()
        material = MaterialConfig()

        self.assertEqual(compute.random_seed, 42)
        self.assertFalse(compute.fast_mode)
        self.assertEqual(compute.relaxation_steps, 300)
        self.assertEqual(compute.neb_intermediate_images, 10)
        self.assertEqual(material.element, "Ni")
        self.assertEqual(material.surface_thicknesses, (4, 6, 8))
        self.assertEqual(material.adsorption_facet, (1, 1, 1))

    def test_invalid_config_values_are_rejected(self) -> None:
        """Reject invalid work budgets, materials, and workflow names."""
        invalid_configs = [
            lambda: ComputeConfig(random_seed=-1),
            lambda: ComputeConfig(relaxation_steps=0),
            lambda: MaterialConfig(surface_facets=()),
            lambda: MaterialConfig(surface_thicknesses=(4, 0)),
            lambda: RunConfig(workflows=("unknown",)),
        ]

        for invalid_config in invalid_configs:
            with self.subTest(invalid_config=invalid_config):
                with self.assertRaises(ValueError):
                    invalid_config()

    def test_toml_config_is_loaded_with_a_resolved_output_path(self) -> None:
        """Load the repository configuration independent of the working directory."""
        config = load_experiment_config(PROJECT_ROOT / "experiment.toml")

        self.assertEqual(config.run.workflows, ("bulk",))
        self.assertEqual(config.run.output_directory, PROJECT_ROOT / "outputs")
        self.assertEqual(config.model.model_name, "uma-s-1p2")
        self.assertEqual(config.compute.adsorption_candidate_count, 5)
        self.assertFalse(config.tracking.enabled)
