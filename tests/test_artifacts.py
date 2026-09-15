"""Unit tests for controlled workflow artifact output paths."""

import json
import tempfile
import unittest
from pathlib import Path

from uma_catalysis.artifacts import ArtifactStore


class ArtifactStoreTests(unittest.TestCase):
    """Validate safe generated-artifact path handling and metadata writing."""

    def test_json_artifact_is_written_below_the_output_root(self) -> None:
        """Create nested outputs without allowing paths outside the run directory."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = ArtifactStore(Path(temporary_directory) / "outputs")
            path = store.write_json({"energy": -1.2}, Path("part1/result.json"))

            self.assertTrue(path.is_file())
            self.assertEqual(json.loads(path.read_text()), {"energy": -1.2})
            with self.assertRaises(ValueError):
                store.path(Path("../outside.json"))
            with self.assertRaises(ValueError):
                store.path(Path("/outside.json"))
