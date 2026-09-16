"""Tests for opt-in CodeCarbon tracking behavior."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from uma_catalysis.artifacts import ArtifactStore
from uma_catalysis.structs.config import TrackingConfig
from uma_catalysis.tracking import track_emissions


class FakeTracker:
    """Record context-manager tracking lifecycle calls."""

    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class TrackingTests(unittest.TestCase):
    """Validate that tracking is opt-in and always stopped."""

    def test_enabled_tracking_starts_and_stops_one_tracker(self) -> None:
        """Yield the configured CSV destination around the workflow boundary."""
        tracker = FakeTracker()
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifacts = ArtifactStore(Path(temporary_directory))
            with patch("uma_catalysis.tracking._build_tracker", return_value=tracker):
                with track_emissions(TrackingConfig(enabled=True), artifacts) as path:
                    self.assertEqual(path, artifacts.root_directory / "emissions.csv")
                    self.assertTrue(tracker.started)

        self.assertTrue(tracker.stopped)


if __name__ == "__main__":
    unittest.main()
