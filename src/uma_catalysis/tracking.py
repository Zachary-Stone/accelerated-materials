"""Provide an opt-in, narrow CodeCarbon tracking boundary."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from uma_catalysis.artifacts import ArtifactStore
from uma_catalysis.structs.config import TrackingConfig


def _build_tracker(config: TrackingConfig, destination: Path) -> Any:
    """Construct the deferred CodeCarbon tracker for one configured output."""
    from codecarbon import EmissionsTracker

    return EmissionsTracker(
        project_name=config.project_name,
        output_dir=str(destination.parent),
        output_file=destination.name,
        save_to_file=True,
        log_level="error",
    )


@contextmanager
def track_emissions(
    config: TrackingConfig, artifacts: ArtifactStore
) -> Iterator[Path | None]:
    """
    Track execution emissions only when explicitly enabled in configuration.

    Yields the expected CSV artifact path when tracking is enabled, otherwise
    ``None``. The tracker is stopped even if a workflow raises an exception.
    """
    if not config.enabled:
        yield None
        return

    destination = artifacts.path(Path(config.output_file))
    destination.parent.mkdir(parents=True, exist_ok=True)
    tracker = _build_tracker(config, destination)
    tracker.start()
    try:
        yield destination
    finally:
        tracker.stop()
