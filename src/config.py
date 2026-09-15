"""Provide project-wide paths for UMA catalysis workflows."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    """
    Provide stable paths shared by future UMA catalysis workflows.

    The scientific execution configuration will be added in a later refactoring
    step. This class intentionally contains only repository-level paths.
    """

    @property
    def project_root(self) -> Path:
        """
        Return the absolute repository root that contains ``src``.

        Returns
        -------
        pathlib.Path
            Absolute project root directory.
        """
        return Path(__file__).resolve().parent.parent

    @property
    def output_directory(self) -> Path:
        """
        Return the default location for generated workflow artifacts.

        Returns
        -------
        pathlib.Path
            Directory intended for structures, trajectories, plots, and reports.
        """
        return self.project_root / "outputs"


settings = Settings()
