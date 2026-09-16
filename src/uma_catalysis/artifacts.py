"""Create and write workflow artifacts beneath one controlled output root."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ArtifactStore:
    """
    Own output paths and serialization for a single workflow run.

    Parameters
    ----------
    root_directory : pathlib.Path
        Root directory below which all generated artifacts are written.
    """

    root_directory: Path

    def __post_init__(self) -> None:
        """Create the configured artifact root directory."""
        self.root_directory.mkdir(parents=True, exist_ok=True)

    def path(self, relative_path: Path) -> Path:
        """
        Return a validated path below the artifact root.

        Parameters
        ----------
        relative_path : pathlib.Path
            Relative destination beneath ``root_directory``.

        Returns
        -------
        pathlib.Path
            Absolute path below the artifact root.

        Raises
        ------
        ValueError
            If the path is absolute, empty, or escapes the artifact root.
        """
        if relative_path.is_absolute() or not relative_path.parts:
            raise ValueError("relative_path must be a non-empty relative path.")
        if ".." in relative_path.parts:
            raise ValueError("relative_path must not escape the artifact root.")
        return self.root_directory / relative_path

    def directory(self, relative_path: Path) -> Path:
        """
        Create and return an artifact directory below the output root.

        Parameters
        ----------
        relative_path : pathlib.Path
            Relative directory destination beneath ``root_directory``.

        Returns
        -------
        pathlib.Path
            Created directory path.
        """
        destination = self.path(relative_path)
        destination.mkdir(parents=True, exist_ok=True)
        return destination

    def write_atoms(
        self, atoms: Any, relative_path: Path, image_format: str | None = None
    ) -> Path:
        """
        Write an ASE-compatible structure or trajectory.

        Parameters
        ----------
        atoms : Any
            ASE-compatible atoms object or sequence of objects.
        relative_path : pathlib.Path
            Relative output filename beneath ``root_directory``.
        image_format : str or None, optional
            Explicit ASE writer format. When omitted, ASE infers it from the
            destination suffix. Default is None.

        Returns
        -------
        pathlib.Path
            Written artifact path.
        """
        from ase.io import write

        destination = self.path(relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        kwargs = {} if image_format is None else {"format": image_format}
        write(str(destination), atoms, **kwargs)
        return destination

    def write_figure(self, figure: Any, relative_path: Path, dpi: int = 200) -> Path:
        """
        Write a Matplotlib-compatible figure without displaying it.

        Parameters
        ----------
        figure : Any
            Figure object exposing ``savefig``.
        relative_path : pathlib.Path
            Relative image filename beneath ``root_directory``.
        dpi : int, optional
            Image resolution in dots per inch. Default is 200.

        Returns
        -------
        pathlib.Path
            Written figure path.
        """
        if dpi < 1:
            raise ValueError("dpi must be at least 1.")
        destination = self.path(relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=dpi, bbox_inches="tight")
        return destination

    def write_json(self, data: Mapping[str, Any], relative_path: Path) -> Path:
        """
        Serialize JSON-compatible result metadata beneath the output root.

        Parameters
        ----------
        data : collections.abc.Mapping[str, Any]
            JSON-compatible mapping to serialize.
        relative_path : pathlib.Path
            Relative JSON filename beneath ``root_directory``.

        Returns
        -------
        pathlib.Path
            Written JSON artifact path.
        """
        destination = self.path(relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, sort_keys=True)
            file.write("\n")
        return destination
