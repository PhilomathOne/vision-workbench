"""Dataset catalog — index, search, and version-track datasets."""

from pathlib import Path
from typing import Optional

import yaml

from vision_workbench.data.schema import DatasetManifest


class DatasetCatalog:
    """Registry of known datasets with version tracking.

    The catalog maintains an index of dataset locations and their manifests,
    enabling quick lookup and validation.
    """

    def __init__(self, workspace: Optional[Path] = None) -> None:
        self.workspace = workspace or Path("./vw_workspace")
        self._index: dict[str, Path] = {}  # name -> root path

    def register(self, name: str, path: Path) -> None:
        """Add a dataset to the catalog."""
        if not path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {path}")
        self._index[name] = path

    def get(self, name: str) -> Path:
        """Look up a dataset by name."""
        if name not in self._index:
            raise KeyError(f"Dataset '{name}' not in catalog. Available: {list(self._index)}")
        return self._index[name]

    def list(self) -> dict[str, Path]:
        """Return all registered datasets."""
        return dict(self._index)

    def load_manifest(self, name: str) -> DatasetManifest:
        """Load and parse the dataset.yaml manifest."""
        root = self.get(name)
        manifest_path = root / "dataset.yaml"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No dataset.yaml found in {root}")
        with open(manifest_path, "r") as f:
            raw = yaml.safe_load(f)
        return DatasetManifest(**raw)

    def scan(self, directory: Path) -> list[str]:
        """Discover all valid datasets under a directory."""
        found: list[str] = []
        for manifest in directory.rglob("dataset.yaml"):
            root = manifest.parent
            name = root.name
            self.register(name, root)
            found.append(name)
        return found

    def remove(self, name: str) -> None:
        """Remove a dataset from the catalog."""
        self._index.pop(name, None)
