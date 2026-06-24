"""Model Zoo — pre-trained model download and cache management."""

import hashlib
from pathlib import Path
from typing import Optional

import structlog
import yaml

logger = structlog.get_logger(__name__)

# Known models registry — maps name → download info
KNOWN_MODELS: dict[str, dict] = {
    "yolov8n": {
        "task": "object_detection", "framework": "ultralytics",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n.pt",
        "file": "yolov8n.pt", "size_mb": 6.2, "input_shape": [3, 640, 640],
    },
    "yolov8m": {
        "task": "object_detection", "framework": "ultralytics",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8m.pt",
        "file": "yolov8m.pt", "size_mb": 52.0, "input_shape": [3, 640, 640],
    },
    "yolov8x": {
        "task": "object_detection", "framework": "ultralytics",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8x.pt",
        "file": "yolov8x.pt", "size_mb": 136.0, "input_shape": [3, 640, 640],
    },
}


class ModelZoo:
    """Manage pre-trained model download and local caching.

    Models are stored in ``<workspace>/models/zoo/`` with a ``zoo_index.yaml``
    tracking available models and their metadata.
    """

    def __init__(self, workspace: Optional[Path] = None) -> None:
        self.workspace = workspace or Path("./vw_workspace")
        self.zoo_dir = self.workspace / "models" / "zoo"
        self.zoo_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.zoo_dir / "zoo_index.yaml"
        self._index: dict = self._load_index()

    def list(self) -> list[str]:
        """List cached model names."""
        return list(self._index.get("entries", {}))

    def resolve(self, name: str) -> Optional[Path]:
        """Get the local path of a cached model, or None if not cached."""
        entry = self._index.get("entries", {}).get(name)
        if entry:
            p = self.zoo_dir / entry.get("file", name)
            if p.exists():
                return p
        return None

    def pull(self, name: str) -> Path:
        """Download a known model to the zoo cache.

        Args:
            name: Model name (e.g., "yolov8n").

        Returns:
            Path to the cached model file.
        """
        if name not in KNOWN_MODELS:
            raise KeyError(f"Unknown model '{name}'. Known: {list(KNOWN_MODELS)}")

        info = KNOWN_MODELS[name]
        file_name = info["file"]
        dest = self.zoo_dir / file_name

        if dest.exists():
            logger.info("zoo.already_cached", model=name, path=str(dest))
            return dest

        # Download
        import urllib.request

        logger.info("zoo.downloading", model=name, url=info["url"])
        urllib.request.urlretrieve(info["url"], str(dest))

        # Verify
        size_mb = dest.stat().st_size / (1024 * 1024)
        logger.info("zoo.downloaded", model=name, size_mb=round(size_mb, 1))

        # Update index
        self._index.setdefault("entries", {})[name] = {
            "file": file_name, "task": info["task"], "framework": info["framework"],
            "input_shape": info.get("input_shape"), "size_mb": round(size_mb, 1),
        }
        self._save_index()
        return dest

    def info(self, name: str) -> Optional[dict]:
        """Get metadata for a model (cached or known)."""
        cached = self._index.get("entries", {}).get(name)
        if cached:
            return cached
        return KNOWN_MODELS.get(name)

    def remove(self, name: str) -> None:
        """Remove a model from the local cache."""
        entry = self._index.get("entries", {}).pop(name, None)
        if entry:
            p = self.zoo_dir / entry["file"]
            if p.exists():
                p.unlink()
            self._save_index()

    def _load_index(self) -> dict:
        if self._index_path.exists():
            with open(self._index_path, "r") as f:
                return yaml.safe_load(f) or {}
        return {"entries": {}}

    def _save_index(self) -> None:
        with open(self._index_path, "w") as f:
            yaml.dump(self._index, f, default_flow_style=False, allow_unicode=True)
