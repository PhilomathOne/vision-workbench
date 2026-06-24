"""Base detector class — the strategy pattern anchor for all vision detectors."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from vision_workbench.core.result import DetectionResult
from vision_workbench.core.registry import detector_registry


class BaseDetector(ABC):
    """All vision detectors implement this interface.

    Usage::

        detector = YOLODetector()
        detector.initialize(model="yolov8n.pt")
        result = detector.process(image)  # BGR (H, W, 3) uint8
        detector.cleanup()
    """

    task_type: str = ""
    framework: str = ""

    @abstractmethod
    def initialize(self, **params: Any) -> None:
        """Load model weights and prepare for inference."""
        ...

    @abstractmethod
    def process(self, image: np.ndarray) -> DetectionResult:
        """Run detection on a single BGR image (H, W, 3) uint8."""
        ...

    @abstractmethod
    def cleanup(self) -> None:
        """Release resources."""
        ...


def discover_detectors() -> None:
    """Auto-import all detector subpackages to trigger @register decorators."""
    import importlib
    import pkgutil

    import vision_workbench.detectors as pkg

    for mod_info in pkgutil.iter_modules(pkg.__path__, pkg.__name__ + "."):
        try:
            importlib.import_module(mod_info.name)
        except ImportError:
            pass  # Skip detectors with missing optional deps
