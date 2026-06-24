"""Core abstractions for the Vision Workbench platform."""

from vision_workbench.core.base import BaseDetector, BaseStage
from vision_workbench.core.config import PipelineConfig
from vision_workbench.core.context import PipelineContext
from vision_workbench.core.exceptions import VisionWorkbenchError
from vision_workbench.core.registry import Registry
from vision_workbench.core.result import DetectionResult
from vision_workbench.core.types import BoundingBox, Keypoint

__all__ = [
    "BaseDetector",
    "BaseStage",
    "BoundingBox",
    "DetectionResult",
    "Keypoint",
    "PipelineConfig",
    "PipelineContext",
    "Registry",
    "VisionWorkbenchError",
]
