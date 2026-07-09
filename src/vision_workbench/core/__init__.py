"""Core abstractions for the Vision Workbench platform."""

from vision_workbench.core.base import BaseDetector, BaseStage
from vision_workbench.core.config import PipelineConfig
from vision_workbench.core.context import PipelineContext
from vision_workbench.core.exceptions import VisionWorkbenchError
from vision_workbench.core.registry import Registry
from vision_workbench.core.result import DetectionResult, StageResult

__all__ = [
    "BaseDetector",
    "BaseStage",
    "DetectionResult",
    "PipelineConfig",
    "PipelineContext",
    "Registry",
    "StageResult",
    "VisionWorkbenchError",
]
