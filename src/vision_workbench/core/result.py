"""Unified result types for pipeline stages and detectors."""

from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field

from vision_workbench.core.types import BoundingBox, Keypoint


class DetectionResult(BaseModel):
    """Normalized output from any vision detector.

    All detectors (regardless of framework) produce this unified format.
    """

    model_config = {"arbitrary_types_allowed": True}

    # Source provenance
    source_path: Optional[str] = None
    image_shape: tuple[int, int, int] = (0, 0, 0)  # (H, W, C)

    # Detection outputs
    boxes: list[BoundingBox] = Field(default_factory=list)
    keypoints: list[list[Keypoint]] = Field(default_factory=list)  # [instance_idx][kp_idx]
    masks: list[np.ndarray] = Field(default_factory=list)  # binary masks (H, W)
    classifications: list[dict[str, Any]] = Field(default_factory=list)  # [{label, confidence}]
    text: list[str] = Field(default_factory=list)  # OCR transcripts

    # Detector provenance
    detector_name: str = ""
    task_type: str = ""
    framework: str = ""
    processing_time_ms: float = 0.0


class StageResult(BaseModel):
    """Base type for pipeline stage outputs.

    Each stage returns a specialized subclass carrying its own
    artifact references and metrics.
    """

    stage_name: str = ""
    success: bool = True
    error_message: str = ""
    duration_seconds: float = 0.0
    artifacts: dict[str, str] = Field(default_factory=dict)  # name -> path
    metrics: dict[str, Any] = Field(default_factory=dict)
