"""Domain primitive types shared across all modules.

All types are immutable (frozen Pydantic models) to ensure
safe passage through the pipeline context.
"""

from enum import StrEnum
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, Field


class TaskType(StrEnum):
    """Supported vision task types."""

    OBJECT_DETECTION = "object_detection"
    CLASSIFICATION = "classification"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    FACE_DETECTION = "face_detection"
    POSE_ESTIMATION = "pose_estimation"
    HAND_TRACKING = "hand_tracking"
    OCR = "ocr"
    FEATURE_MATCHING = "feature_matching"


class BoundingBox(BaseModel, frozen=True):
    """Normalized or pixel-coordinate bounding box."""

    x1: float
    y1: float
    x2: float
    y2: float
    coord_type: Literal["pixel", "normalized"] = "pixel"
    confidence: Optional[float] = None
    class_id: Optional[int] = None
    class_name: Optional[str] = None


class Keypoint(BaseModel, frozen=True):
    """A single keypoint (e.g., joint, landmark)."""

    x: float
    y: float
    visibility: Literal["visible", "occluded", "not_present"] = "visible"
    confidence: Optional[float] = None
    name: Optional[str] = None


class SegmentationMask(BaseModel):
    """Reference to a segmentation mask stored on disk.

    Not frozen — mask data is large and lives on filesystem.
    """

    mask_path: str
    format: Literal["png", "rle", "polygon"] = "png"
    height: int
    width: int
    class_id: int


class ImageMetadata(BaseModel, frozen=True):
    """Metadata for a single image in a dataset."""

    path: str
    width: int
    height: int
    channels: int
    format: str = "jpg"
    file_size_bytes: int = 0
    md5_hash: str = ""
    exif: dict = Field(default_factory=dict)


class DatasetSplit(BaseModel, frozen=True):
    """Record of a dataset train/val/test split."""

    train: list[str] = Field(default_factory=list)
    val: list[str] = Field(default_factory=list)
    test: list[str] = Field(default_factory=list)
    split_method: str = "random"
    random_seed: int = 42
    split_timestamp: str = ""


class ModalityType(StrEnum):
    """Multi-modal input types."""

    RGB = "rgb"
    DEPTH = "depth"
    INFRARED = "infrared"
    LIDAR_PROJECTION = "lidar_projection"


class MultiModalSample(BaseModel):
    """A multi-modal data sample (image + auxiliary modalities)."""

    modalities: dict[ModalityType, np.ndarray] = Field(default_factory=dict)
    annotations: Optional[list] = None
    metadata: dict = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}
