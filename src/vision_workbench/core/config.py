"""Pipeline configuration — 5-stage YOLO-focused workflow."""

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field


class RuntimeConfig(BaseModel):
    """Global runtime settings."""

    workspace: Path = Path("./vw_workspace")
    device: str = "auto"
    seed: int = 42


class DataStageConfig(BaseModel):
    """Stage 1: Data preparation with YOLO format support."""

    source: str = "data/raw/"
    target: str = "data/processed/"
    format: str = "yolo"  # yolo | coco
    classes: list[str] = Field(default_factory=list)
    split: dict = Field(default_factory=lambda: {"train": 0.7, "val": 0.2, "test": 0.1, "seed": 42})
    validation: dict = Field(default_factory=lambda: {"min_resolution": [1, 1], "max_resolution": [8192, 8192], "allowed_formats": [".jpg", ".jpeg", ".png", ".bmp"]})


class TrainStageConfig(BaseModel):
    """Stage 2: YOLO training via Ultralytics."""

    model: str = "yolov8n"
    pretrained: bool = True
    epochs: int = 100
    batch: int = 16
    imgsz: int = 640
    lr0: float = 0.01
    optimizer: str = "auto"
    device: Optional[str] = None  # overrides runtime.device


class ValidateStageConfig(BaseModel):
    """Stage 3: Validation + evaluation."""

    split: str = "val"
    conf: float = 0.001
    iou: float = 0.6
    plots: bool = True  # generate PR curve + confusion matrix


class ExportStageConfig(BaseModel):
    """Stage 4: ONNX export + optional FP16 quantization."""

    formats: list[str] = Field(default_factory=lambda: ["onnx"])
    opset: int = 17
    simplify: bool = True
    fp16: bool = False
    int8: bool = False


class DeployStageConfig(BaseModel):
    """Stage 5: Deployment package generation."""

    enabled: bool = False
    devices: list[dict] = Field(default_factory=list)


class PipelineConfig(BaseModel):
    """Root pipeline configuration. Loaded from YAML."""

    name: str = "untitled"
    description: str = ""
    stages: list[str] = Field(default_factory=list)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    data: Optional[DataStageConfig] = None
    train: Optional[TrainStageConfig] = None
    validation: Optional[ValidateStageConfig] = None
    export: Optional[ExportStageConfig] = None
    deploy: Optional[DeployStageConfig] = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        import yaml
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)
