"""Pipeline configuration model — the single source of truth for experiments.

Configuration is hierarchical: defaults < template < user config < CLI overrides.
"""

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field


class RuntimeConfig(BaseModel):
    """Global runtime settings."""

    workspace: Path = Path("./vw_workspace")
    artifacts_dir: Path = Path("./vw_workspace/artifacts")
    device: Literal["cpu", "cuda", "mps", "auto"] = "auto"
    seed: int = 42
    log_level: Literal["DEBUG", "INFO", "WARN", "ERROR"] = "INFO"
    log_file: Optional[Path] = None
    cache: bool = True


class DataStageConfig(BaseModel):
    """Stage 1: Data cleaning configuration."""

    enabled: bool = True
    source: str = "data/raw/"
    target: str = "data/processed/"
    dataset_name: str = "dataset"
    validation: dict = Field(default_factory=lambda: {
        "min_resolution": [32, 32],
        "max_resolution": [4096, 4096],
        "allowed_formats": [".jpg", ".jpeg", ".png", ".bmp"],
    })
    dedup: dict = Field(default_factory=lambda: {
        "enabled": True,
        "method": "phash",
        "threshold": 5,
    })
    split: dict = Field(default_factory=lambda: {
        "train": 0.7,
        "val": 0.15,
        "test": 0.15,
        "method": "random",
        "seed": 42,
    })
    augment: dict = Field(default_factory=lambda: {"enabled": False})


class AnnotateStageConfig(BaseModel):
    """Stage 2: Annotation management configuration."""

    enabled: bool = True
    input_format: str = "coco"
    output_format: str = "coco"
    source_path: Optional[str] = None
    pre_annotation: dict = Field(default_factory=lambda: {"enabled": False})
    quality_checks: dict = Field(default_factory=lambda: {"enabled": True})


class TrainStageConfig(BaseModel):
    """Stage 3: Model training configuration."""

    enabled: bool = True
    framework: str = "torch"
    task: str = "object_detection"
    model: dict = Field(default_factory=dict)
    data: dict = Field(default_factory=dict)
    training: dict = Field(default_factory=lambda: {
        "epochs": 100,
        "batch_size": 16,
        "optimizer": "adamw",
        "lr": 0.001,
    })
    callbacks: dict = Field(default_factory=dict)


class ValidateStageConfig(BaseModel):
    """Stage 4: Validation/testing configuration."""

    enabled: bool = True
    batch_size: int = 32
    metrics: list[str] = Field(default_factory=lambda: ["mAP", "mAP_50", "mAP_75"])
    regression_test: dict = Field(default_factory=lambda: {"enabled": False})


class EvaluateStageConfig(BaseModel):
    """Stage 5: Model evaluation configuration."""

    enabled: bool = True
    curves: list[str] = Field(default_factory=lambda: ["pr_curve", "confusion_matrix"])
    profiling: dict = Field(default_factory=lambda: {"enabled": False})


class OptimizeStageConfig(BaseModel):
    """Stage 6: Quantization/pruning configuration."""

    enabled: bool = False
    methods: list[dict] = Field(default_factory=list)


class ExportStageConfig(BaseModel):
    """Stage 7: Model export configuration."""

    enabled: bool = False
    onnx: dict = Field(default_factory=lambda: {"opset_version": 17})
    targets: list[dict] = Field(default_factory=list)


class DeployStageConfig(BaseModel):
    """Stage 8: Edge deployment configuration."""

    enabled: bool = False
    devices: list[dict] = Field(default_factory=list)


class PipelineConfig(BaseModel):
    """Root pipeline configuration.

    Loaded from YAML and validated by Pydantic before execution.
    """

    # Metadata
    name: str = "untitled"
    description: str = ""
    version: str = "1.0"

    # Runtime
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    # Stage selection
    stages: list[str] = Field(default_factory=list)

    # Stage-specific configs
    data: Optional[DataStageConfig] = None
    annotate: Optional[AnnotateStageConfig] = None
    train: Optional[TrainStageConfig] = None
    validation: Optional[ValidateStageConfig] = None  # "validate" shadows BaseModel method
    evaluate: Optional[EvaluateStageConfig] = None
    optimize: Optional[OptimizeStageConfig] = None
    export: Optional[ExportStageConfig] = None
    deploy: Optional[DeployStageConfig] = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load and validate a pipeline config from a YAML file."""
        import yaml

        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)
