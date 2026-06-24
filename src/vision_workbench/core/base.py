"""Abstract base classes for all pipeline stages and detectors."""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import numpy as np

from vision_workbench.core.context import PipelineContext
from vision_workbench.core.result import DetectionResult, StageResult

TConfig = TypeVar("TConfig")
TResult = TypeVar("TResult", bound=StageResult)


class BaseStage(ABC, Generic[TConfig, TResult]):
    """Abstract base for all pipeline stages.

    Design principles:
    - Stage state is immutable (all params via config).
    - Input/output flows only through PipelineContext.
    - dry_run() previews side effects for config validation.
    - depends_on declares prerequisite stages for DAG scheduling.
    """

    name: str = ""
    description: str = ""
    depends_on: list[str] = []

    @abstractmethod
    def validate_inputs(self, ctx: PipelineContext) -> tuple[bool, list[str]]:
        """Check that ctx contains all required inputs for this stage.

        Returns:
            (is_valid, list_of_missing_keys)
        """
        ...

    @abstractmethod
    def run(self, config: TConfig, ctx: PipelineContext) -> tuple[TResult, PipelineContext]:
        """Execute stage logic.

        Returns:
            (stage_result, updated_context)
        """
        ...

    @abstractmethod
    def dry_run(self, config: TConfig, ctx: PipelineContext) -> dict:
        """Preview operations without side effects.

        Returns:
            Summary dict describing what will happen.
        """
        ...


class BaseDetector(ABC):
    """Abstract base for all vision detectors (inference only).

    Design principles:
    - initialize() loads model weights once.
    - process() runs inference on a single BGR image.
    - cleanup() frees GPU/memory resources.
    - task_type and framework are both class and instance properties.
    """

    task_type: str = ""
    framework: str = ""

    @abstractmethod
    def initialize(self, **params: Any) -> None:
        """Load model weights, warm up. Called once before process()."""
        ...

    @abstractmethod
    def process(self, image: np.ndarray) -> DetectionResult:
        """Run detection on a single BGR image (H, W, 3) uint8.

        Returns:
            Normalized DetectionResult.
        """
        ...

    @abstractmethod
    def cleanup(self) -> None:
        """Release GPU memory, close sessions, destroy windows."""
        ...


class BaseExporter(ABC):
    """Abstract base for model exporters (ONNX, TensorRT, TFLite, etc.).

    Each platform exporter converts a trained PyTorch model to the
    target runtime format.
    """

    platform: str = ""

    @abstractmethod
    def export(
        self,
        model: Any,
        output_path: str,
        input_shape: tuple[int, ...],
        **kwargs: Any,
    ) -> str:
        """Export model to the target platform format.

        Returns:
            Path to the exported model file.
        """
        ...

    @abstractmethod
    def validate(self, original: Any, exported_path: str, sample_input: np.ndarray) -> bool:
        """Verify exported model output matches original."""
        ...
