"""Exception hierarchy for Vision Workbench.

All platform errors inherit from VisionWorkbenchError for unified
error handling and logging.
"""


class VisionWorkbenchError(Exception):
    """Base exception for all Vision Workbench errors."""


# --- Configuration errors ---


class ConfigError(VisionWorkbenchError):
    """Configuration-related errors."""


class ConfigValidationError(ConfigError):
    """Config failed Pydantic validation."""


class ConfigMissingKeyError(ConfigError):
    """Required key missing from configuration."""


# --- Registry errors ---


class RegistryError(VisionWorkbenchError):
    """Component registry lookup or registration error."""


# --- Dependency errors ---


class DependencyError(VisionWorkbenchError):
    """Missing or incompatible optional dependency."""


class MissingDependency(DependencyError):
    """An optional framework/library is not installed."""


class VersionConflict(DependencyError):
    """Installed version is incompatible."""


# --- Pipeline errors ---


class PipelineError(VisionWorkbenchError):
    """Pipeline execution error."""


class StageInputError(PipelineError):
    """A stage is missing required inputs in the context."""


class StageExecutionError(PipelineError):
    """A stage failed during execution."""


class StageNotFoundError(PipelineError):
    """Referenced stage is not registered."""


# --- Data errors ---


class DataError(VisionWorkbenchError):
    """Data loading or validation error."""


class DataValidationError(DataError):
    """Data failed integrity checks."""


class AnnotationFormatError(DataError):
    """Unsupported or malformed annotation format."""


# --- Model errors ---


class ModelError(VisionWorkbenchError):
    """Model-related error."""


class ModelNotFoundError(ModelError):
    """Model not found in registry or filesystem."""


class ExportError(ModelError):
    """Model export (ONNX/TensorRT/etc.) failed."""


class InferenceError(ModelError):
    """Inference failed."""


# --- Deploy errors ---


class DeployError(VisionWorkbenchError):
    """Deployment error."""


class ConnectionError(DeployError):
    """Could not connect to edge device."""


class BenchmarkError(DeployError):
    """Edge benchmark execution failed."""
