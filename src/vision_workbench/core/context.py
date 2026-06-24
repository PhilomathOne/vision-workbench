"""Immutable pipeline context — the data bus between pipeline stages.

Each stage reads from and writes to the context. Context snapshots are
immutable — every mutation produces a new snapshot for traceability.
"""

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


class ContextKey:
    """Standard context key constants to prevent string typos."""

    DATASET_DIR = "artifacts.dataset_dir"
    ANNOTATIONS = "artifacts.annotations"
    CHECKPOINT_PATH = "artifacts.checkpoint_path"
    VAL_METRICS = "metrics.validation"
    EVAL_REPORT = "artifacts.evaluation_report"
    OPTIMIZED_MODEL = "artifacts.optimized_model"
    EXPORTED_MODELS = "artifacts.exports"
    DEPLOY_STATUS = "metadata.deploy_status"
    DATA_PROVENANCE = "metadata.data_provenance"
    STAGE_HISTORY = "metadata.stage_history"
    ZOO_MODEL = "artifacts.zoo_model"
    DEPLOY_PACKAGE = "artifacts.deploy_package"


class PipelineContext(BaseModel, frozen=True):
    """Immutable context snapshot passed between pipeline stages.

    Design:
    - Frozen (immutable) for traceability — every stage produces a new version.
    - Dotted-path keys support nested access (e.g., "artifacts.dataset_dir").
    - version/parent_version form a linked list for audit trails.
    """

    # Artifact path index (dotted-path → value)
    artifacts: dict[str, Any] = Field(default_factory=dict)

    # Numerical metrics (dotted-path → float or nested dict)
    metrics: dict[str, Any] = Field(default_factory=dict)

    # Text/structured metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Execution audit trail
    stage_history: list[dict] = Field(default_factory=list)

    # Version info for traceability
    version: int = 0
    parent_version: Optional[int] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    created_by: str = ""

    def evolve(self, **updates: Any) -> "PipelineContext":
        """Create a new context snapshot with the given updates.

        Supports dotted-path keys for nested updates::

            ctx.evolve(**{"artifacts.dataset_dir": "/path/to/data"})
        """
        data = self.model_dump()
        for key, value in updates.items():
            PipelineContext._set_nested(data, key, value)
        data["version"] = self.version + 1
        data["parent_version"] = self.version
        data["created_at"] = datetime.now(timezone.utc).isoformat()
        return PipelineContext(**data)

    def get(self, key: str, default: Any = None) -> Any:
        """Read a nested value by dotted-path key."""
        keys = key.split(".")
        current: Any = self.model_dump()
        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError, IndexError):
            return default

    def record_stage(self, stage_name: str, success: bool, **meta: Any) -> "PipelineContext":
        """Record a stage execution in the history."""
        entry = {
            "stage": stage_name,
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **meta,
        }
        history = list(self.metadata.get("stage_history", [])) + [entry]
        return self.evolve(**{"metadata.stage_history": history})

    @staticmethod
    def _set_nested(data: dict, key: str, value: Any) -> None:
        """Set a nested dict value via dotted-path key."""
        keys = key.split(".")
        d = data
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value
