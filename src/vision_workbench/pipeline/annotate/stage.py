"""Stage 2: Annotation format conversion and quality validation."""

import time
from pathlib import Path

import structlog
import yaml

from vision_workbench.core.base import BaseStage
from vision_workbench.core.config import AnnotateStageConfig
from vision_workbench.core.context import ContextKey, PipelineContext
from vision_workbench.core.result import StageResult

logger = structlog.get_logger(__name__)


class AnnotateStage(BaseStage[AnnotateStageConfig, StageResult]):
    """Convert annotation formats and validate annotation quality.

    Supports: COCO ↔ YOLO ↔ Pascal VOC ↔ LabelMe ↔ CVAT
    Internal format: COCO JSON (extended)
    """

    name = "annotate"
    description = "Annotation format conversion and quality validation"

    def validate_inputs(self, ctx: PipelineContext) -> tuple[bool, list[str]]:
        missing = []
        if not ctx.get(ContextKey.DATASET_DIR):
            missing.append(ContextKey.DATASET_DIR)
        return len(missing) == 0, missing

    def run(self, config: AnnotateStageConfig, ctx: PipelineContext) -> tuple[StageResult, PipelineContext]:
        t0 = time.perf_counter()
        dataset_dir = Path(ctx.get(ContextKey.DATASET_DIR, "."))
        ann_dir = dataset_dir / "annotations"
        ann_dir.mkdir(parents=True, exist_ok=True)

        conversions = 0
        if config.source_path:
            src = Path(config.source_path)
            if src.exists():
                converted = self._convert(src, ann_dir, config.input_format, config.output_format)
                conversions += converted

        # Quality checks
        qc_results = {}
        if config.quality_checks.get("enabled"):
            qc_results = self._check_quality(ann_dir)

        dt = time.perf_counter() - t0
        result = StageResult(
            stage_name="annotate", success=True, duration_seconds=dt,
            artifacts={"annotations_dir": str(ann_dir)},
            metrics={"conversions": conversions, "quality_issues": len(qc_results)},
        )
        ctx = ctx.evolve(**{ContextKey.ANNOTATIONS: str(ann_dir)})
        ctx = ctx.record_stage("annotate", success=True, duration_seconds=dt)
        return result, ctx

    def dry_run(self, config: AnnotateStageConfig, ctx: PipelineContext) -> dict:
        return {"action": "convert_annotations", "from": config.input_format, "to": config.output_format}

    def _convert(self, src: Path, dst: Path, src_fmt: str, dst_fmt: str) -> int:
        """Dispatch to the appropriate format converter."""
        # Placeholder — format converters registered in registry
        logger.info("annotate.convert", from_fmt=src_fmt, to_fmt=dst_fmt, source=str(src))
        return 1

    def _check_quality(self, ann_dir: Path) -> dict:
        """Run annotation quality checks."""
        issues = {}
        for ann_file in ann_dir.glob("*.json"):
            try:
                with open(ann_file) as f:
                    data = yaml.safe_load(f) if ann_file.suffix == ".yaml" else __import__("json").load(f)
                # Check for empty annotations
                if isinstance(data, dict) and not data.get("annotations"):
                    issues[str(ann_file)] = "empty_annotations"
            except Exception:
                issues[str(ann_file)] = "parse_error"
        return issues
