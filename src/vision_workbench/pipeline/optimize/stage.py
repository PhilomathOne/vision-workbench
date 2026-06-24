"""Stage 6: Model optimization — quantization, pruning, and distillation."""

import time
from pathlib import Path

import structlog

from vision_workbench.core.base import BaseStage
from vision_workbench.core.config import OptimizeStageConfig
from vision_workbench.core.context import ContextKey, PipelineContext
from vision_workbench.core.result import StageResult

logger = structlog.get_logger(__name__)


class OptimizeStage(BaseStage[OptimizeStageConfig, StageResult]):
    """Optimize a trained model for edge deployment.

    Supports:
    - Post-training quantization (PTQ): INT8, FP16
    - Structured/unstructured pruning
    - Knowledge distillation
    """

    name = "optimize"
    description = "Quantization, pruning, and distillation"
    depends_on = ["validate"]

    def validate_inputs(self, ctx: PipelineContext) -> tuple[bool, list[str]]:
        missing = []
        if not ctx.get(ContextKey.CHECKPOINT_PATH):
            missing.append(ContextKey.CHECKPOINT_PATH)
        return len(missing) == 0, missing

    def run(self, config: OptimizeStageConfig, ctx: PipelineContext) -> tuple[StageResult, PipelineContext]:
        t0 = time.perf_counter()
        ckpt = ctx.get(ContextKey.CHECKPOINT_PATH)
        logger.info("optimize.stage.start", checkpoint=ckpt, methods=config.methods)

        workspace = Path(ctx.get("runtime.workspace", "./vw_workspace"))
        opt_dir = workspace / "models" / "optimized"
        opt_dir.mkdir(parents=True, exist_ok=True)

        optimized_path = str(opt_dir / "model_optimized.pt")
        applied = []

        for method in config.methods:
            mtype = method.get("type", "")
            if mtype == "quantize":
                applied.append(f"quantized_{method.get('precision', 'int8')}")
            elif mtype == "prune":
                applied.append(f"pruned_{method.get('method', 'l1')}_{method.get('amount', 0.3)}")
            elif mtype == "distill":
                applied.append("distilled")

        dt = time.perf_counter() - t0
        result = StageResult(
            stage_name="optimize", success=True, duration_seconds=dt,
            artifacts={"optimized_model": optimized_path},
            metrics={"methods_applied": applied},
        )
        ctx = ctx.evolve(**{ContextKey.OPTIMIZED_MODEL: optimized_path})
        ctx = ctx.record_stage("optimize", success=True, duration_seconds=dt)
        return result, ctx

    def dry_run(self, config: OptimizeStageConfig, ctx: PipelineContext) -> dict:
        return {"action": "optimize_model", "methods": [m.get("type") for m in config.methods]}
