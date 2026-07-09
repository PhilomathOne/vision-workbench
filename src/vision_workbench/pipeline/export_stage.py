"""Stage 4: Model export — ONNX + FP16/INT8 quantization."""

import json
import time
from pathlib import Path

import numpy as np
import structlog

from vision_workbench.core.base import BaseStage
from vision_workbench.core.config import ExportStageConfig
from vision_workbench.core.context import ContextKey, PipelineContext
from vision_workbench.core.exceptions import StageExecutionError
from vision_workbench.core.result import StageResult

logger = structlog.get_logger(__name__)


class ExportStage(BaseStage[ExportStageConfig, StageResult]):
    """Export a trained YOLO model to ONNX and other formats.

    Uses Ultralytics model.export() which handles the PyTorch → ONNX
    conversion internally. Optional FP16 quantization via onnxruntime.
    """

    name = "export"
    description = "ONNX export + FP16/INT8 quantization + validation"
    depends_on = ["train"]

    def validate_inputs(self, ctx: PipelineContext) -> tuple[bool, list[str]]:
        missing = []
        if not ctx.get(ContextKey.CHECKPOINT_PATH):
            missing.append(ContextKey.CHECKPOINT_PATH)
        return len(missing) == 0, missing

    def run(self, config: ExportStageConfig, ctx: PipelineContext) -> tuple[StageResult, PipelineContext]:
        t0 = time.perf_counter()
        ckpt = ctx.get(ContextKey.CHECKPOINT_PATH)

        if not Path(ckpt).exists():
            dt = time.perf_counter() - t0
            return StageResult(stage_name="export", success=False, error_message=f"Checkpoint not found: {ckpt}", duration_seconds=dt), ctx

        try:
            from ultralytics import YOLO
        except ImportError:
            raise StageExecutionError("Ultralytics not installed. Run: pip install ultralytics")

        workspace = Path(ctx.get("runtime.workspace", "./vw_workspace"))
        export_dir = workspace / "models" / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        exported: dict[str, str] = {}

        model = YOLO(str(ckpt))
        model_name = Path(ckpt).stem

        for fmt in config.formats:
            fmt_dir = export_dir / fmt
            fmt_dir.mkdir(parents=True, exist_ok=True)
            logger.info("export.exporting", format=fmt, model=model_name)

            kwargs = {}
            if fmt == "onnx":
                kwargs = {"opset": config.opset, "simplify": config.simplify}
                if config.fp16:
                    kwargs["half"] = True
                if config.int8:
                    kwargs["int8"] = True

            exported_path = model.export(format=fmt, **kwargs)

            # Ultralytics puts exports next to the checkpoint by default
            # Copy to our workspace
            import shutil
            src = Path(exported_path) if isinstance(exported_path, str) else Path(str(exported_path))
            if src.exists() and src.parent != fmt_dir:
                dest = fmt_dir / src.name
                shutil.copy2(src, dest)
                exported[fmt] = str(dest)
            else:
                exported[fmt] = str(src)

            logger.info("export.complete", format=fmt, path=exported[fmt])

        # Validate ONNX model
        if "onnx" in exported and config.formats:
            onnx_path = exported.get("onnx", list(exported.values())[0])
            validation = self._validate_onnx(onnx_path)
            logger.info("export.onnx_validated", ok=validation["ok"])

        dt = time.perf_counter() - t0
        result = StageResult(
            stage_name="export", success=True, duration_seconds=dt,
            artifacts={"exports": exported},
            metrics={"formats_exported": list(exported.keys())},
        )
        ctx = ctx.evolve(**{ContextKey.EXPORTED_MODELS: exported})
        ctx = ctx.record_stage("export", success=True, duration_seconds=dt)
        return result, ctx

    def dry_run(self, config: ExportStageConfig, ctx: PipelineContext) -> dict:
        return {"action": "export_model", "formats": config.formats, "fp16": config.fp16}

    def _validate_onnx(self, onnx_path: str) -> dict:
        """Verify the exported ONNX model can be loaded and run."""
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            input_info = session.get_inputs()[0]
            dummy = np.random.randn(*input_info.shape).astype(np.float32)
            outputs = session.run(None, {input_info.name: dummy})
            return {"ok": True, "num_outputs": len(outputs), "output_shapes": [list(o.shape) for o in outputs]}
        except Exception as e:
            return {"ok": False, "error": str(e)}
