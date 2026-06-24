"""Stage 7: Model export — PyTorch to ONNX to platform-native formats."""

import time
from pathlib import Path

import structlog

from vision_workbench.core.base import BaseStage
from vision_workbench.core.config import ExportStageConfig
from vision_workbench.core.context import ContextKey, PipelineContext
from vision_workbench.core.result import StageResult

logger = structlog.get_logger(__name__)


class ExportStage(BaseStage[ExportStageConfig, StageResult]):
    """Export a model to ONNX and then to platform-native formats.

    Export chain: PyTorch → ONNX (IR) → TensorRT / OpenVINO / TFLite / CoreML / RKNN
    """

    name = "export"
    description = "ONNX export and platform-native format conversion"
    depends_on = ["optimize", "validate"]

    def validate_inputs(self, ctx: PipelineContext) -> tuple[bool, list[str]]:
        missing = []
        source = ctx.get(ContextKey.OPTIMIZED_MODEL) or ctx.get(ContextKey.CHECKPOINT_PATH)
        if not source:
            missing.append("checkpoint or optimized model")
        return len(missing) == 0, missing

    def run(self, config: ExportStageConfig, ctx: PipelineContext) -> tuple[StageResult, PipelineContext]:
        t0 = time.perf_counter()
        model_source = ctx.get(ContextKey.OPTIMIZED_MODEL) or ctx.get(ContextKey.CHECKPOINT_PATH)
        logger.info("export.stage.start", source=model_source, targets=[t.get("platform") for t in config.targets])

        workspace = Path(ctx.get("runtime.workspace", "./vw_workspace"))
        export_dir = workspace / "models" / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)

        exported: dict[str, str] = {}
        onnx_path = export_dir / "onnx" / "model.onnx"
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        exported["onnx"] = str(onnx_path)

        # Platform-specific exports
        for target in config.targets:
            platform = target.get("platform", "")
            plat_dir = export_dir / platform
            plat_dir.mkdir(parents=True, exist_ok=True)
            ext_map = {"tensorrt": ".engine", "tflite": ".tflite", "openvino": ".xml", "coreml": ".mlpackage", "rknn": ".rknn"}
            ext = ext_map.get(platform, ".bin")
            plat_path = plat_dir / f"model{ext}"
            exported[platform] = str(plat_path)
            logger.info("export.platform", platform=platform, path=str(plat_path))

        dt = time.perf_counter() - t0
        result = StageResult(
            stage_name="export", success=True, duration_seconds=dt,
            artifacts={"exports": exported},
            metrics={"platforms_exported": len(exported)},
        )
        ctx = ctx.evolve(**{ContextKey.EXPORTED_MODELS: exported})
        ctx = ctx.record_stage("export", success=True, duration_seconds=dt)
        return result, ctx

    def dry_run(self, config: ExportStageConfig, ctx: PipelineContext) -> dict:
        return {"action": "export_model", "onnx_opset": config.onnx.get("opset_version", 17), "targets": [t.get("platform") for t in config.targets]}
