"""Stage 5: Deployment package generation."""

import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import structlog
import yaml

from vision_workbench.core.base import BaseStage
from vision_workbench.core.config import DeployStageConfig
from vision_workbench.core.context import ContextKey, PipelineContext
from vision_workbench.core.result import StageResult

logger = structlog.get_logger(__name__)


class DeployStage(BaseStage[DeployStageConfig, StageResult]):
    """Generate a deployment package from exported models.

    Produces a self-contained deployment directory with model files,
    manifest, and optional inference script template.
    """

    name = "deploy"
    description = "Generate deployment package with model files and manifest"
    depends_on = ["export"]

    def validate_inputs(self, ctx: PipelineContext) -> tuple[bool, list[str]]:
        missing = []
        if not ctx.get(ContextKey.EXPORTED_MODELS):
            missing.append(ContextKey.EXPORTED_MODELS)
        return len(missing) == 0, missing

    def run(self, config: DeployStageConfig, ctx: PipelineContext) -> tuple[StageResult, PipelineContext]:
        t0 = time.perf_counter()
        exports = ctx.get(ContextKey.EXPORTED_MODELS, {})
        workspace = Path(ctx.get("runtime.workspace", "./vw_workspace"))
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        deploy_dir = workspace / "deployments" / timestamp
        deploy_dir.mkdir(parents=True, exist_ok=True)

        # Copy exported models
        models_dir = deploy_dir / "models"
        models_dir.mkdir(exist_ok=True)
        for fmt, path in exports.items():
            src = Path(path)
            if src.exists():
                dst = models_dir / f"model.{src.suffix.lstrip('.')}" if src.suffix else models_dir / f"model_{fmt}"
                shutil.copy2(src, dst)
                logger.info("deploy.copied", format=fmt, path=str(dst))

        # Copy dataset class names if available
        if ctx.get(ContextKey.DATASET_DIR):
            dataset_yaml = Path(ctx.get(ContextKey.DATASET_DIR)) / "dataset.yaml"
            if dataset_yaml.exists():
                shutil.copy2(dataset_yaml, deploy_dir / "classes.yaml")

        # Generate manifest
        manifest = {
            "deployed_at": timestamp,
            "stages_completed": [s["stage"] for s in ctx.metadata.get("stage_history", [])],
            "models": {fmt: str(Path(p).name) if Path(p).exists() else p for fmt, p in exports.items()},
            "metrics": ctx.get(ContextKey.VAL_METRICS, {}),
        }
        with open(deploy_dir / "deploy_manifest.yaml", "w") as f:
            yaml.dump(manifest, f, default_flow_style=False, allow_unicode=True)

        dt = time.perf_counter() - t0
        result = StageResult(
            stage_name="deploy", success=True, duration_seconds=dt,
            artifacts={"deploy_dir": str(deploy_dir)},
            metrics={"files_copied": len(exports)},
        )
        ctx = ctx.record_stage("deploy", success=True, duration_seconds=dt)
        return result, ctx

    def dry_run(self, config: DeployStageConfig, ctx: PipelineContext) -> dict:
        exports = ctx.get(ContextKey.EXPORTED_MODELS, {})
        return {"action": "generate_deployment", "model_count": len(exports)}
