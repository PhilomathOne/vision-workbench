"""Stage 2: YOLO training via Ultralytics."""

import time
from pathlib import Path

import structlog

from vision_workbench.core.base import BaseStage
from vision_workbench.core.config import TrainStageConfig
from vision_workbench.core.context import ContextKey, PipelineContext
from vision_workbench.core.exceptions import StageExecutionError
from vision_workbench.core.result import StageResult

logger = structlog.get_logger(__name__)


class TrainStage(BaseStage[TrainStageConfig, StageResult]):
    """Train a YOLO model using Ultralytics.

    Ultralytics auto-downloads pre-trained weights, so no ModelZoo needed.
    """

    name = "train"
    description = "YOLO training via Ultralytics"
    depends_on = ["data"]

    def validate_inputs(self, ctx: PipelineContext) -> tuple[bool, list[str]]:
        missing = []
        if not ctx.get(ContextKey.DATASET_DIR):
            missing.append(ContextKey.DATASET_DIR)
        return len(missing) == 0, missing

    def run(self, config: TrainStageConfig, ctx: PipelineContext) -> tuple[StageResult, PipelineContext]:
        t0 = time.perf_counter()
        dataset_dir = Path(ctx.get(ContextKey.DATASET_DIR, "."))
        dataset_yaml = dataset_dir / "dataset.yaml"

        if not dataset_yaml.exists():
            dt = time.perf_counter() - t0
            return StageResult(stage_name="train", success=False, error_message=f"dataset.yaml not found at {dataset_yaml}", duration_seconds=dt), ctx

        try:
            from ultralytics import YOLO
        except ImportError:
            raise StageExecutionError("Ultralytics not installed. Run: pip install ultralytics")

        model_name = f"{config.model}.pt" if config.pretrained else f"{config.model}.yaml"
        device = config.device or ctx.get("runtime.device", "auto")
        logger.info("train.start", model=config.model, epochs=config.epochs, device=device, data=str(dataset_yaml))

        model = YOLO(model_name)
        results = model.train(
            data=str(dataset_yaml),
            epochs=config.epochs,
            batch=config.batch,
            imgsz=config.imgsz,
            lr0=config.lr0,
            optimizer=config.optimizer,
            device=device,
            seed=ctx.get("runtime.seed", 42),
            verbose=True,
        )

        # Find best checkpoint
        workspace = Path(ctx.get("runtime.workspace", "./vw_workspace"))
        # Ultralytics saves to runs/detect/<exp_name>/weights/best.pt by default
        ckpt_dir = workspace / "models" / "checkpoints" / config.model
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Copy best.pt from Ultralytics default location
        import shutil
        ultralytics_runs = sorted(Path("runs").glob("detect/*"), key=lambda p: p.stat().st_mtime, reverse=True)
        best_src = None
        if ultralytics_runs:
            best_src = ultralytics_runs[0] / "weights" / "best.pt"
        if best_src and best_src.exists():
            shutil.copy2(best_src, ckpt_dir / "best.pt")
            shutil.copy2(ultralytics_runs[0] / "weights" / "last.pt", ckpt_dir / "last.pt")
            best_path = ckpt_dir / "best.pt"
        else:
            best_path = ckpt_dir / "best.pt"

        dt = time.perf_counter() - t0
        result = StageResult(
            stage_name="train", success=True, duration_seconds=dt,
            artifacts={"checkpoint_path": str(best_path)},
            metrics={"epochs_completed": config.epochs, "model": config.model},
        )
        ctx = ctx.evolve(**{ContextKey.CHECKPOINT_PATH: str(best_path)})
        ctx = ctx.record_stage("train", success=True, duration_seconds=dt)
        return result, ctx

    def dry_run(self, config: TrainStageConfig, ctx: PipelineContext) -> dict:
        return {"action": "train_yolo", "model": config.model, "epochs": config.epochs, "device": config.device or "auto"}
