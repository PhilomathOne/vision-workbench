"""Pipeline orchestrator — DAG scheduler, parallel executor, and resumer."""

from typing import Optional

import structlog

from vision_workbench.core.base import BaseStage
from vision_workbench.core.config import PipelineConfig
from vision_workbench.core.context import PipelineContext
from vision_workbench.core.exceptions import StageInputError, StageNotFoundError

logger = structlog.get_logger(__name__)


class PipelineOrchestrator:
    """Executes pipeline stages in order, managing context flow.

    Responsibilities:
    - Resolve stage names to stage instances.
    - Run stages sequentially, passing context between them.
    - Support dry-run preview of the execution plan.
    - Support resume from a specific stage.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._stage_registry: dict[str, BaseStage] = {}

    def register_stage(self, stage: BaseStage) -> None:
        """Register a stage instance for use in pipelines."""
        self._stage_registry[stage.name] = stage

    def resolve_stages(self) -> list[BaseStage]:
        """Resolve the stage names from config to stage instances."""
        stages: list[BaseStage] = []
        for name in self.config.stages:
            if name not in self._stage_registry:
                raise StageNotFoundError(
                    f"Stage '{name}' is not registered. "
                    f"Available: {list(self._stage_registry.keys())}"
                )
            stages.append(self._stage_registry[name])
        return stages

    def run(self, ctx: Optional[PipelineContext] = None) -> PipelineContext:
        """Execute all configured stages in order.

        Args:
            ctx: Optional initial context. Created fresh if not provided.

        Returns:
            Final PipelineContext after all stages complete.
        """
        if ctx is None:
            ctx = PipelineContext(created_by="orchestrator")

        stages = self.resolve_stages()
        logger.info(
            "pipeline.start",
            name=self.config.name,
            stages=[s.name for s in stages],
        )

        for stage in stages:
            logger.info("stage.starting", stage=stage.name)

            # Validate inputs
            ok, missing = stage.validate_inputs(ctx)
            if not ok:
                raise StageInputError(
                    f"Stage '{stage.name}' missing inputs: {missing}"
                )

            # Execute
            result, ctx = stage.run(
                getattr(self.config, stage.name, None), ctx
            )
            ctx = ctx.record_stage(
                stage.name,
                success=result.success,
                duration_seconds=result.duration_seconds,
            )

            logger.info(
                "stage.complete",
                stage=stage.name,
                success=result.success,
                duration_s=result.duration_seconds,
            )

        logger.info("pipeline.complete", name=self.config.name, version=ctx.version)
        return ctx

    def run_from(self, from_stage: str, ctx: PipelineContext) -> PipelineContext:
        """Resume execution from a specific stage (inclusive).

        All completed stages before `from_stage` are skipped.
        """
        stages = self.resolve_stages()
        start_idx = None
        for i, stage in enumerate(stages):
            if stage.name == from_stage:
                start_idx = i
                break

        if start_idx is None:
            raise StageNotFoundError(
                f"Stage '{from_stage}' not found in pipeline stages: "
                f"{[s.name for s in stages]}"
            )

        logger.info("pipeline.resume", from_stage=from_stage)
        for stage in stages[start_idx:]:
            result, ctx = stage.run(
                getattr(self.config, stage.name, None), ctx
            )
            ctx = ctx.record_stage(stage.name, success=result.success)

        return ctx

    def dry_run(self) -> list[dict]:
        """Preview execution plan without running stages.

        Returns:
            List of stage summaries describing what each would do.
        """
        ctx = PipelineContext(created_by="dry-run")
        stages = self.resolve_stages()
        plan: list[dict] = []

        for stage in stages:
            stage_config = getattr(self.config, stage.name, None)
            try:
                summary = stage.dry_run(stage_config, ctx)
                plan.append({"stage": stage.name, "ok": True, "summary": summary})
            except Exception as e:
                plan.append({"stage": stage.name, "ok": False, "error": str(e)})

        return plan
