#!/usr/bin/env python3
"""Vision Workbench — One-click object detection pipeline.

Usage:
    python run_pipeline.py                        # default config
    python run_pipeline.py configs/my_config.yaml # custom config
    python run_pipeline.py --dry-run              # preview only
"""

import sys
from pathlib import Path

from vision_workbench.core.config import PipelineConfig
from vision_workbench.core.context import PipelineContext
from vision_workbench.core.orchestrator import PipelineOrchestrator
from vision_workbench.pipeline import discover_stages


def main(config_path: str = "configs/detect.yaml", dry_run: bool = False) -> PipelineContext:
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"Error: Config file not found: {config_file}")
        print("Usage: python run_pipeline.py [config.yaml] [--dry-run]")
        sys.exit(1)

    print(f"Loading config: {config_file}")
    config = PipelineConfig.from_yaml(str(config_file))
    print(f"Pipeline: {config.name}")
    print(f"Stages: {config.stages}")

    stages = discover_stages()
    if not stages:
        print("Warning: No pipeline stages discovered. Check your installation.")
        sys.exit(1)
    print(f"Discovered stages: {list(stages.keys())}")

    orchestrator = PipelineOrchestrator(config)
    for stage_cls in stages.values():
        orchestrator.register_stage(stage_cls())

    if dry_run:
        print("\n--- Dry Run ---")
        plan = orchestrator.dry_run()
        for step in plan:
            status = "✓" if step["ok"] else "✗"
            print(f"  [{status}] {step['stage']}: {step.get('summary', step.get('error', ''))}")
        return PipelineContext()

    print("\n--- Running Pipeline ---")
    ctx = orchestrator.run()
    print(f"\nPipeline complete. Context version: {ctx.version}")
    print(f"Artifacts: {ctx.artifacts}")
    print(f"Metrics: {ctx.metrics}")
    return ctx


if __name__ == "__main__":
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    config = next((a for a in args if not a.startswith("--")), "configs/detect.yaml")
    main(config, dry_run)
