"""Integration tests for pipeline stage discovery and orchestration."""

from pathlib import Path
import tempfile

import numpy as np
import pytest
from PIL import Image

from vision_workbench.core.config import (
    DataStageConfig,
    PipelineConfig,
    RuntimeConfig,
)
from vision_workbench.core.context import PipelineContext
from vision_workbench.core.orchestrator import PipelineOrchestrator


class TestPipelineIntegration:
    def test_stage_discovery(self):
        """All built-in pipeline stages should be discoverable."""
        from vision_workbench.pipeline import discover_stages
        stages = discover_stages()
        expected = {"data", "annotate", "train", "validate", "evaluate", "optimize", "export", "deploy"}
        found = set(stages.keys())
        assert expected == found, f"Missing stages: {expected - found}"

    def test_data_stage_end_to_end(self):
        """DataStage: create a mini dataset, run data stage, verify output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "raw"
            dst = Path(tmpdir) / "processed"
            src.mkdir()

            # Create 5 simple test images
            for i in range(5):
                img = Image.new("RGB", (100, 100), color=(i * 40, 100, 200))
                img.save(src / f"img_{i:03d}.jpg")

            config = DataStageConfig(
                enabled=True,
                source=str(src),
                target=str(dst),
                dataset_name="test-dataset",
                validation={"min_resolution": [1, 1], "max_resolution": [4096, 4096], "allowed_formats": [".jpg", ".jpeg", ".png", ".bmp"], "blur_threshold": 0},
                dedup={"enabled": False},
            )
            ctx = PipelineContext(created_by="test")
            from vision_workbench.pipeline.data.stage import DataStage

            stage = DataStage()
            result, ctx = stage.run(config, ctx)
            assert result.success
            assert result.metrics["total"] == 5
            assert ctx.get("artifacts.dataset_dir") == str(dst)
            assert (dst / "dataset.yaml").exists()
            # Check split directories exist
            assert (dst / "images" / "train").exists()

    def test_orchestrator_with_data_stage(self):
        """PipelineOrchestrator should run a single-stage pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "raw"
            dst = Path(tmpdir) / "processed"
            src.mkdir()
            for i in range(3):
                img = Image.new("RGB", (64, 64), color=(255, 0, 0))
                img.save(src / f"img_{i}.jpg")

            config = PipelineConfig(
                name="test-pipeline",
                stages=["data"],
                runtime=RuntimeConfig(workspace=Path(tmpdir)),
                data=DataStageConfig(enabled=True, source=str(src), target=str(dst), dataset_name="mini"),
            )
            from vision_workbench.pipeline.data.stage import DataStage

            orch = PipelineOrchestrator(config)
            orch.register_stage(DataStage())
            ctx = PipelineContext(created_by="test")
            ctx = ctx.evolve(**{"metadata.data_source": str(src)})
            ctx = orch.run(ctx)
            assert ctx.version >= 1
            assert ctx.get("artifacts.dataset_dir") == str(dst)

    def test_pipeline_context_round_trip(self):
        """PipelineContext should preserve data through a simulated multi-stage run."""
        ctx = PipelineContext(created_by="test")

        # Simulate data stage output
        ctx = ctx.evolve(**{"artifacts.dataset_dir": "/tmp/data"})
        ctx = ctx.record_stage("data", success=True)

        # Simulate train stage output
        ctx = ctx.evolve(**{"artifacts.checkpoint_path": "/tmp/ckpt/best.pt"})
        ctx = ctx.record_stage("train", success=True)

        # Simulate export stage output
        ctx = ctx.evolve(**{"artifacts.exports": {"onnx": "/tmp/exports/model.onnx", "tensorrt": "/tmp/exports/model.engine"}})
        ctx = ctx.record_stage("export", success=True)

        # Verify the full chain
        assert ctx.get("artifacts.dataset_dir") == "/tmp/data"
        assert ctx.get("artifacts.checkpoint_path") == "/tmp/ckpt/best.pt"
        assert ctx.get("artifacts.exports")["onnx"] == "/tmp/exports/model.onnx"
        history = ctx.get("metadata.stage_history")
        assert len(history) == 3

    def test_config_load_from_yaml(self):
        """PipelineConfig should load from a YAML file."""
        import yaml

        import os
        tmp_path = Path(tempfile.mktemp(suffix=".yaml"))
        try:
            with open(tmp_path, "w") as f:
                yaml.dump({
                    "name": "test",
                    "stages": ["data"],
                    "runtime": {"workspace": "./test_ws", "seed": 99},
                    "data": {"enabled": True, "source": "./src", "target": "./dst", "dataset_name": "ds"},
                }, f)
            config = PipelineConfig.from_yaml(str(tmp_path))
            assert config.name == "test"
            assert config.runtime.seed == 99
            assert config.data.enabled is True
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
