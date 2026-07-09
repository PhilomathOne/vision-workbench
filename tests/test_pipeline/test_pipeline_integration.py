"""Integration tests for pipeline stage discovery and orchestration."""

import tempfile
from pathlib import Path

import yaml
from PIL import Image

from vision_workbench.core.config import DataStageConfig, PipelineConfig, RuntimeConfig
from vision_workbench.core.context import PipelineContext
from vision_workbench.core.orchestrator import PipelineOrchestrator


class TestPipelineIntegration:
    def test_stage_discovery(self):
        from vision_workbench.pipeline import discover_stages
        stages = discover_stages()
        expected = {"data", "train", "validate", "export", "deploy"}
        found = set(stages.keys())
        assert expected == found, f"Missing: {expected - found}, Extra: {found - expected}"

    def test_data_stage_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "raw"
            dst = Path(tmpdir) / "processed"
            src.mkdir()
            for i in range(10):
                img = Image.new("RGB", (200, 200), color=(i * 25, 100, 200))
                img.save(src / f"img_{i:03d}.jpg")
                # Create YOLO label for half of the images
                if i < 5:
                    label_path = src / f"img_{i:03d}.txt"
                    label_path.write_text("0 0.5 0.5 0.3 0.3\n1 0.2 0.2 0.1 0.1\n")

            config = DataStageConfig(
                source=str(src), target=str(dst), classes=["person", "car"],
                split={"train": 0.7, "val": 0.2, "test": 0.1, "seed": 42},
            )
            from vision_workbench.pipeline.data_stage import DataStage
            stage = DataStage()
            ctx = PipelineContext(created_by="test")
            result, ctx = stage.run(config, ctx)
            assert result.success
            assert result.metrics["total_images"] == 10
            assert result.metrics["images_with_annotations"] == 5
            assert (dst / "dataset.yaml").exists()
            assert (dst / "images" / "train").exists()
            assert (dst / "labels" / "train").exists()

    def test_orchestrator_with_data_stage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "raw"
            dst = Path(tmpdir) / "processed"
            src.mkdir()
            for i in range(5):
                img = Image.new("RGB", (100, 100), color=(255, 0, 0))
                img.save(src / f"img_{i}.jpg")

            config = PipelineConfig(
                name="test", stages=["data"],
                runtime=RuntimeConfig(workspace=Path(tmpdir)),
                data=DataStageConfig(source=str(src), target=str(dst), classes=["object"]),
            )
            from vision_workbench.pipeline.data_stage import DataStage
            orch = PipelineOrchestrator(config)
            orch.register_stage(DataStage())
            ctx = orch.run()
            assert ctx.version >= 1

    def test_config_load_from_yaml(self):
        tmp = Path(tempfile.mktemp(suffix=".yaml"))
        try:
            with open(tmp, "w") as f:
                yaml.dump({
                    "name": "test", "stages": ["data"],
                    "runtime": {"workspace": "./test_ws", "seed": 99},
                    "data": {"source": "./src", "target": "./dst", "classes": ["a", "b"]},
                }, f)
            config = PipelineConfig.from_yaml(str(tmp))
            assert config.name == "test"
            assert config.runtime.seed == 99
            assert config.data.classes == ["a", "b"]
        finally:
            if tmp.exists():
                tmp.unlink()

    def test_pipeline_context_round_trip(self):
        ctx = PipelineContext(created_by="test")
        ctx = ctx.evolve(**{"artifacts.dataset_dir": "/tmp/data"})
        ctx = ctx.record_stage("data", success=True)
        ctx = ctx.evolve(**{"artifacts.checkpoint_path": "/tmp/ckpt/best.pt"})
        ctx = ctx.record_stage("train", success=True)
        ctx = ctx.evolve(**{"artifacts.exports": {"onnx": "/tmp/model.onnx"}})
        ctx = ctx.record_stage("export", success=True)
        assert ctx.get("artifacts.dataset_dir") == "/tmp/data"
        assert ctx.get("artifacts.checkpoint_path") == "/tmp/ckpt/best.pt"
        assert ctx.get("artifacts.exports")["onnx"] == "/tmp/model.onnx"
        assert len(ctx.get("metadata.stage_history")) == 3
