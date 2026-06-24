"""Tests for PipelineContext — the immutable data bus."""

import pytest

from vision_workbench.core.context import ContextKey, PipelineContext


class TestPipelineContext:
    def test_initial_state(self):
        ctx = PipelineContext(created_by="test")
        assert ctx.version == 0
        assert ctx.parent_version is None
        assert ctx.created_by == "test"
        assert ctx.artifacts == {}
        assert ctx.metrics == {}
        assert ctx.stage_history == []

    def test_evolve_creates_new_version(self):
        ctx = PipelineContext(created_by="test")
        ctx2 = ctx.evolve(**{"artifacts.dataset_dir": "/data"})
        assert ctx2.version == 1
        assert ctx2.parent_version == 0
        assert ctx2 is not ctx  # immutable
        assert ctx2.get("artifacts.dataset_dir") == "/data"

    def test_evolve_nested_keys(self):
        ctx = PipelineContext(created_by="test")
        ctx = ctx.evolve(**{"artifacts.dataset_dir": "/data"})
        ctx = ctx.evolve(**{"metrics.validation.mAP": 0.85})
        assert ctx.get("artifacts.dataset_dir") == "/data"
        assert ctx.get("metrics.validation.mAP") == 0.85

    def test_get_default(self):
        ctx = PipelineContext(created_by="test")
        assert ctx.get("nonexistent.key", "default") == "default"

    def test_get_nested(self):
        ctx = PipelineContext(created_by="test")
        ctx = ctx.evolve(**{
            "artifacts.dataset_dir": "/data",
            "artifacts.checkpoint_path": "/ckpt/best.pt",
        })
        assert ctx.get("artifacts.dataset_dir") == "/data"
        assert ctx.get("artifacts.checkpoint_path") == "/ckpt/best.pt"

    def test_record_stage(self):
        ctx = PipelineContext(created_by="test")
        ctx2 = ctx.record_stage("data", success=True, duration=1.5)
        history = ctx2.get("metadata.stage_history")
        assert len(history) == 1
        assert history[0]["stage"] == "data"
        assert history[0]["success"] is True
        assert history[0]["duration"] == 1.5

    def test_record_multiple_stages(self):
        ctx = PipelineContext(created_by="test")
        ctx = ctx.record_stage("data", success=True)
        ctx = ctx.record_stage("train", success=True)
        history = ctx.get("metadata.stage_history")
        assert len(history) == 2
        assert history[0]["stage"] == "data"
        assert history[1]["stage"] == "train"

    def test_version_chain(self):
        ctx = PipelineContext(created_by="test")
        assert ctx.version == 0
        ctx = ctx.evolve(**{"x": 1})
        assert ctx.version == 1
        assert ctx.parent_version == 0
        ctx = ctx.evolve(**{"y": 2})
        assert ctx.version == 2
        assert ctx.parent_version == 1

    def test_context_key_constants(self):
        """Ensure ContextKey constants exist and are strings."""
        assert ContextKey.DATASET_DIR.startswith("artifacts")
        assert ContextKey.CHECKPOINT_PATH.startswith("artifacts")
        assert ContextKey.VAL_METRICS.startswith("metrics")
