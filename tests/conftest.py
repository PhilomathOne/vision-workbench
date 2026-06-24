"""Shared pytest fixtures for Vision Workbench tests."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from vision_workbench.core.context import PipelineContext


@pytest.fixture
def sample_image() -> np.ndarray:
    """A 640x480 BGR test image with simple geometric shapes."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Red rectangle (face-like)
    img[100:300, 150:350, 2] = 255
    # Blue rectangle
    img[50:200, 400:550, 0] = 255
    return img


@pytest.fixture
def empty_context() -> PipelineContext:
    """A fresh, empty PipelineContext."""
    return PipelineContext(created_by="test")


@pytest.fixture
def tmp_workspace() -> Path:
    """Temporary workspace directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
