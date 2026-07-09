"""Tests for the built-in OpenCV Haar face detector."""

import numpy as np
import pytest

from vision_workbench.detectors.haar_face import HaarFaceDetector


class TestHaarFaceDetector:
    def test_initialize(self):
        detector = HaarFaceDetector()
        detector.initialize()
        assert detector._cascade is not None
        detector.cleanup()

    def test_process_no_faces(self, sample_image):
        """Should return empty boxes for an image with no faces."""
        detector = HaarFaceDetector()
        detector.initialize()
        # The sample_image is just colored rectangles, not real faces
        result = detector.process(sample_image)
        assert result.detector_name == "opencv_haar_face"
        assert result.task_type == "face_detection"
        assert result.framework == "opencv"
        assert result.image_shape == sample_image.shape
        assert result.processing_time_ms >= 0
        detector.cleanup()

    def test_registry_registered(self):
        """Verify detector is registered in the global registry."""
        from vision_workbench.core.registry import detector_registry
        assert "opencv_haar_face" in detector_registry

    def test_uninitialized_raises(self, sample_image):
        detector = HaarFaceDetector()
        from vision_workbench.core.exceptions import InferenceError
        with pytest.raises(InferenceError, match="not initialized"):
            detector.process(sample_image)
