"""OpenCV Haar Cascade face detector — zero extra dependencies."""

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vision_workbench.core.exceptions import InferenceError
from vision_workbench.core.result import BoundingBox, DetectionResult
from vision_workbench.core.registry import detector_registry
from vision_workbench.detectors.base import BaseDetector


@detector_registry.register("opencv_haar_face", task="face_detection", framework="opencv", description="Haar Cascade face detection (built-in, zero deps)")
class HaarFaceDetector(BaseDetector):
    """Face detection using OpenCV's built-in Haar cascades."""

    task_type = "face_detection"
    framework = "opencv"

    def __init__(self) -> None:
        self._cascade: cv2.CascadeClassifier | None = None

    def initialize(self, **params: Any) -> None:
        cascade_path = params.get("cascade_path")
        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(str(cascade_path))
        if self._cascade.empty():
            raise InferenceError(f"Failed to load Haar cascade: {cascade_path}")

    def process(self, image: np.ndarray) -> DetectionResult:
        if self._cascade is None:
            raise InferenceError("Detector not initialized. Call initialize() first.")
        t0 = time.perf_counter()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        boxes = [BoundingBox(x1=float(x), y1=float(y), x2=float(x + w), y2=float(y + h), class_name="face") for (x, y, w, h) in faces]
        dt = (time.perf_counter() - t0) * 1000
        return DetectionResult(
            source_path=None,
            image_shape=image.shape,
            boxes=boxes,
            detector_name="opencv_haar_face",
            task_type="face_detection",
            framework="opencv",
            processing_time_ms=dt,
        )

    def cleanup(self) -> None:
        self._cascade = None
