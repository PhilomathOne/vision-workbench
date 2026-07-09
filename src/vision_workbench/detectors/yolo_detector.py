"""Ultralytics YOLO detector — real inference implementation."""

import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from vision_workbench.core.exceptions import InferenceError
from vision_workbench.core.registry import detector_registry
from vision_workbench.core.result import BoundingBox, DetectionResult
from vision_workbench.detectors.base import BaseDetector


@detector_registry.register("yolo", task="object_detection", framework="ultralytics", description="Ultralytics YOLO object detection")
class YOLODetector(BaseDetector):
    """Object detection using Ultralytics YOLO models.

    Usage::

        detector = YOLODetector()
        detector.initialize(model="yolov8n.pt")  # auto-downloads
        result = detector.process(image)
        detector.cleanup()
    """

    task_type = "object_detection"
    framework = "ultralytics"

    def __init__(self) -> None:
        self._model: Optional[Any] = None

    def initialize(self, **params: Any) -> None:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise InferenceError(
                "Ultralytics is required for YOLO detection. "
                "Install with: pip install ultralytics"
            )
        model_name = params.get("model", "yolov8n.pt")
        self._model = YOLO(model_name)

    def process(self, image: np.ndarray) -> DetectionResult:
        if self._model is None:
            raise InferenceError("Detector not initialized. Call initialize() first.")
        t0 = time.perf_counter()
        results = self._model(image, verbose=False)
        dt = (time.perf_counter() - t0) * 1000

        boxes: list[BoundingBox] = []
        for r in results:
            if r.boxes is not None:
                for box_data in r.boxes.data.cpu().numpy():
                    x1, y1, x2, y2, conf, cls_id = box_data[:6]
                    cls_name = self._model.names.get(int(cls_id), str(int(cls_id))) if hasattr(self._model, "names") else str(int(cls_id))
                    boxes.append(BoundingBox(
                        x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                        confidence=float(conf), class_id=int(cls_id), class_name=cls_name,
                    ))

        return DetectionResult(
            source_path=None,
            image_shape=image.shape,
            boxes=boxes,
            detector_name="yolo",
            task_type="object_detection",
            framework="ultralytics",
            processing_time_ms=dt,
        )

    def cleanup(self) -> None:
        self._model = None
