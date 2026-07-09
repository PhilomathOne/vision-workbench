"""Vision detectors — auto-register on import."""

from vision_workbench.detectors.base import BaseDetector
from vision_workbench.detectors.haar_face import HaarFaceDetector
from vision_workbench.detectors.yolo_detector import YOLODetector

__all__ = ["BaseDetector", "HaarFaceDetector", "YOLODetector"]
