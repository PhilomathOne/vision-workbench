#!/usr/bin/env python3
"""Vision Workbench — Quick object detection on a single image.

Usage:
    python run_detect.py <image_path> [--detector yolo|haar] [--model yolov8n.pt]
"""

import sys
from pathlib import Path

import cv2

from vision_workbench.core.registry import detector_registry
from vision_workbench.detectors import YOLODetector  # noqa: F401 — register detector
from vision_workbench.detectors.haar_face import HaarFaceDetector  # noqa: F401
from vision_workbench.viz.annotate import annotate


def main(image_path: str, detector_name: str = "yolo", model: str = "yolov8n.pt", output: str = "output.jpg"):
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"Error: Image not found: {img_path}")
        sys.exit(1)

    image = cv2.imread(str(img_path))
    if image is None:
        print(f"Error: Could not read image: {img_path}")
        sys.exit(1)

    print(f"Loading detector: {detector_name}")
    detector_cls = detector_registry.get(detector_name)
    detector = detector_cls()
    detector.initialize(model=model)
    print(f"Running detection on {image.shape[1]}x{image.shape[0]} image...")

    result = detector.process(image)
    print(f"Detected {len(result.boxes)} objects in {result.processing_time_ms:.1f}ms")

    for box in result.boxes:
        label = f"{box.class_name} ({box.confidence:.2f})" if box.confidence else box.class_name or "object"
        print(f"  [{label}] bbox=({box.x1:.0f},{box.y1:.0f},{box.x2:.0f},{box.y2:.0f})")

    annotated = annotate(image, result)
    cv2.imwrite(output, annotated)
    print(f"Annotated image saved to: {output}")

    detector.cleanup()


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("Usage: python run_detect.py <image> [--detector yolo|haar] [--model yolov8n.pt] [-o output.jpg]")
        sys.exit(1)

    image = args[0]
    detector = "yolo"
    model = "yolov8n.pt"
    output = "output.jpg"

    i = 1
    while i < len(args):
        if args[i] == "--detector" and i + 1 < len(args):
            detector = args[i + 1]; i += 2
        elif args[i] == "--model" and i + 1 < len(args):
            model = args[i + 1]; i += 2
        elif args[i] == "-o" and i + 1 < len(args):
            output = args[i + 1]; i += 2
        else:
            i += 1

    main(image, detector, model, output)
