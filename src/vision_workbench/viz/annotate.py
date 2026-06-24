"""Core annotation drawing — boxes, keypoints, masks, and text on images."""

from typing import Optional

import cv2
import numpy as np

from vision_workbench.core.result import BoundingBox, DetectionResult
from vision_workbench.core.types import Keypoint

# Default palette — 20 distinct colors
DEFAULT_COLORS: list[tuple[int, int, int]] = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128),
    (128, 0, 255), (0, 128, 255), (192, 192, 192), (64, 64, 64),
]

SKELETON_CONNECTIONS: dict[str, list[tuple[int, int]]] = {
    "pose": [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)],
    "hand": [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)],
}


def annotate(
    image: np.ndarray,
    result: DetectionResult,
    color_map: Optional[dict[str, tuple[int, int, int]]] = None,
    line_thickness: int = 2,
    font_scale: float = 0.5,
    show_confidence: bool = True,
) -> np.ndarray:
    """Draw all detection annotations on an image copy.

    Args:
        image: BGR image (H, W, 3) uint8.
        result: DetectionResult with boxes, keypoints, masks, text.
        color_map: Optional per-class color mapping.
        line_thickness: Box/line thickness in pixels.
        font_scale: Text size scale factor.
        show_confidence: Whether to display confidence scores.

    Returns:
        Annotated image copy.
    """
    canvas = image.copy()
    color_map = color_map or {}

    # Draw bounding boxes
    for i, box in enumerate(result.boxes):
        color = _get_color(box.class_name, box.class_id, i, color_map)
        cv2.rectangle(
            canvas,
            (int(box.x1), int(box.y1)),
            (int(box.x2), int(box.y2)),
            color, line_thickness,
        )
        label_parts = []
        if box.class_name:
            label_parts.append(box.class_name)
        if show_confidence and box.confidence is not None:
            label_parts.append(f"{box.confidence:.2f}")
        if label_parts:
            label = " ".join(label_parts)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.rectangle(canvas, (int(box.x1), int(box.y1) - th - 4), (int(box.x1) + tw + 4, int(box.y1)), color, -1)
            cv2.putText(canvas, label, (int(box.x1) + 2, int(box.y1) - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

    # Draw keypoints
    for instance_kps in result.keypoints:
        for kp in instance_kps:
            if kp.visibility != "not_present":
                color = (0, 255, 0) if kp.visibility == "visible" else (0, 165, 255)
                cv2.circle(canvas, (int(kp.x), int(kp.y)), 3, color, -1)

    # Draw skeleton connections for pose/hand
    if result.task_type in SKELETON_CONNECTIONS and result.keypoints:
        connections = SKELETON_CONNECTIONS[result.task_type]
        for instance_kps in result.keypoints:
            for i, j in connections:
                if i < len(instance_kps) and j < len(instance_kps):
                    a, b = instance_kps[i], instance_kps[j]
                    if a.visibility != "not_present" and b.visibility != "not_present":
                        cv2.line(canvas, (int(a.x), int(a.y)), (int(b.x), int(b.y)), (0, 255, 255), 1)

    # Draw masks (transparent overlay)
    for mask in result.masks:
        if mask is not None and mask.shape[:2] == canvas.shape[:2]:
            overlay = canvas.copy()
            overlay[mask > 0] = (0, 255, 0)
            canvas = cv2.addWeighted(canvas, 0.7, overlay, 0.3, 0)

    return canvas


def draw_comparison_grid(
    images: dict[str, np.ndarray],
    titles: Optional[dict[str, str]] = None,
    columns: int = 2,
) -> np.ndarray:
    """Create a side-by-side comparison grid of annotated images.

    Args:
        images: Dict of detector_name → annotated image.
        titles: Optional dict of detector_name → display title.
        columns: Number of columns in the grid.

    Returns:
        Combined grid image.
    """
    titles = titles or {k: k for k in images}
    imgs = list(images.values())
    names = list(images.keys())
    rows = (len(imgs) + columns - 1) // columns
    h, w = imgs[0].shape[:2]
    grid = np.zeros((h * rows, w * columns, 3), dtype=np.uint8)
    for idx, (img, name) in enumerate(zip(imgs, names)):
        r, c = idx // columns, idx % columns
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = img
        cv2.putText(grid, titles[name], (c * w + 5, r * h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return grid


def _get_color(
    class_name: Optional[str],
    class_id: Optional[int],
    index: int,
    color_map: dict,
) -> tuple[int, int, int]:
    if class_name and class_name in color_map:
        return color_map[class_name]
    if class_id is not None and class_id in color_map:
        return color_map[class_id]
    if class_id is not None:
        return DEFAULT_COLORS[class_id % len(DEFAULT_COLORS)]
    return DEFAULT_COLORS[index % len(DEFAULT_COLORS)]
