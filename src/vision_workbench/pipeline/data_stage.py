"""Stage 1: Data preparation with YOLO format support."""

import random
import shutil
import time
from pathlib import Path

import cv2
import structlog
import yaml
from PIL import Image

from vision_workbench.core.base import BaseStage
from vision_workbench.core.config import DataStageConfig
from vision_workbench.core.context import ContextKey, PipelineContext
from vision_workbench.core.result import StageResult

logger = structlog.get_logger(__name__)


class DataStage(BaseStage[DataStageConfig, StageResult]):
    """Prepare a YOLO-format dataset for training.

    Input:  Raw image directory with optional YOLO .txt labels.
    Output: Standard directory layout + Ultralytics dataset.yaml.
    """

    name = "data"
    description = "YOLO dataset preparation: discovery, validation, label parsing, split, dataset.yaml generation"

    def validate_inputs(self, ctx: PipelineContext) -> tuple[bool, list[str]]:
        return True, []

    def run(self, config: DataStageConfig, ctx: PipelineContext) -> tuple[StageResult, PipelineContext]:
        t0 = time.perf_counter()
        src = Path(config.source)
        dst = Path(config.target)
        dst.mkdir(parents=True, exist_ok=True)
        logger.info("data.start", source=str(src), target=str(dst))

        if not src.exists():
            dt = time.perf_counter() - t0
            return StageResult(stage_name="data", success=False, error_message=f"Source not found: {src}", duration_seconds=dt), ctx

        # 1. Discover images
        images = self._discover(src, config.validation.get("allowed_formats", [".jpg", ".jpeg", ".png", ".bmp"]))
        logger.info("data.discovered", count=len(images))
        if not images:
            dt = time.perf_counter() - t0
            return StageResult(stage_name="data", success=False, error_message="No images found", duration_seconds=dt), ctx

        # 2. Validate images
        valid, issues = self._validate(images, config.validation)
        logger.info("data.validated", valid=len(valid), issues=len(issues))

        # 3. Parse YOLO labels
        has_labels = self._check_labels(valid)
        if has_labels:
            logger.info("data.labels_found", format="yolo")
        else:
            logger.warning("data.no_labels", hint="Only images found, no .txt labels. This is fine for inference-only datasets.")

        # 4. Split dataset
        splits = self._split(valid, config.split)

        # 5. Populate standard layout
        self._populate(dst, valid, splits, has_labels)

        # 6. Generate dataset.yaml
        classes = config.classes or self._infer_classes(valid)
        yaml_path = dst / "dataset.yaml"
        self._write_yaml(dst, yaml_path, classes)

        # 7. Generate label statistics
        stats = self._compute_stats(valid)

        dt = time.perf_counter() - t0
        result = StageResult(
            stage_name="data", success=True, duration_seconds=dt,
            artifacts={"dataset_dir": str(dst), "dataset_yaml": str(yaml_path)},
            metrics={"total_images": len(valid), "train": len(splits["train"]), "val": len(splits["val"]), "test": len(splits.get("test", [])), "issues": len(issues), "has_labels": has_labels, **stats},
        )
        ctx = ctx.evolve(**{ContextKey.DATASET_DIR: str(dst)})
        ctx = ctx.record_stage("data", success=True, duration_seconds=dt)
        return result, ctx

    def dry_run(self, config: DataStageConfig, ctx: PipelineContext) -> dict:
        images = self._discover(Path(config.source), [".jpg", ".jpeg", ".png", ".bmp"])
        return {"action": "prepare_yolo_dataset", "source": config.source, "images_found": len(images)}

    # -- helpers --

    def _discover(self, root: Path, formats: list[str]) -> list[Path]:
        fmt = {f.lower() for f in formats}
        return sorted(p for p in root.rglob("*") if p.suffix.lower() in fmt and p.is_file())

    def _validate(self, images: list[Path], cfg: dict) -> tuple[list[Path], list[dict]]:
        min_w, min_h = cfg.get("min_resolution", [1, 1])
        max_w, max_h = cfg.get("max_resolution", [8192, 8192])
        valid, issues = [], []
        for p in images:
            try:
                with Image.open(p) as img:
                    w, h = img.size
                    if w < min_w or h < min_h:
                        issues.append({"path": str(p), "reason": "too_small", "size": [w, h]})
                    elif w > max_w or h > max_h:
                        issues.append({"path": str(p), "reason": "too_large", "size": [w, h]})
                    else:
                        img.verify()
                        valid.append(p)
            except Exception:
                issues.append({"path": str(p), "reason": "corrupt"})
        return valid, issues

    def _check_labels(self, images: list[Path]) -> bool:
        """Check if YOLO .txt labels exist alongside images."""
        for img in images:
            label = img.with_suffix(".txt")
            if label.exists():
                return True
        return False

    def _split(self, images: list[Path], cfg: dict) -> dict[str, list[Path]]:
        rng = random.Random(cfg.get("seed", 42))
        shuffled = list(images)
        rng.shuffle(shuffled)
        n = len(shuffled)
        tr = cfg.get("train", 0.7)
        vr = cfg.get("val", 0.2)
        te = int(n * tr)
        ve = te + int(n * vr)
        return {"train": shuffled[:te], "val": shuffled[te:ve], "test": shuffled[ve:]}

    def _populate(self, dst: Path, images: list[Path], splits: dict, has_labels: bool) -> None:
        for split_name, imgs in splits.items():
            img_dir = dst / "images" / split_name
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir = dst / "labels" / split_name
            lbl_dir.mkdir(parents=True, exist_ok=True)
            for src_path in imgs:
                dst_img = img_dir / src_path.name
                if not dst_img.exists():
                    shutil.copy2(src_path, dst_img)
                if has_labels:
                    src_lbl = src_path.with_suffix(".txt")
                    if src_lbl.exists():
                        shutil.copy2(src_lbl, lbl_dir / src_lbl.name)

    def _write_yaml(self, dst: Path, yaml_path: Path, classes: list[str]) -> None:
        data = {
            "path": str(dst.absolute()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test" if (dst / "images" / "test").exists() else "",
            "nc": len(classes),
            "names": classes,
        }
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    def _infer_classes(self, images: list[Path]) -> list[str]:
        """Try to infer class names from labels."""
        class_ids = set()
        for img in images:
            lbl = img.with_suffix(".txt")
            if lbl.exists():
                for line in open(lbl):
                    cid = line.strip().split()[0]
                    class_ids.add(int(cid))
        return [f"class_{i}" for i in sorted(class_ids)] if class_ids else ["object"]

    def _compute_stats(self, images: list[Path]) -> dict:
        total_boxes = 0
        img_with_boxes = 0
        for img in images:
            lbl = img.with_suffix(".txt")
            if lbl.exists():
                boxes = len(lbl.read_text().strip().splitlines())
                if boxes > 0:
                    img_with_boxes += 1
                    total_boxes += boxes
        return {"total_annotations": total_boxes, "images_with_annotations": img_with_boxes, "images_without_annotations": len(images) - img_with_boxes}
