"""Stage 1: Data cleaning, validation, splitting, and augmentation."""

import hashlib
import random
import time
from pathlib import Path

import cv2
import numpy as np
import structlog
import yaml
from PIL import Image

from vision_workbench.core.base import BaseStage
from vision_workbench.core.config import DataStageConfig
from vision_workbench.core.context import ContextKey, PipelineContext
from vision_workbench.core.result import StageResult
from vision_workbench.data.schema import DatasetManifest, DatasetSchema

logger = structlog.get_logger(__name__)


class DataStage(BaseStage[DataStageConfig, StageResult]):
    """Pipeline stage for dataset preparation.

    Validates images, removes duplicates, filters low-quality samples,
    splits into train/val/test, and optionally applies augmentation.
    """

    name = "data"
    description = "Dataset cleaning, validation, splitting, and augmentation"

    def validate_inputs(self, ctx: PipelineContext) -> tuple[bool, list[str]]:
        """DataStage is self-contained — source is in config, not context."""
        return True, []

    def run(self, config: DataStageConfig, ctx: PipelineContext) -> tuple[StageResult, PipelineContext]:
        t0 = time.perf_counter()
        logger.info("data.stage.start", source=config.source, target=config.target)

        src = Path(config.source)
        dst = Path(config.target)
        dst.mkdir(parents=True, exist_ok=True)

        # Discover
        images = self._discover_images(src, config.validation.get("allowed_formats", [".jpg", ".jpeg", ".png", ".bmp"]))
        logger.info("data.discovered", count=len(images))
        if not images:
            dt = time.perf_counter() - t0
            return StageResult(stage_name="data", success=False, error_message="No valid images found", duration_seconds=dt), ctx

        # Validate
        valid, issues = self._validate(images, config.validation)
        logger.info("data.validated", valid=len(valid), issues=len(issues))

        # Deduplicate
        if config.dedup.get("enabled"):
            valid, dup_removed = self._deduplicate(valid, config.dedup)

        # Quality filter
        valid, blur_removed = self._quality_filter(valid, config.validation)

        # Split
        splits = self._split(valid, config.split)

        # Populate standard layout
        DatasetSchema.create(dst)
        self._copy_images(dst, splits)

        # Manifest
        manifest = self._build_manifest(config, len(valid), splits)
        with open(dst / "dataset.yaml", "w", encoding="utf-8") as f:
            yaml.dump(manifest.model_dump(), f, default_flow_style=False, allow_unicode=True)

        dt = time.perf_counter() - t0
        result = StageResult(
            stage_name="data", success=True, duration_seconds=dt,
            artifacts={"dataset_dir": str(dst)},
            metrics={"total": len(valid), "train": len(splits.get("train", [])), "val": len(splits.get("val", [])), "test": len(splits.get("test", [])), "issues": len(issues)},
        )
        ctx = ctx.evolve(**{ContextKey.DATASET_DIR: str(dst)})
        ctx = ctx.record_stage("data", success=True, duration_seconds=dt)
        return result, ctx

    def dry_run(self, config: DataStageConfig, ctx: PipelineContext) -> dict:
        images = self._discover_images(Path(config.source), [".jpg", ".jpeg", ".png", ".bmp"])
        return {"action": "prepare_dataset", "source": config.source, "images_found": len(images), "split_ratio": config.split}

    # -- helpers --

    def _discover_images(self, root: Path, formats: list[str]) -> list[Path]:
        fmt = {f.lower() for f in formats}
        return sorted(p for p in root.rglob("*") if p.suffix.lower() in fmt and p.is_file())

    def _validate(self, images: list[Path], cfg: dict) -> tuple[list[Path], list[dict]]:
        min_w, min_h = cfg.get("min_resolution", [32, 32])
        max_w, max_h = cfg.get("max_resolution", [4096, 4096])
        valid, issues = [], []
        for p in images:
            try:
                with Image.open(p) as img:
                    w, h = img.size
                    if w < min_w or h < min_h:
                        issues.append({"path": str(p), "reason": "too_small"})
                    elif w > max_w or h > max_h:
                        issues.append({"path": str(p), "reason": "too_large"})
                    else:
                        img.verify()
                        valid.append(p)
            except Exception:
                issues.append({"path": str(p), "reason": "corrupt"})
        return valid, issues

    def _deduplicate(self, images: list[Path], cfg: dict) -> tuple[list[Path], int]:
        import imagehash
        hashes: dict[str, Path] = {}
        unique, dup = [], 0
        for p in images:
            try:
                h = str(imagehash.phash(Image.open(p)))
                if h not in hashes:
                    hashes[h] = p
                    unique.append(p)
                else:
                    dup += 1
            except Exception:
                unique.append(p)
        return unique, dup

    def _quality_filter(self, images: list[Path], cfg: dict) -> tuple[list[Path], int]:
        threshold = cfg.get("blur_threshold", 100)
        valid, removed = [], 0
        for p in images:
            img = cv2.imread(str(p))
            if img is None:
                removed += 1
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(gray, cv2.CV_64F).var() < threshold:
                removed += 1
            else:
                valid.append(p)
        return valid, removed

    def _split(self, images: list[Path], cfg: dict) -> dict[str, list[str]]:
        rng = random.Random(cfg.get("seed", 42))
        shuffled = list(images)
        rng.shuffle(shuffled)
        n = len(shuffled)
        tr, vr = cfg.get("train", 0.7), cfg.get("val", 0.15)
        te = int(n * tr)
        ve = te + int(n * vr)
        return {"train": [str(p) for p in shuffled[:te]], "val": [str(p) for p in shuffled[te:ve]], "test": [str(p) for p in shuffled[ve:]]}

    def _copy_images(self, dst: Path, splits: dict) -> None:
        for split_name, paths in splits.items():
            sdir = dst / "images" / split_name
            sdir.mkdir(parents=True, exist_ok=True)
            for sp in paths:
                src = Path(sp)
                target = sdir / src.name
                if not target.exists():
                    img = cv2.imread(str(src))
                    if img is not None:
                        cv2.imwrite(str(target), img)

    def _build_manifest(self, cfg: DataStageConfig, total: int, splits: dict) -> DatasetManifest:
        return DatasetManifest(name=cfg.dataset_name, version="1.0.0", description=f"Processed from {cfg.source}", image_count={k: len(v) for k, v in splits.items()})
