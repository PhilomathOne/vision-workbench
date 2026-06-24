"""Standardized dataset directory layout and description schema."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class CategoryInfo(BaseModel):
    """A single category/label definition."""
    id: int
    name: str
    supercategory: str = ""


class ImageStats(BaseModel):
    """Aggregate statistics for a dataset's images."""
    avg_width: int = 0
    avg_height: int = 0
    total_size_gb: float = 0.0


class AnnotationStats(BaseModel):
    """Aggregate statistics for annotations."""
    total_boxes: int = 0
    boxes_per_image_avg: float = 0.0
    class_distribution: dict[str, int] = Field(default_factory=dict)


class ProvenanceInfo(BaseModel):
    """Where the data came from and what was done to it."""
    source: str = ""
    collection_location: str = ""
    collection_period: str = ""
    preprocessing: list[str] = Field(default_factory=list)


class DatasetManifest(BaseModel):
    """Contents of a dataset.yaml file — the identity card for a dataset."""

    name: str
    version: str = "1.0.0"
    created: str = ""
    description: str = ""
    license: str = ""

    tasks: list[str] = Field(default_factory=list)
    categories: list[CategoryInfo] = Field(default_factory=list)

    image_count: dict[str, int] = Field(default_factory=dict)  # {train: N, val: N, test: N}
    image_stats: ImageStats = Field(default_factory=ImageStats)
    annotation_stats: AnnotationStats = Field(default_factory=AnnotationStats)
    provenance: ProvenanceInfo = Field(default_factory=ProvenanceInfo)
    checksums: dict[str, str] = Field(default_factory=dict)


class DatasetSchema:
    """Standard directory structure for datasets.

    Expected layout::

        <dataset_root>/
        ├── dataset.yaml          # DatasetManifest (required)
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        ├── annotations/
        │   ├── instances_train.json
        │   ├── instances_val.json
        │   └── instances_test.json
        └── splits/
            └── split_*.json
    """

    IMAGES_DIR = "images"
    ANNOTATIONS_DIR = "annotations"
    SPLITS_DIR = "splits"
    MANIFEST_FILE = "dataset.yaml"

    SUBDIRS = {"train", "val", "test"}

    @classmethod
    def validate(cls, root: Path) -> tuple[bool, list[str]]:
        """Check that a directory conforms to the schema."""
        issues: list[str] = []
        manifest = root / cls.MANIFEST_FILE
        if not manifest.exists():
            issues.append(f"Missing {cls.MANIFEST_FILE}")
        images = root / cls.IMAGES_DIR
        if not images.exists():
            issues.append(f"Missing {cls.IMAGES_DIR}/ directory")
        return len(issues) == 0, issues

    @classmethod
    def create(cls, root: Path) -> None:
        """Create the standard directory structure."""
        root.mkdir(parents=True, exist_ok=True)
        (root / cls.IMAGES_DIR).mkdir(exist_ok=True)
        (root / cls.ANNOTATIONS_DIR).mkdir(exist_ok=True)
        (root / cls.SPLITS_DIR).mkdir(exist_ok=True)
        for sub in cls.SUBDIRS:
            (root / cls.IMAGES_DIR / sub).mkdir(exist_ok=True)
