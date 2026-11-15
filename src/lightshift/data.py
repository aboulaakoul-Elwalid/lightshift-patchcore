"""MVTec AD 2 data loading (local layout verified 2026-11-15)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

CATEGORIES = ("can", "vial", "sheet_metal")


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str  # "good" | "bad" | other split tag
    defect_class: str | None
    gt_path: Path | None = None


def category_root(root: Path, category: str) -> Path:
    if category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r}")
    p = Path(root) / category
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; check the data mount")
    return p


def scan_split(root: Path, category: str, split: str) -> list[Sample]:
    base = category_root(root, category) / split
    samples: list[Sample] = []
    for cls_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        label = "good" if cls_dir.name == "good" else "bad"
        gt_dir = base / "ground_truth" / cls_dir.name
        for img in sorted(cls_dir.glob("*.png")):
            gt: Path | None = None
            if label == "bad" and gt_dir.exists():
                stem = img.stem.split("_")[0] + "_mask.png"
                cand = gt_dir / stem
                gt = cand if cand.exists() else next(gt_dir.glob(img.stem + "*"), None)
            samples.append(Sample(img, label, cls_dir.name, gt))
    return samples


def load_image(sample: Sample):
    from PIL import Image

    return Image.open(sample.path).convert("RGB")


def load_mask(sample: Sample) -> np.ndarray | None:
    if sample.gt_path is None or not sample.gt_path.exists():
        return None
    from PIL import Image

    return np.asarray(Image.open(sample.gt_path), dtype=np.uint8)
