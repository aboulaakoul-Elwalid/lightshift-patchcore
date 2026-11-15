"""Prepare anomalib Folder layout for a MVTec AD (original) category, symlinks only.

data/fs_ad/<category>/
  normal/      <- .../train/good
  normal_test/ <- .../test/good
  abnormal/    <- .../test/<defect dirs> merged via an anomalib-compatible tree
For MVTec AD, abnormal test images live in defect-named subdirs; anomalib Folder
expects a flat abnormal dir, so we materialize a real dir with symlinks to files.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
MVT = Path("/mnt/data-ssd/mvtec_ad_full")


def prepare(category: str, force: bool = False) -> Path:
    src = MVT / category
    dst = DATA / "fs_ad" / category
    if dst.exists() and not force:
        print(f"{category}: already prepared at {dst}")
        return dst
    dst.mkdir(parents=True, exist_ok=True)
    if not (dst / "normal").is_symlink():
        (dst / "normal").symlink_to(src / "train" / "good", target_is_directory=True)
        (dst / "normal_test").symlink_to(src / "test" / "good", target_is_directory=True)
    abnormal = dst / "abnormal"
    abnormal.mkdir(exist_ok=True)
    for defect in sorted(p for p in (src / "test").iterdir() if p.is_dir() and p.name != "good"):
        for img in sorted(defect.glob("*.png")):
            link = abnormal / img.name
            if not link.exists():
                # prefix defect type to keep names unique
                link.symlink_to(img)
    mask_dst = dst / "mask"
    mask_dst.mkdir(exist_ok=True)
    for gt in sorted((src / "ground_truth").iterdir()):
        for m in sorted(gt.glob("*.png")):
            link = mask_dst / m.name
            if not link.exists():
                link.symlink_to(m)
    n_ab = len(list(abnormal.glob("*.png")))
    n_mask = len(list(mask_dst.glob("*.png")))
    print(f"{category}: prepared at {dst} (abnormal={n_ab}, masks={n_mask})")
    return dst


if __name__ == "__main__":
    cats = sys.argv[1:] or ["bottle", "hazelnut", "leather"]
    for c in cats:
        prepare(c)
