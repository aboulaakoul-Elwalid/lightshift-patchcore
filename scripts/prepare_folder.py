"""Prepare anomalib Folder layout for a MVTec AD 2 category, symlinks only.

data/fs/<category>/
  normal/        <- .../train/good
  normal_test/   <- .../test_public/good
  abnormal/      <- .../test_public/bad
  mask/          <- .../test_public/ground_truth/bad/<stem>_mask.png renamed to <stem>.png
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
AD2 = DATA / "mvtec-ad-2"


def prepare(category: str, force: bool = False) -> Path:
    src = AD2 / category
    dst = DATA / "fs" / category
    if dst.exists() and not force:
        print(f"{category}: already prepared at {dst}")
        return dst
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "normal").symlink_to(src / "train" / "good", target_is_directory=True)
    (dst / "normal_test").symlink_to(src / "test_public" / "good", target_is_directory=True)
    (dst / "abnormal").symlink_to(src / "test_public" / "bad", target_is_directory=True)
    mask_dst = dst / "mask"
    mask_dst.mkdir()
    for m in (src / "test_public" / "ground_truth" / "bad").glob("*_mask.png"):
        stem = m.name[len(category) + 1: -len("_mask.png")]  # strip category prefix if present
        # actual mask naming: <orig-image-stem>_mask.png where orig-stem may be 000_regular
        stem = m.name[: -len("_mask.png")]
        (mask_dst / f"{stem}.png").symlink_to(m)
    print(f"{category}: prepared at {dst}")
    return dst


if __name__ == "__main__":
    cats = sys.argv[1:] or ["can", "vial", "sheet_metal"]
    for c in cats:
        prepare(c)
