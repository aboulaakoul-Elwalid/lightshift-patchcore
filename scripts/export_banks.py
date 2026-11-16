"""Export CPU-inference banks for the demo app from MVTec AD train normals.

Uses the same protocol as scripts/pilot_shift_eval.py (strided 25% coreset,
seeded 70/15/15 split) so app thresholds == pilot thresholds.
Writes app/banks/<cat>.pt: {bank: (P,C) float32, threshold: float, category: str}
"""
from __future__ import annotations

import json
import sys
from PIL import Image
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from pilot_shift_eval import build_bank, load_split, image_score, features, OUT  # noqa: E402

DEST = REPO / "app" / "banks"
DEST.mkdir(parents=True, exist_ok=True)

for cat in sys.argv[1:] or ["bottle", "hazelnut", "leather"]:
    bank_files, cal_files, tgt_files, _, _ = load_split(cat)
    bank = build_bank(bank_files).cpu()
    pjson = OUT / f"{cat}_pilot.json"
    if pjson.exists():
        thr = json.loads(pjson.read_text())["threshold_clean"]
    else:
        import numpy as np
        from lightshift.b3 import conformal_threshold

        sys.path.insert(0, str(REPO / "src"))
        scores = np.array([image_score(features(Image.open(p).convert("RGB")), bank) for p in cal_files])
        thr = conformal_threshold(scores, 0.1)
    torch.save({"bank": bank, "threshold": float(thr), "category": cat}, DEST / f"{cat}.pt")
    print(f"{cat}: bank {tuple(bank.shape)} thr={float(thr):.4f}")
