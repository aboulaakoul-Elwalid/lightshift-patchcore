"""Export CPU-inference banks for the demo app (same split/protocol as scripts/qr4_ncurve.py).

Writes app/banks/<cat>.pt = {bank, threshold, ref_mean, ref_var, category, n_cal}
and ~12 held-out *normal* photos per category to app/static/calib/<cat>/NN.jpg (256 px)
so the demo can run "calibrate on k normals under the new lighting" without any upload.
Those photos come from the target pool of the split: never inside the memory bank.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
import qr4_ncurve as q  # noqa: E402
from lightshift import b2  # noqa: E402
from lightshift.b3 import conformal_threshold  # noqa: E402
from lightshift.patchcore import Extractor, build_bank, image_scores  # noqa: E402

DEST = REPO / "app" / "banks"
CALIB = REPO / "app" / "static" / "calib"
DEST.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    ext = Extractor()
    for cat in sys.argv[1:] or ["bottle", "hazelnut", "leather"]:
        bank_f, cal_f, pool_f, _, _ = q.split(cat)
        raw = q.feats(ext, bank_f)
        ref = b2.pool([b2.image_moments(x) for x in raw])
        bank = build_bank(list(raw), 0.25, 0).cpu()
        thr = conformal_threshold(image_scores(q.feats(ext, cal_f), bank.to(ext.device)), 0.1)
        torch.save({"bank": bank, "threshold": float(thr), "ref_mean": ref.mean.cpu(), "ref_var": ref.var.cpu(),
                    "category": cat, "n_cal": len(cal_f)}, DEST / f"{cat}.pt")
        (CALIB / cat).mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(pool_f[:12]):
            Image.open(p).convert("RGB").resize((256, 256)).save(CALIB / cat / f"{i:02d}.jpg", quality=88)
        print(f"{cat}: bank {tuple(bank.shape)} thr={float(thr):.4f} n_cal={len(cal_f)}", flush=True)
