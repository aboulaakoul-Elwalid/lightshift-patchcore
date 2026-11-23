"""QR4 (+QR3 partial): B2 alignment vs number k of target-lighting normals.

Deployable protocol (no clean twins of target images):
  split of train/good per category (seeded): bank 60% / source calibration 10% / target pool 30%.
  reference moments  = per-channel moments of the bank images (reference lighting)
  target moments     = pooled moments of k random target-pool images *under the shift*
  variants:
    none      vanilla PatchCore (B3-only when thr=target)
    b2a       mean alignment
    b2b       mean+std alignment (corrected; pilot B2b had a bug)
    oracle_a  pilot-style paired mean gap (needs clean twins) - upper bound, NOT deployable
  thresholds:
    src  = conformal(alpha) of clean calibration scores   (what vanilla ships with)
    tgt  = conformal(alpha) of the k aligned target normals scored with the same aligned model
  metrics on shifted test set: image AUROC, FPR (good > thr), TPR (bad > thr)
Shifts: 5 families x severities {2,3,4} + a "clean" control (alignment must not hurt).
k in {0,5,10,25,50}; 10 random draws per k>0 (draw seeds fixed).

Outputs: results/qr4/<cat>.json, results/qr4/qr4_rows.csv
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from lightshift import b2  # noqa: E402
from lightshift.b3 import conformal_threshold  # noqa: E402
from lightshift.patchcore import Extractor, build_bank, image_scores  # noqa: E402
from lightshift.shift import SHIFT_FAMILIES, shifted_pil  # noqa: E402

MVT = Path("/mnt/data-ssd/mvtec_ad_full")
OUT = REPO / "results" / "qr4"
OUT.mkdir(parents=True, exist_ok=True)
ALPHA, SEED = 0.1, 0
import os
KS = (0, 5, 10, 25, 50)
DRAWS = int(os.environ.get("QR4_DRAWS", 10))
SEVS = (2, 3, 4)


def split(cat: str):
    files = sorted((MVT / cat / "train" / "good").glob("*.png"))
    idx = np.random.default_rng(SEED).permutation(len(files))
    b, c = int(0.6 * len(files)), int(0.7 * len(files))
    pick = lambda ii: [files[i] for i in ii]  # noqa: E731
    good = sorted((MVT / cat / "test" / "good").glob("*.png"))
    bad = sorted(p for p in (MVT / cat / "test").glob("*/*.png") if p.parent.name != "good")
    return pick(idx[:b]), pick(idx[b:c]), pick(idx[c:]), good, bad


def feats(ext, paths, fam=None, sev=None):
    out = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        if fam is not None:
            img = shifted_pil(img, fam, sev, rng_seed=SEED)
        out.append(ext.raw(img))
    return torch.stack(out)  # (N, P, C)


def metrics(s_good, s_bad, thr_src, thr_tgt):
    y = np.r_[np.zeros(len(s_good)), np.ones(len(s_bad))]
    d = {"auroc": float(roc_auc_score(y, np.r_[s_good, s_bad]))}
    for name, t in (("src", thr_src), ("tgt", thr_tgt)):
        if t is None:
            d[f"fpr_{name}"] = d[f"tpr_{name}"] = None
        else:
            d[f"fpr_{name}"] = float(np.mean(s_good > t))
            d[f"tpr_{name}"] = float(np.mean(s_bad > t))
    return d


def run_cat(cat: str, splitter=None, ks=None, tag: str = "", variants=("b2a", "b2b", "oracle_a"),
            aug_families=None, sevs=None):
    """aug_families: None = ordinary bank (B0); else bank built from randomly shifted
    images drawn from these families (B1), at the SAME bank size (equal memory)."""
    t0 = time.time()
    ks = tuple(ks or KS)
    sevs = tuple(sevs or SEVS)
    ext = Extractor()
    bank_f, cal_f, pool_f, good_f, bad_f = (splitter or split)(cat)
    print(f"[{cat}] bank {len(bank_f)} cal {len(cal_f)} pool {len(pool_f)} test {len(good_f)}/{len(bad_f)}", flush=True)

    bank_raw = feats(ext, bank_f)
    ref = b2.pool([b2.image_moments(x) for x in bank_raw])
    if aug_families:
        del bank_raw
        rng = np.random.default_rng(SEED + 7)
        aug = []
        for p in bank_f:
            img = Image.open(p).convert("RGB")
            if rng.random() > 1.0 / (len(aug_families) + 1):  # keep ~1/(F+1) of images clean
                img = shifted_pil(img, aug_families[rng.integers(len(aug_families))], int(rng.integers(1, 5)), rng_seed=SEED)
            aug.append(ext.raw(img))
        bank_raw = torch.stack(aug)
    bank = build_bank(list(bank_raw), 0.25, SEED)
    del bank_raw
    thr_src = conformal_threshold(image_scores(feats(ext, cal_f), bank), ALPHA)
    print(f"[{cat}] bank {tuple(bank.shape)} thr_src={thr_src:.4f}", flush=True)

    pool_clean = feats(ext, pool_f)
    clean_mom = [b2.image_moments(x) for x in pool_clean]
    del pool_clean

    conds = [("clean", 0)] + [(f, s) for f in SHIFT_FAMILIES for s in sevs]
    rows = []
    for fam, sev in conds:
        key = fam if fam == "clean" else f"{fam}-s{sev}"
        sh = None if fam == "clean" else fam
        tg, tb = feats(ext, good_f, sh, sev), feats(ext, bad_f, sh, sev)
        pl = feats(ext, pool_f, sh, sev)
        pl_mom = [b2.image_moments(x) for x in pl]
        pl_none = image_scores(pl, bank)

        def emit(variant, k, draw, sg, sb, thr_t):
            rows.append({"cat": cat, "shift": key, "variant": variant, "k": k, "draw": draw,
                         **metrics(sg, sb, thr_src, thr_t)})

        sg0, sb0 = image_scores(tg, bank), image_scores(tb, bank)
        emit("none", 0, 0, sg0, sb0, None)
        for k in ks:
            if k <= 0 or k > len(pool_f):
                continue
            for d in range(DRAWS):
                idx = np.random.default_rng(1000 * k + d).permutation(len(pool_f))[:k]
                thr_none = conformal_threshold(pl_none[idx], ALPHA)
                emit("none", k, d, sg0, sb0, thr_none)
                tm = b2.pool([pl_mom[i] for i in idx])
                om = b2.pool([clean_mom[i] for i in idx])
                for variant, mode, tmom, rmom in [v for v in (
                    ("b2a", "mean", tm, ref),
                    ("b2b", "meanstd", tm, ref),
                    ("oracle_a", "mean", tm, om),
                ) if v[0] in variants]:
                    sg = image_scores(b2.align(tg, mode, tmom, rmom), bank)
                    sb = image_scores(b2.align(tb, mode, tmom, rmom), bank)
                    sp = image_scores(b2.align(pl[idx], mode, tmom, rmom), bank)
                    emit(variant, k, d, sg, sb, conformal_threshold(sp, ALPHA))
        print(f"[{cat}] {key} done ({time.time() - t0:.0f}s)", flush=True)
        del tg, tb, pl
    (OUT / f"{tag}{cat}.json").write_text(json.dumps({"thr_src": thr_src, "rows": rows}))
    return rows


if __name__ == "__main__":
    cats = sys.argv[1:] or ["bottle", "hazelnut", "leather"]
    allrows = []
    for c in cats:
        allrows += run_cat(c)
    with (OUT / "qr4_rows.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0]))
        w.writeheader()
        w.writerows(allrows)
    print("done")
