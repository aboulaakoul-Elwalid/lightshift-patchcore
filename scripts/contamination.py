"""Robustness of B2 and B3 to contaminated calibration photos.

Registered negative-control spirit (QR4): a repair must not erase real defects. Here: of the k = 10
"normal" photos used for calibration, m are in fact DEFECTIVE (operator error), m in {0, 1, 2, 3}.
Contaminants come from one half of the defective test images; evaluation (TPR) uses the other half,
so no image is both calibration and test.
Conditions: brightness-4, shading-3, color_temp-4. Variants: none (B3 only) and b2a (B2a + B3).
Outputs results/contamination/rows.csv
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qr4_ncurve as q  # noqa: E402
from lightshift import b2  # noqa: E402
from lightshift.b3 import conformal_threshold  # noqa: E402
from lightshift.patchcore import Extractor, build_bank, image_scores  # noqa: E402

OUT = q.REPO / "results" / "contamination"
OUT.mkdir(exist_ok=True)
CONDS = [("brightness", 4), ("shading", 3), ("color_temp", 4)]
import os
K, DRAWS = int(os.environ.get("CONTAM_K", 10)), 20
TAG = "" if K == 10 else f"_k{K}"


def run(cat):
    ext = Extractor()
    bank_f, cal_f, pool_f, good_f, bad_f = q.split(cat)
    raw = q.feats(ext, bank_f)
    ref = b2.pool([b2.image_moments(x) for x in raw])
    bank = build_bank(list(raw), 0.25, 0)
    del raw
    thr_src = conformal_threshold(image_scores(q.feats(ext, cal_f), bank), 0.1)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(bad_f))
    half = len(bad_f) // 2
    contam_idx, eval_idx = perm[:half], perm[half:]
    rows = []
    for fam, sev in CONDS:
        tg = q.feats(ext, good_f, fam, sev)
        tb = q.feats(ext, bad_f, fam, sev)
        pl = q.feats(ext, pool_f, fam, sev)
        pool_all = list(pl) + list(tb[contam_idx])  # indices >= len(pool_f) are defective
        mom = [b2.image_moments(x) for x in pool_all]
        sg0, sb0 = image_scores(tg, bank), image_scores(tb[eval_idx], bank)
        for m in ((0, 1, 2, 3) if K == 10 else (0, 2, 4, 5, 8)):
            for d in range(DRAWS):
                r = np.random.default_rng(100 * m + d)
                idx = list(r.permutation(len(pool_f))[: K - m]) + [len(pool_f) + int(i) for i in r.choice(len(contam_idx), m, replace=False)]
                cal_raw = [pool_all[i] for i in idx]
                s_none = image_scores(__import__("torch").stack(cal_raw), bank)
                tm = b2.pool([mom[i] for i in idx])
                for variant in ("none", "b2a"):
                    if variant == "none":
                        thr, sg, sb = conformal_threshold(s_none, 0.1), sg0, sb0
                    else:
                        al = lambda x: b2.align(x, "mean", tm, ref)  # noqa: E731
                        thr = conformal_threshold(image_scores(al(__import__("torch").stack(cal_raw)), bank), 0.1)
                        sg, sb = image_scores(al(tg), bank), image_scores(al(tb[eval_idx]), bank)
                    rows.append({"cat": cat, "shift": f"{fam}-s{sev}", "m": m, "draw": d, "variant": variant,
                                 "fpr": float(np.mean(sg > thr)), "tpr": float(np.mean(sb > thr)), "thr": float(thr)})
        print(f"[{cat}] {fam}-{sev} done", flush=True)
    return rows


if __name__ == "__main__":
    allr = []
    for c in sys.argv[1:] or ["bottle", "hazelnut", "leather"]:
        allr += run(c)
    with (OUT / f"rows{TAG}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allr[0]))
        w.writeheader()
        w.writerows(allr)
    print("done")
