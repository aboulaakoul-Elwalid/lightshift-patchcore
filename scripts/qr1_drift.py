"""QR1: does per-channel feature-statistic drift predict the accuracy drop?

For each category and each of 5 families x 5 severities:
  drift_l2 / drift_l3 / drift_all = mean over channels of |mu_shift - mu_ref| / sd_ref
      (mu_shift: pooled over the whole target pool under the shift; mu_ref, sd_ref: bank images)
  auroc_drop = AUROC(clean) - AUROC(shifted)            (vanilla PatchCore)
  fpr_src    = false-alarm rate at the clean conformal threshold
Spearman rho is computed per category over the 25 conditions, and per family over 5 severities.
Outputs results/qr1/qr1_rows.csv and results/qr1/qr1_summary.md
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qr4_ncurve as q  # noqa: E402
from lightshift import b2  # noqa: E402
from lightshift.b3 import conformal_threshold  # noqa: E402
from lightshift.patchcore import Extractor, build_bank, image_scores  # noqa: E402
from lightshift.shift import SHIFT_FAMILIES  # noqa: E402

OUT = q.REPO / "results" / "qr1"
OUT.mkdir(exist_ok=True)


def run(cat):
    ext = Extractor()
    bank_f, cal_f, pool_f, good_f, bad_f = q.split(cat)
    raw = q.feats(ext, bank_f)
    ref = b2.pool([b2.image_moments(x) for x in raw])
    bank = build_bank(list(raw), 0.25, 0)
    del raw
    thr = conformal_threshold(image_scores(q.feats(ext, cal_f), bank), 0.1)
    sd = torch.sqrt(ref.var + 1e-6)
    y = np.r_[np.zeros(len(good_f)), np.ones(len(bad_f))]
    rows = []
    for fam, sev in [("clean", 0)] + [(f, s) for f in SHIFT_FAMILIES for s in range(1, 6)]:
        sh = None if fam == "clean" else fam
        tg, tb = q.feats(ext, good_f, sh, sev), q.feats(ext, bad_f, sh, sev)
        pl = b2.pool([b2.image_moments(x) for x in q.feats(ext, pool_f, sh, sev)])
        d = (pl.mean - ref.mean).abs() / sd
        sg, sb = image_scores(tg, bank), image_scores(tb, bank)
        rows.append({"cat": cat, "family": fam, "sev": sev, "drift_l2": float(d[:512].mean()), "drift_l3": float(d[512:].mean()),
                     "drift_all": float(d.mean()), "auroc": float(roc_auc_score(y, np.r_[sg, sb])),
                     "fpr_src": float(np.mean(sg > thr)), "mean_good": float(sg.mean())})
        print(f"[{cat}] {fam}-{sev} drift={rows[-1]['drift_all']:.3f} auroc={rows[-1]['auroc']:.3f} fpr={rows[-1]['fpr_src']:.2f}", flush=True)
    c0 = rows[0]["auroc"]
    for r in rows:
        r["auroc_drop"] = c0 - r["auroc"]
    return rows


if __name__ == "__main__":
    allr = []
    for c in sys.argv[1:] or ["bottle", "hazelnut", "leather"]:
        allr += run(c)
    with (OUT / "qr1_rows.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allr[0]))
        w.writeheader()
        w.writerows(allr)
    L = ["| scope | drift measure | target | Spearman rho | n |", "|---|---|---|---|---|"]
    shifted = [r for r in allr if r["family"] != "clean"]
    for cat in sorted({r["cat"] for r in shifted}):
        rs = [r for r in shifted if r["cat"] == cat]
        for m in ("drift_l2", "drift_l3", "drift_all"):
            for t in ("auroc_drop", "fpr_src", "mean_good"):
                rho = spearmanr([r[m] for r in rs], [r[t] for r in rs]).statistic
                L.append(f"| {cat} (25 conditions) | {m} | {t} | {rho:+.2f} | {len(rs)} |")
    for cat in sorted({r["cat"] for r in shifted}):
        for fam in SHIFT_FAMILIES:
            rs = [r for r in shifted if r["cat"] == cat and r["family"] == fam]
            rho = spearmanr([r["drift_all"] for r in rs], [r["auroc_drop"] for r in rs]).statistic
            L.append(f"| {cat}/{fam} (5 severities) | drift_all | auroc_drop | {rho:+.2f} | 5 |")
    (OUT / "qr1_summary.md").write_text("\n".join(L))
    print("\n".join(L))
