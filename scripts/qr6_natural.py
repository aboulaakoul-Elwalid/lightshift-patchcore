"""QR6 on REAL lighting shift: MVTec AD 2 test_public.

AD 2 captures every test scene under several lighting conditions, encoded in the file
name (regular / overexposed / underexposed / shift_1..4). Train and validation are all
`regular`. So the bank sees one lighting and the test set shows others - a natural
version of our synthetic study, with no shift generator involved.

Protocol per category and per lighting condition L:
  bank          = train/good (regular); reference moments from the same images
  source thr    = conformal(alpha) of validation/good scores
  target normals= k random images from test_public/good of condition L (the operator's k photos)
  evaluated on  = the REMAINING goods of L (>= 1) + all bad of L   (calibration images never scored)
  variants      none (vanilla; B3-only when thr=target) | b2a | b2b
  averaged over all (or 40 random) k-subsets, because n_L is tiny (4-12 goods).
Outputs results/qr6/qr6_natural_rows.csv
"""
import csv
import itertools
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qr4_ncurve as q  # noqa: E402
from lightshift import b2  # noqa: E402
from lightshift.b3 import conformal_threshold  # noqa: E402
from lightshift.patchcore import Extractor, build_bank, image_scores  # noqa: E402

AD2 = Path("/mnt/data-ssd/mvtec-ad-2")
OUT = q.REPO / "results" / "qr6"
OUT.mkdir(exist_ok=True)


def cond(p: Path) -> str:
    return re.sub(r"^\d+_", "", p.stem)


def pngs(d):
    return sorted(Path(d).glob("*.png"))


def mean_dict(vals):
    out = {}
    for m in vals[0]:
        xs = [v[m] for v in vals if v[m] is not None]
        out[m] = float(np.mean(xs)) if xs else None
    return out


def run(cat):
    ext = Extractor()
    r = AD2 / cat
    tr, va = pngs(r / "train/good"), pngs(r / "validation/good")
    good, bad = pngs(r / "test_public/good"), pngs(r / "test_public/bad")
    raw = q.feats(ext, tr)
    ref = b2.pool([b2.image_moments(x) for x in raw])
    bank = build_bank(list(raw), 0.25, 0)
    del raw
    thr_src = conformal_threshold(image_scores(q.feats(ext, va), bank), 0.1)
    print(f"[{cat}] bank {tuple(bank.shape)} thr_src={thr_src:.4f}", flush=True)
    rows = []

    def met(sg, sb, thr_t):
        y = np.r_[np.zeros(len(sg)), np.ones(len(sb))]
        d = {"auroc": float(roc_auc_score(y, np.r_[sg, sb]))}
        for nm, t in (("src", thr_src), ("tgt", thr_t)):
            d[f"fpr_{nm}"] = None if t is None else float(np.mean(sg > t))
            d[f"tpr_{nm}"] = None if t is None else float(np.mean(sb > t))
        return d

    for L in sorted({cond(p) for p in good}):
        g = [p for p in good if cond(p) == L]
        b = [p for p in bad if cond(p) == L]
        fg, fb = q.feats(ext, g), q.feats(ext, b)
        mom = [b2.image_moments(x) for x in fg]
        sg0, sb0 = image_scores(fg, bank), image_scores(fb, bank)
        n = len(g)

        def add(variant, k, vals):
            rows.append({"cat": cat, "cond": L, "variant": variant, "k": k, "n_good": n, "n_bad": len(b), **mean_dict(vals)})

        add("none", 0, [met(sg0, sb0, None)])
        for k in (2, 3, 4, 5, 8, 11):
            if k > n - 1:  # keep at least one good photo out for evaluation
                continue
            combos = list(itertools.combinations(range(n), k))
            if len(combos) > 40:
                rng = np.random.default_rng(k)
                combos = [combos[i] for i in rng.choice(len(combos), 40, replace=False)]
            acc = {"none": [], "b2a": [], "b2b": []}
            for cmb in combos:
                cmb = list(cmb)
                rest = [i for i in range(n) if i not in cmb]
                acc["none"].append(met(sg0[rest], sb0, conformal_threshold(sg0[cmb], 0.1)))
                tm = b2.pool([mom[i] for i in cmb])
                for v, mode in (("b2a", "mean"), ("b2b", "meanstd")):
                    sg = image_scores(b2.align(fg, mode, tm, ref), bank)
                    sb = image_scores(b2.align(fb, mode, tm, ref), bank)
                    acc[v].append(met(sg[rest], sb, conformal_threshold(sg[cmb], 0.1)))
            for v, vals in acc.items():
                add(v, k, vals)
        y = np.r_[np.zeros(n), np.ones(len(b))]
        print(f"[{cat}] {L}: n_good={n} n_bad={len(b)} vanilla AUROC={roc_auc_score(y, np.r_[sg0, sb0]):.3f} "
              f"FPR@src={np.mean(sg0 > thr_src):.2f}", flush=True)
    return rows


if __name__ == "__main__":
    allr = []
    for c in sys.argv[1:] or ["vial", "sheet_metal", "can"]:
        allr += run(c)
        with (OUT / "qr6_natural_rows.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(allr[0]))
            w.writeheader()
            w.writerows(allr)
    print("done")
