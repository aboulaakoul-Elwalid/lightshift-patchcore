"""Aggregate results/qr4/qr4_rows.csv into QR4/QR3 summary tables (markdown)."""
import sys as _s; from pathlib import Path as _P; _s.path.insert(0, str(_P(__file__).resolve().parents[1] / "src"))
import sys
from pathlib import Path

import numpy as np
import pandas as pd

def md(rows):
    cols = list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out)


OUT = Path(__file__).resolve().parents[1] / "results" / "qr4"
d = pd.read_csv(OUT / "qr4_rows.csv")
m = d.groupby(["cat", "shift", "variant", "k"], as_index=False).mean(numeric_only=True).drop(columns="draw")
sh = m[m["shift"] != "clean"]
lines = []

# --- QR4: AUROC recovery. Only conditions where shift costs vanilla >= 0.01 AUROC vs clean.
clean = m[(m["shift"] == "clean") & (m.variant == "none") & (m.k == 0)].set_index("cat")["auroc"]
van = sh[(sh.variant == "none") & (sh.k == 0)].set_index(["cat", "shift"])["auroc"]
drop = (van.rename("v").reset_index().assign(c=lambda x: x["cat"].map(clean)).assign(drop=lambda x: x.c - x.v))
hurt = drop[drop["drop"] >= 0.01][["cat", "shift", "drop", "v", "c"]]
lines += [f"## QR4 - AUROC recovery (conditions where shift costs vanilla >= 0.01 AUROC; n={len(hurt)} of {len(drop)})", ""]
rows = []
for var in ("b2a", "b2b", "oracle_a"):
    r = {"variant": var}
    for k in (5, 10, 25, 50):
        x = sh[(sh.variant == var) & (sh.k == k)].merge(hurt, on=["cat", "shift"])
        rec = ((x.auroc - x.v) / x["drop"]).clip(-5, 5)
        r[f"k={k}"] = f"{rec.mean():.2f}"
    rows.append(r)
lines += [md(rows), "",
          "Cell = mean over (category, shift) of (AUROC_variant - AUROC_vanilla) / (AUROC_clean - AUROC_vanilla); 1.0 = fully recovered.", ""]

# per-condition 10-vs-50 ratio, literal QR4 wording
for var in ("b2a", "b2b"):
    g = {k: sh[(sh.variant == var) & (sh.k == k)].merge(hurt, on=["cat", "shift"]).set_index(["cat", "shift"]) for k in (10, 50)}
    g10, g50 = g[10].auroc - g[10].v, g[50].auroc - g[50].v
    ok = g50 > 0.005
    ratio = (g10[ok] / g50[ok])
    lines.append(f"- {var}: gain(k=10)/gain(k=50) median {ratio.median():.2f} over {int(ok.sum())} conditions with gain(50) > 0.005; "
                 f"share of conditions with ratio >= 0.5: {(ratio >= 0.5).mean():.0%}")
lines.append("")

# --- Decision level
lines += ["## Decision level at alpha = 0.1 (mean over shifted conditions, all categories)", ""]
rows = []
for var in ("none", "b2a", "b2b", "oracle_a"):
    for k in (0, 5, 10, 25, 50):
        x = sh[(sh.variant == var) & (sh.k == k)]
        if x.empty:
            continue
        rows.append({"variant": var, "k": k, "AUROC": f"{x.auroc.mean():.3f}",
                     "FPR@source-thr": f"{x.fpr_src.mean():.3f}", "FPR@target-conformal-thr": "-" if k == 0 else f"{x.fpr_tgt.mean():.3f}",
                     "TPR@target-thr": "-" if k == 0 else f"{x.tpr_tgt.mean():.3f}"})
lines += [md(rows), ""]

# --- QR3: realised FPR of the target-conformal threshold vs alpha + Beta band (per draw, all cells)
from scipy.stats import beta  # noqa: E402
from lightshift.b3 import conformal_k  # noqa: E402

lines += ["## QR3 - realised FPR of the conformal threshold calibrated on k target normals (alpha = 0.1)", ""]
raw = d[(d["shift"] != "clean") & (d.k > 0) & d.fpr_tgt.notna()]
rows = []
for var in ("none", "b2a", "b2b"):
    for k in (5, 10, 25, 50):
        x = raw[(raw.variant == var) & (raw.k == k)]
        kk = conformal_k(k, 0.1)
        band = beta.ppf(0.99, k + 1 - kk, kk) if k + 1 - kk > 0 else 1.0
        rows.append({"variant": var, "k": k, "mean FPR": f"{x.fpr_tgt.mean():.3f}", "p90 FPR": f"{x.fpr_tgt.quantile(.9):.3f}",
                     "99% band (theory)": f"{band:.2f}", "cells over band": f"{(x.fpr_tgt > band).mean():.0%}",
                     "mean TPR": f"{x.tpr_tgt.mean():.3f}"})
lines += [md(rows), ""]
lines += ["(test-good sets are 20-40 images, so a single cell's FPR moves in steps of 2.5-5 points; read means, not cells.)", ""]

(OUT / "qr4_summary.md").write_text("\n".join(lines))
print("\n".join(lines))
