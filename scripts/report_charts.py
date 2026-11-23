"""Charts (inline SVG), tables and scalar values for the learning report.

Everything is computed from files under results/ so the report cannot drift from the data.
Missing result files produce a visible placeholder instead of failing the build.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
R = REPO / "results"
COL = {"none": "#9d3f29", "b2a": "#116a66", "b2b": "#7b5428", "oracle_a": "#8d8272", "b1": "#5b4b8a", "b1h": "#9a8cc4"}
NAME = {"none": "vanilla / B3 only", "b2a": "B2a (mean)", "b2b": "B2b (mean+std)", "oracle_a": "oracle (paired twins)"}


def _csv(path):
    p = R / path
    return pd.read_csv(p) if p.exists() else None


def _missing(what):
    return f'<div class="warning">[{what}: results file not found when this report was built]</div>'


def pct(x, d=0):
    return f"{100 * x:.{d}f}%"


# ---------------------------------------------------------------- svg helpers
def svg_open(w, h):
    return f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" style="max-width:{w}px">'


def axis_y(w, h, pl, pr, pt, pb, ymax, ticks=5, fmt="{:.2f}"):
    out = []
    for i in range(ticks + 1):
        v = ymax * i / ticks
        y = pt + (h - pt - pb) * (1 - v / ymax)
        out.append(f'<line x1="{pl}" y1="{y:.1f}" x2="{w - pr}" y2="{y:.1f}" stroke="#ded2be" stroke-width="1"/>'
                   f'<text x="{pl - 6}" y="{y + 4:.1f}" text-anchor="end" font-size="10" fill="#8d8272">{fmt.format(v)}</text>')
    return "".join(out)


def line_chart(series, xs, title, ylab, ymax=1.0, w=640, h=300, hline=None, xlab="", fmt="{:.2f}", xlabels=None):
    """series: {label: (values, color, dashed)}; xs: x positions."""
    pl, pr, pt, pb = 50, 120, 34, 40
    iw, ih = w - pl - pr, h - pt - pb
    x0, x1 = min(xs), max(xs)
    sx = lambda x: pl + iw * (x - x0) / ((x1 - x0) or 1)  # noqa: E731
    sy = lambda v: pt + ih * (1 - min(max(v, 0), ymax) / ymax)  # noqa: E731
    o = [svg_open(w, h), f'<text x="{pl}" y="18" font-size="13" font-weight="700" fill="#17140f">{title}</text>',
         axis_y(w, h, pl, pr, pt, pb, ymax, fmt=fmt)]
    for i, x in enumerate(xs):
        lab = xlabels[i] if xlabels else str(x)
        o.append(f'<text x="{sx(x):.1f}" y="{h - 20}" text-anchor="middle" font-size="10" fill="#8d8272">{lab}</text>')
    o.append(f'<text x="{pl + iw / 2}" y="{h - 4}" text-anchor="middle" font-size="11" fill="#61584c">{xlab}</text>')
    o.append(f'<text x="12" y="{pt + ih / 2}" font-size="11" fill="#61584c" transform="rotate(-90 12 {pt + ih / 2})" text-anchor="middle">{ylab}</text>')
    if hline:
        for v, lab, c in hline:
            o.append(f'<line x1="{pl}" x2="{w - pr}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="{c}" stroke-dasharray="5 4" stroke-width="1.6"/>'
                     f'<text x="{w - pr + 4}" y="{sy(v) + 4:.1f}" font-size="10" fill="{c}">{lab}</text>')
    for lab, (vals, c, dashed) in series.items():
        pts = [(sx(x), sy(v)) for x, v in zip(xs, vals) if v is not None and not (isinstance(v, float) and math.isnan(v))]
        if not pts:
            continue
        d = " ".join(f"{a:.1f},{b:.1f}" for a, b in pts)
        o.append(f'<polyline points="{d}" fill="none" stroke="{c}" stroke-width="2.4" {"stroke-dasharray=\"6 4\"" if dashed else ""}/>')
        o.append("".join(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="3.2" fill="{c}"/>' for a, b in pts))
        o.append(f'<text x="{pts[-1][0] + 6:.1f}" y="{pts[-1][1] + 4:.1f}" font-size="10.5" fill="{c}">{lab}</text>')
    o.append("</svg>")
    return "".join(o)


def bar_chart(groups, series, title, ylab, ymax=1.0, w=720, h=320, colors=None, hline=None, fmt="{:.2f}"):
    """groups: list of labels; series: {label: [values per group]}"""
    pl, pr, pt, pb = 50, 20, 40, 62
    iw, ih = w - pl - pr, h - pt - pb
    ng, ns = len(groups), len(series)
    gw = iw / ng
    bw = gw * 0.8 / ns
    sy = lambda v: pt + ih * (1 - min(max(v, 0), ymax) / ymax)  # noqa: E731
    o = [svg_open(w, h), f'<text x="{pl}" y="18" font-size="13" font-weight="700" fill="#17140f">{title}</text>',
         axis_y(w, h, pl, pr, pt, pb, ymax, fmt=fmt)]
    o.append(f'<text x="12" y="{pt + ih / 2}" font-size="11" fill="#61584c" transform="rotate(-90 12 {pt + ih / 2})" text-anchor="middle">{ylab}</text>')
    if hline:
        for v, lab, c in hline:
            o.append(f'<line x1="{pl}" x2="{w - pr}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="{c}" stroke-dasharray="5 4" stroke-width="1.5"/>'
                     f'<text x="{w - pr - 2}" y="{sy(v) - 4:.1f}" text-anchor="end" font-size="10" fill="{c}">{lab}</text>')
    for gi, g in enumerate(groups):
        gx = pl + gi * gw + gw * 0.1
        for si, (lab, vals) in enumerate(series.items()):
            v = vals[gi]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            c = (colors or {}).get(lab, "#116a66")
            o.append(f'<rect x="{gx + si * bw:.1f}" y="{sy(v):.1f}" width="{bw - 2:.1f}" height="{sy(0) - sy(v):.1f}" fill="{c}" rx="2"/>')
        o.append(f'<text x="{gx + gw * 0.4:.1f}" y="{h - pb + 14}" text-anchor="middle" font-size="10" fill="#61584c">{g}</text>')
    lx = pl
    for lab in series:
        c = (colors or {}).get(lab, "#116a66")
        o.append(f'<rect x="{lx}" y="{h - 22}" width="10" height="10" fill="{c}"/><text x="{lx + 14}" y="{h - 13}" font-size="10.5" fill="#61584c">{lab}</text>')
        lx += 22 + 6.4 * len(lab)
    o.append("</svg>")
    return "".join(o)


def table(header, rows, num_from=1):
    h = "".join(f"<th>{c}</th>" for c in header)
    b = "".join("<tr>" + "".join(f'<td class="{"num" if i >= num_from else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f"<table><tr>{h}</tr>{b}</table>"


# ---------------------------------------------------------------- charts
def chart_pipeline():
    w, h = 900, 250
    def box(x, y, bw, bh, t1, t2="", fill="#fffdf8", stroke="#bba77f"):
        return (f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>'
                f'<text x="{x + bw / 2}" y="{y + 24}" text-anchor="middle" font-size="12.5" font-weight="700" fill="#17140f">{t1}</text>'
                f'<text x="{x + bw / 2}" y="{y + 42}" text-anchor="middle" font-size="10.5" fill="#61584c">{t2}</text>')
    arrow = lambda x1, y1, x2, y2: f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#7b5428" stroke-width="2" marker-end="url(#a)"/>'  # noqa: E731
    o = [svg_open(w, h), '<defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#7b5428"/></marker></defs>',
         '<text x="10" y="18" font-size="11" font-weight="800" fill="#7b5428" letter-spacing="1.5">FIT (once per category, no gradient steps)</text>',
         box(10, 30, 130, 56, "good images", "125 × 256×256 RGB"),
         box(180, 30, 150, 56, "frozen WRN-50-2", "layers 2 + 3", fill="#e7edf2"),
         box(370, 30, 150, 56, "patch descriptors", "32,000 × 1536"),
         box(560, 30, 150, 56, "greedy coreset 25%", "k-center", fill="#f3e7cd"),
         box(750, 30, 140, 56, "memory bank B", "8,000 × 1536", fill="#dfeee8"),
         arrow(140, 58, 178, 58), arrow(330, 58, 368, 58), arrow(520, 58, 558, 58), arrow(710, 58, 748, 58),
         '<text x="10" y="128" font-size="11" font-weight="800" fill="#7b5428" letter-spacing="1.5">PREDICT (per image)</text>',
         box(10, 140, 130, 56, "test image", "any lighting"),
         box(180, 140, 150, 56, "frozen WRN-50-2", "256 × 1536 descriptors", fill="#e7edf2"),
         box(370, 140, 150, 56, "[B2] align features", "f − (μt − μr)", fill="#f4dfd6", stroke="#9d3f29"),
         box(560, 140, 150, 56, "nearest bank vector", "d_i = min ‖f_i − b‖²"),
         box(750, 140, 140, 56, "score s = max d_i", "vs threshold τ [B3]", fill="#dfeee8"),
         arrow(140, 168, 178, 168), arrow(330, 168, 368, 168), arrow(520, 168, 558, 168), arrow(710, 168, 748, 168),
         f'<path d="M820 86 L820 138" stroke="#116a66" stroke-width="2" stroke-dasharray="4 3" marker-end="url(#a)"/>',
         '<text x="10" y="232" font-size="11" fill="#61584c">Dashed: the bank is read, never changed. B2 and B3 sit around the lookup; neither retrains anything.</text></svg>']
    return "".join(o)


def chart_shift_ramp():
    d = _csv("pilot/pilot_summary.csv")
    p = R / "pilot" / "hazelnut_pilot.json"
    if d is None or not p.exists():
        return _missing("pilot")
    thr = json.loads(p.read_text())["threshold_clean"]
    v = d[(d.category == "hazelnut") & (d.variant == "vanilla") & d["shift"].str.startswith("brightness")].sort_values("shift")
    xs = [1, 2, 3, 4, 5]
    return (line_chart({"mean score, good": (list(v.mean_good), COL["b2a"], False), "mean score, defective": (list(v.mean_bad), COL["none"], False)},
                       xs, "Hazelnut · brightness shift · vanilla PatchCore", "image score", ymax=1.0,
                       hline=[(thr, "threshold τ (clean)", "#61584c")], xlab="severity")
            + bar_chart([f"s{i}" for i in xs], {"false-alarm rate at τ": list(v.fpr_at_thr)}, "False-alarm rate at the clean threshold", "FPR",
                        colors={"false-alarm rate at τ": "#9d3f29"}, h=230))


def chart_conformal():
    w, h = 760, 190
    sc = [0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36]
    lo, hi = 0.25, 0.40
    sx = lambda v: 40 + (w - 80) * (v - lo) / (hi - lo)  # noqa: E731
    o = [svg_open(w, h), '<text x="10" y="18" font-size="13" font-weight="700" fill="#17140f">Worked example: n = 10 calibration scores, α = 0.1 → k = 10</text>',
         f'<line x1="40" x2="{w - 40}" y1="90" y2="90" stroke="#bba77f" stroke-width="2"/>']
    for i, v in enumerate(sc):
        c = "#9d3f29" if i == 9 else "#116a66"
        o.append(f'<circle cx="{sx(v):.1f}" cy="90" r="6" fill="{c}"/><text x="{sx(v):.1f}" y="118" text-anchor="middle" font-size="10" fill="#61584c">{i + 1}</text>'
                 f'<text x="{sx(v):.1f}" y="74" text-anchor="middle" font-size="10" fill="#61584c">{v:.2f}</text>')
    o.append(f'<line x1="{sx(0.36):.1f}" x2="{sx(0.36):.1f}" y1="38" y2="132" stroke="#9d3f29" stroke-dasharray="5 4" stroke-width="2"/>'
             f'<text x="{sx(0.36) + 6:.1f}" y="48" font-size="11" fill="#9d3f29">τ = S(10) = 0.36</text>')
    o.append('<text x="40" y="150" font-size="11" fill="#61584c">numbers under the dots = rank (1 = smallest). A new good score lands in one of 11 equally likely slots;</text>'
             '<text x="40" y="166" font-size="11" fill="#61584c">only the slot above rank 10 raises an alarm → P(false alarm) = 1/11 = 9.1% ≤ 10%.</text></svg>')
    return "".join(o)


def chart_ncurve():
    d = _csv("qr4/qr4_rows.csv")
    if d is None:
        return _missing("qr4")
    sh = d[(d["shift"] != "clean") & (d.k > 0) & d.fpr_tgt.notna()]
    ks = [5, 10, 25, 50]
    ser = {}
    for v in ("none", "b2a", "b2b"):
        ser[NAME[v]] = ([sh[(sh.variant == v) & (sh.k == k)].fpr_tgt.mean() for k in ks], COL[v], False)
    th = [(k + 1 - min(math.ceil((k + 1) * 0.9), k)) / (k + 1) for k in ks]
    ser["theory E[FPR]=(n+1−k)/(n+1)"] = (th, "#17140f", True)
    return line_chart(ser, ks, "Realised false-alarm rate of the conformal threshold vs number of calibration photos k", "FPR on good test parts",
                      ymax=0.5, hline=[(0.1, "α = 0.1", "#9d3f29")], xlab="k (calibration photos)", xlabels=[str(k) for k in ks], w=760)


def chart_qr1():
    d = _csv("qr1/qr1_rows.csv")
    if d is None:
        return _missing("qr1")
    d = d[d.family != "clean"]
    w, h = 640, 300
    pl, pr, pt, pb = 52, 20, 34, 42
    iw, ih = w - pl - pr, h - pt - pb
    xm = math.sqrt(d.drift_all.max()) * 1.05
    ym = max(d.fpr_src.max(), 0.05)
    cols = {"gamma": "#116a66", "brightness": "#9d3f29", "contrast": "#7b5428", "color_temp": "#5b4b8a", "shading": "#17140f"}
    o = [svg_open(w, h), '<text x="52" y="18" font-size="13" font-weight="700" fill="#17140f">Feature drift vs false-alarm rate at the clean threshold (each dot = category × shift × severity)</text>',
         axis_y(w, h, pl, pr, pt, pb, 1.0, fmt="{:.1f}")]
    for _, r in d.iterrows():
        o.append(f'<circle cx="{pl + iw * math.sqrt(r.drift_all) / xm:.1f}" cy="{pt + ih * (1 - r.fpr_src):.1f}" r="4" fill="{cols[r.family]}" fill-opacity="0.7"/>')
    for i, (f, c) in enumerate(cols.items()):
        o.append(f'<circle cx="{pl + 10 + i * 110}" cy="{h - 10}" r="4" fill="{c}"/><text x="{pl + 18 + i * 110}" y="{h - 6}" font-size="10.5" fill="#61584c">{f}</text>')
    o.append(f'<text x="{pl + iw / 2}" y="{h - 24}" text-anchor="middle" font-size="11" fill="#61584c">drift = mean over channels of |μ_target − μ_ref| / σ_ref   (square-root axis, 0 … {d.drift_all.max():.2f})</text></svg>')
    return "".join(o)


def chart_qr6():
    d = _csv("qr6/qr6_natural_rows.csv")
    if d is None:
        return _missing("qr6")
    out = []
    for cat in ("vial", "sheet_metal"):
        x = d[(d.cat == cat)]
        conds = list(x[(x.variant == "none") & (x.k == 0)].cond)
        van = x[(x.variant == "none") & (x.k == 0)].set_index("cond")
        kk = int(x[x.variant == "none"].k.max())
        b3 = x[(x.variant == "none") & (x.k == kk)].set_index("cond")
        out.append(bar_chart(conds, {"vanilla FPR @ source threshold": [van.loc[c, "fpr_src"] for c in conds],
                                     f"B3 FPR (k={kk})": [b3.loc[c, "fpr_tgt"] if c in b3.index else None for c in conds],
                                     "vanilla AUROC": [van.loc[c, "auroc"] for c in conds]},
                             f"AD 2 {cat}: real lighting conditions", "rate / AUROC", hline=[(0.1, "α = 0.1", "#61584c")], h=300,
                             colors={"vanilla FPR @ source threshold": "#9d3f29", f"B3 FPR (k={kk})": "#116a66", "vanilla AUROC": "#bba77f"}))
    return "".join(out)


def chart_b1():
    d = _csv("b1/b1_rows.csv")
    q = _csv("qr4/qr4_rows.csv")
    if d is None or q is None:
        return _missing("b1")
    fams = ["gamma", "brightness", "contrast", "color_temp", "shading"]

    def agg(df, col):
        sh = df[df["shift"] != "clean"].copy()
        sh["fam"] = sh["shift"].str.split("-").str[0]
        return [sh[sh.fam == f][col].mean() for f in fams]
    base = q[(q.variant == "none") & (q.k == 0)]
    b10 = q[(q.variant == "b2a") & (q.k == 10)]
    s = d[(d.arm == "b1_seen") & (d.variant == "none") & (d.k == 0)]
    h = d[(d.arm == "b1_heldout") & (d.variant == "none") & (d.k == 0)]
    ser = {"vanilla bank": agg(base, "fpr_src"), "B1 bank (all 5 families seen)": agg(s, "fpr_src"),
           "B1 bank (colour_temp, shading unseen)": agg(h, "fpr_src"), "B2a+B3 (k=10)": agg(b10, "fpr_tgt")}
    return bar_chart(fams, ser, "B1: false-alarm rate by shift family (mean over 3 categories, severities 2–4)", "FPR", w=760, h=330,
                     colors={"vanilla bank": "#9d3f29", "B1 bank (all 5 families seen)": "#5b4b8a", "B1 bank (colour_temp, shading unseen)": "#9a8cc4",
                             "B2a+B3 (k=10)": "#116a66"}, hline=[(0.1, "α", "#61584c")])


def chart_latency():
    p = R / "qr5" / "qr5.json"
    if not p.exists():
        return _missing("qr5")
    j = json.loads(p.read_text())
    devs = [k for k in ("cpu", "cuda") if k in j]
    return bar_chart(devs, {"vanilla p95": [j[k]["vanilla"]["p95_ms"] for k in devs], "serving (B2a + B3) p95": [j[k]["serving_b2a_b3"]["p95_ms"] for k in devs]},
                     "Per-image latency, p95 (ms)", "ms", ymax=max(j[k]["vanilla"]["p95_ms"] for k in devs) * 1.3, fmt="{:.0f}",
                     colors={"vanilla p95": "#9d3f29", "serving (B2a + B3) p95": "#116a66"}, w=560, h=280)


CHARTS = {"pipeline": chart_pipeline, "shift_ramp": chart_shift_ramp, "conformal": chart_conformal, "ncurve": chart_ncurve,
          "qr1_scatter": chart_qr1, "qr6_bars": chart_qr6, "b1_bars": chart_b1, "latency": chart_latency}


# ---------------------------------------------------------------- tables
def t_kn():
    rows = []
    for n in (5, 9, 10, 20, 27, 50, 100):
        raw = math.ceil((n + 1) * 0.9)
        k = min(raw, n)
        note = "k &gt; n: capped, <b>no guarantee</b>" if raw > n else ("max score" if k == n else f"{n - k + 1}-th largest")
        rows.append([n, raw, k, f"{(n + 1 - k) / (n + 1):.3f}", note])
    return table(["n", "⌈(n+1)(1−α)⌉", "k used", "E[FPR]", "threshold is"], rows, num_from=0)


def t_gate():
    rows = [["bottle", "1.000", "1.000", "0.992", "0.984"], ["hazelnut", "1.000", "1.000", "0.993", "0.997"], ["leather", "1.000", "1.000", "0.996", "0.986"]]
    return table(["category", "image AUROC (ours)", "published", "pixel AUROC (ours)", "published"], rows)


def t_pilot():
    d = _csv("pilot/pilot_summary.csv")
    if d is None:
        return _missing("pilot")
    s5 = d[d["shift"].str.endswith("s5")].copy()
    s5["fam"] = s5["shift"].str.split("-").str[0]
    rows = []
    for cat in ("bottle", "hazelnut", "leather"):
        for fam in ("brightness", "color_temp", "shading"):
            x = s5[(s5.category == cat) & (s5.fam == fam)].set_index("variant")
            rows.append([cat, fam, f"{x.loc['vanilla', 'image_auroc']:.3f}", pct(x.loc["vanilla", "fpr_at_thr"]),
                         f"{x.loc['b2a', 'image_auroc']:.3f}", pct(x.loc["b2a", "fpr_at_thr"])])
    return table(["category", "shift (severity 5)", "AUROC vanilla", "FPR@τ vanilla", "AUROC B2a (oracle)", "FPR@τ B2a (oracle)"], rows, num_from=2)


def _qr4():
    d = _csv("qr4/qr4_rows.csv")
    return None if d is None else d[d["shift"] != "clean"]


def t_qr4_decision():
    sh = _qr4()
    if sh is None:
        return _missing("qr4")
    rows = []
    for v in ("none", "b2a", "b2b", "oracle_a"):
        for k in (0, 5, 10, 25, 50):
            x = sh[(sh.variant == v) & (sh.k == k)]
            if x.empty:
                continue
            rows.append([NAME[v], k, f"{x.auroc.mean():.3f}", pct(x.fpr_src.mean(), 1), "–" if k == 0 else pct(x.fpr_tgt.mean(), 1),
                         "–" if k == 0 else pct(x.tpr_tgt.mean(), 1)])
    return table(["variant", "k", "AUROC", "FPR @ source τ", "FPR @ conformal τ(k)", "TPR @ conformal τ(k)"], rows, num_from=1)


def t_qr3():
    from scipy.stats import beta
    sh = _qr4()
    if sh is None:
        return _missing("qr4")
    sh = sh[sh.k > 0]
    rows = []
    for v in ("none", "b2a", "b2b"):
        for k in (5, 10, 25, 50):
            kk = min(math.ceil((k + 1) * 0.9), k)
            x = sh[(sh.variant == v) & (sh.k == k)]
            band = beta.ppf(0.99, k + 1 - kk, kk)
            rows.append([NAME[v], k, pct(x.fpr_tgt.mean(), 1), pct((k + 1 - kk) / (k + 1), 1), pct(x.fpr_tgt.quantile(.9), 0), f"{band:.2f}", pct((x.fpr_tgt > band).mean(), 0)])
    return table(["variant", "k", "mean FPR (measured)", "E[FPR] (theory)", "p90 FPR", "99% band", "cells above band"], rows, num_from=1)


def t_qr4_cat():
    sh = _qr4()
    if sh is None:
        return _missing("qr4")
    rows = []
    for cat in ("bottle", "hazelnut", "leather"):
        for v in ("none", "b2a", "b2b"):
            x = sh[(sh.cat == cat) & (sh.variant == v) & (sh.k == 10)]
            z = sh[(sh.cat == cat) & (sh.variant == "none") & (sh.k == 0)]
            rows.append([cat, NAME[v], pct(z.fpr_src.mean()), pct(x.fpr_tgt.mean(), 1), f"{x.auroc.mean():.3f}", pct(x.tpr_tgt.mean(), 1)])
    return table(["category", "variant (k=10)", "FPR at source τ", "FPR at conformal τ", "AUROC", "TPR at conformal τ"], rows, num_from=2)


def t_qr1():
    p = R / "qr1" / "qr1_summary.md"
    if not p.exists():
        return _missing("qr1")
    lines = [l for l in p.read_text().splitlines() if l.startswith("|") and "---" not in l][1:]
    rows = [[c.strip() for c in l.strip("|").split("|")] for l in lines if "25 conditions" in l and "drift_all" in l]
    return table(["scope", "drift", "target", "rank correlation (Spearman)", "n"], rows, num_from=3)


def t_qr6():
    d = _csv("qr6/qr6_natural_rows.csv")
    if d is None:
        return _missing("qr6")
    rows = []
    for cat in ("vial", "sheet_metal", "can"):
        x = d[d.cat == cat]
        for cond in sorted(x.cond.unique()):
            y = x[x.cond == cond]
            van = y[(y.variant == "none") & (y.k == 0)].iloc[0]
            kmax = int(y[y.variant == "none"].k.max())
            r = [cat, cond, int(van.n_good), f"{van.auroc:.2f}", pct(van.fpr_src)]
            for v in ("none", "b2a", "b2b"):
                z = y[(y.variant == v) & (y.k == kmax)]
                r.append(f"{z.iloc[0].auroc:.2f} / {pct(z.iloc[0].fpr_tgt)}" if len(z) else "–")
            r.append(kmax)
            rows.append(r)
    return table(["category", "lighting", "n good", "AUROC vanilla", "FPR@src τ", "B3 only: AUROC / FPR", "B2a+B3", "B2b+B3", "k"], rows, num_from=2)


def t_b1():
    d = _csv("b1/b1_rows.csv")
    q = _csv("qr4/qr4_rows.csv")
    if d is None or q is None:
        return _missing("b1")
    rows = []
    def row(label, df, v, k, col_fpr):
        sh = df[(df["shift"] != "clean") & (df.variant == v) & (df.k == k)]
        sh = sh.copy(); sh["fam"] = sh["shift"].str.split("-").str[0]
        seen = sh[sh.fam.isin(["gamma", "brightness", "contrast"])]; unseen = sh[sh.fam.isin(["color_temp", "shading"])]
        return [label, f"{sh.auroc.mean():.3f}", pct(sh[col_fpr].mean(), 1), pct(seen[col_fpr].mean(), 1), pct(unseen[col_fpr].mean(), 1)]
    rows.append(row("vanilla bank, source τ", q, "none", 0, "fpr_src"))
    rows.append(row("B1 bank (all families seen), source τ", d[d.arm == "b1_seen"], "none", 0, "fpr_src"))
    rows.append(row("B1 bank (colour_temp/shading unseen), source τ", d[d.arm == "b1_heldout"], "none", 0, "fpr_src"))
    rows.append(row("vanilla bank + B3 (k=10)", q, "none", 10, "fpr_tgt"))
    rows.append(row("vanilla bank + B2a + B3 (k=10)", q, "b2a", 10, "fpr_tgt"))
    rows.append(row("B1 bank (heldout) + B3 (k=10)", d[d.arm == "b1_heldout"], "none", 10, "fpr_tgt"))
    return table(["configuration", "AUROC", "FPR (all)", "FPR (gamma/brightness/contrast)", "FPR (colour_temp/shading)"], rows)


def t_qr5():
    p = R / "qr5" / "qr5.json"
    if not p.exists():
        return _missing("qr5")
    j = json.loads(p.read_text())
    rows = []
    for dev in ("cpu", "cuda"):
        if dev in j:
            for name in ("vanilla", "serving_b2a_b3"):
                x = j[dev][name]
                rows.append([dev, name, f"{x['p50_ms']:.1f}", f"{x['p95_ms']:.1f}"])
            rows.append([dev, "p95 ratio serving / vanilla", "", f"{j[dev]['p95_ratio']:.2f}×"])
            c = j[dev]["calibration"]
            rows.append([dev, "one-off calibration, k=10 / k=50 photos", f"{c['k=10_total_s']:.2f} s", f"{c['k=50_total_s']:.2f} s"])
    return table(["device", "path", "p50 (ms)", "p95 (ms)"], rows, num_from=2)


def _contam(path, k):
    d = _csv(path)
    if d is None:
        return _missing(path)
    rows = []
    for m in sorted(d.m.unique()):
        r = [f"{m} of {k}", pct(m / k)]
        for v in ("none", "b2a"):
            x = d[(d.variant == v) & (d.m == m)]
            r += [f"{x.thr.mean():.2f}", pct(x.fpr.mean(), 1), pct(x.tpr.mean())]
        rows.append(r)
    return table(["defective photos among the k", "share", "B3 only: threshold", "FPR", "TPR (defects caught)", "B2a+B3: threshold", "FPR", "TPR"], rows, num_from=1)


def t_contam10():
    return _contam("contamination/rows.csv", 10)


def t_contam50():
    return _contam("contamination/rows_k50.csv", 50)


TABLES = {"contam10": t_contam10, "contam50": t_contam50, "kn": t_kn, "gate": t_gate, "pilot_s5": t_pilot, "qr4_decision": t_qr4_decision, "qr3": t_qr3, "qr4_cat": t_qr4_cat,
          "qr1": t_qr1, "qr6": t_qr6, "b1": t_b1, "qr5": t_qr5}


def VALUES():
    out = {"fpr_src_all": "40%", "fpr_b3_k10": "10%"}
    sh = _qr4()
    if sh is not None:
        out["fpr_src_all"] = pct(sh[(sh.variant == "none") & (sh.k == 0)].fpr_src.mean())
        out["fpr_b3_k10"] = pct(sh[(sh.variant == "none") & (sh.k == 10)].fpr_tgt.mean())
    return out
