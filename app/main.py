"""LightShift-PatchCore demo API (v0.3).

CPU PatchCore inference with
  * a live lighting-shift generator,
  * B3: conformal threshold recalibrated on k normal photos from the NEW lighting,
  * B2a: per-channel mean alignment of patch features estimated from those same k photos.

Flow: POST /api/calibrate (k normals, optional simulated lighting) -> calib_id
      POST /api/predict with calib_id -> vanilla vs B3-only vs B2+B3 verdicts side by side.

Run: uvicorn main:app --host 127.0.0.1 --port 8021
"""
from __future__ import annotations

import base64
import io
import sys
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))  # repo layout; on the server the package sits next to main.py
from lightshift import b2  # noqa: E402
from lightshift.b3 import conformal_k, conformal_threshold  # noqa: E402
from lightshift.patchcore import Extractor, l2norm  # noqa: E402
from lightshift.shift import SHIFT_FAMILIES, shifted_pil  # noqa: E402

BANKS = HERE / "banks"
CALIB = HERE / "static" / "calib"
CATEGORIES = ["bottle", "hazelnut", "leather"]
ALPHA = 0.1

app = FastAPI(title="LightShift-PatchCore demo", version="0.3")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

_ext: Extractor | None = None
_banks: dict[str, dict] = {}
_sessions: "OrderedDict[str, dict]" = OrderedDict()


def ext() -> Extractor:
    global _ext
    if _ext is None:
        _ext = Extractor(torch.device("cpu"))
    return _ext


def get_bank(cat: str) -> dict:
    if cat not in _banks:
        b = torch.load(BANKS / f"{cat}.pt", map_location="cpu", weights_only=False)
        b["bank"] = b["bank"].float()
        _banks[cat] = b
    return _banks[cat]


def ref_moments(b: dict) -> b2.Moments:
    return b2.Moments(b["ref_mean"], b["ref_var"])


def maybe_shift(pil: Image.Image, fam: str | None, sev: int) -> Image.Image:
    pil = pil.convert("RGB")
    return shifted_pil(pil, fam, sev, rng_seed=0) if fam else pil


@torch.no_grad()
def patch_dists(raw: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
    q = l2norm(raw)
    return (2.0 - 2.0 * q @ bank.T).min(dim=1).values  # (P,)


# ------------------------------------------------------------------ images
def _png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _heat(d: torch.Tensor):
    side = int(round(d.numel() ** 0.5))
    a = d.reshape(side, side).numpy()
    lo = float(np.percentile(a, 5))
    return np.clip((a - lo) / (float(a.max()) - lo + 1e-6), 0, 1)


def overlay_png(pil: Image.Image, heat: np.ndarray) -> str:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import cm

    base = np.asarray(pil.convert("RGB").resize((384, 384))).astype(np.float32)
    h = np.asarray(Image.fromarray((heat * 255).astype(np.uint8)).resize((384, 384), Image.BILINEAR)) / 255.0
    out = 0.55 * base + 0.45 * cm.inferno(h)[..., :3] * 255.0
    return _png(Image.fromarray(out.astype(np.uint8)))


# ------------------------------------------------------------------ API
@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/meta")
def meta():
    return {"categories": CATEGORIES, "families": list(SHIFT_FAMILIES), "severities": [1, 2, 3, 4, 5], "version": "0.3",
            "builtin_normals": {c: len(list((CALIB / c).glob("*.jpg"))) for c in CATEGORIES if (CALIB / c).exists()}}


def _check(category: str, family: str, severity: int):
    if category not in CATEGORIES:
        return None, None, f"unknown category {category}"
    fam = None if family in (None, "none", "clean", "") else family
    if fam is not None and fam not in SHIFT_FAMILIES:
        return None, None, f"unknown family {family}"
    sev = int(severity) if fam else 0
    if fam and not 1 <= sev <= 5:
        return None, None, "severity must be 1..5"
    return fam, sev, None


@app.post("/api/calibrate")
async def calibrate(category: str = Form("bottle"), family: str = Form("none"), severity: int = Form(0),
                    k: int = Form(10), use_builtin: bool = Form(True), files: list[UploadFile] = File(default=[])):
    """Calibrate on k normal photos of the new lighting.

    use_builtin: take the first k held-out normal photos shipped with the demo and apply the chosen
    simulated lighting to them. Otherwise use the uploaded files as they are (apply the shift
    only if family != none, useful to simulate a lighting with your own photos).
    """
    fam, sev, err = _check(category, family, severity)
    if err:
        return {"error": err}
    imgs: list[Image.Image] = []
    if use_builtin or not files:
        paths = sorted((CALIB / category).glob("*.jpg"))[: max(2, min(int(k), 12))]
        imgs = [Image.open(p) for p in paths]
    else:
        for f in files[:50]:
            imgs.append(Image.open(io.BytesIO(await f.read())))
    if len(imgs) < 2:
        return {"error": "need at least 2 normal photos"}
    b = get_bank(category)
    E = ext()
    raws = [E.raw(maybe_shift(i, fam, sev)) for i in imgs]
    tgt = b2.pool([b2.image_moments(r) for r in raws])
    ref = ref_moments(b)
    s_none = np.array([float(patch_dists(r, b["bank"]).max()) for r in raws])
    s_b2 = np.array([float(patch_dists(b2.align(r, "mean", tgt, ref), b["bank"]).max()) for r in raws])
    n = len(imgs)
    drift = float(((tgt.mean - ref.mean).abs() / torch.sqrt(ref.var + 1e-6)).mean())
    sid = uuid.uuid4().hex[:12]
    _sessions[sid] = {"cat": category, "tgt": tgt, "thr_none": conformal_threshold(s_none, ALPHA),
                      "thr_b2": conformal_threshold(s_b2, ALPHA), "fam": fam, "sev": sev}
    while len(_sessions) > 32:
        _sessions.popitem(last=False)
    return {"calib_id": sid, "n": n, "alpha": ALPHA, "rank_k": conformal_k(n, ALPHA),
            "threshold_source": float(b["threshold"]), "threshold_b3": _sessions[sid]["thr_none"],
            "threshold_b2b3": _sessions[sid]["thr_b2"], "drift": drift,
            "scores_unaligned": sorted(map(float, s_none)), "scores_aligned": sorted(map(float, s_b2)),
            "family": fam or "none", "severity": sev}


def _one(raw: torch.Tensor, bank: torch.Tensor, thr: float, pil: Image.Image, heat=True):
    d = patch_dists(raw, bank)
    s = float(d.max())
    out = {"score": s, "threshold": float(thr), "verdict": "anomaly" if s > thr else "normal", "margin": s / max(thr, 1e-6)}
    if heat:
        out["overlay_png"] = overlay_png(pil, _heat(d))
    return out


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), category: str = Form("bottle"), family: str = Form("none"),
                  severity: int = Form(0), calib_id: str = Form("")):
    fam, sev, err = _check(category, family, severity)
    if err:
        return {"error": err}
    pil = Image.open(io.BytesIO(await file.read())).convert("RGB")
    img = maybe_shift(pil, fam, sev)
    b = get_bank(category)
    raw = ext().raw(img)
    van = _one(raw, b["bank"], b["threshold"], pil)
    res = {**van, "heat_png": van["overlay_png"], "preview_png": _png(img.resize((384, 384))),
           "shifted_png": _png(img.resize((384, 384)))}
    if calib_id:
        s = _sessions.get(calib_id)
        if s is None or s["cat"] != category:
            return {"error": "calibration expired or for another category: calibrate again"}
        al = b2.align(raw, "mean", s["tgt"], ref_moments(b))
        res["pipelines"] = {
            "vanilla": van,
            "b3": _one(raw, b["bank"], s["thr_none"], pil, heat=False),
            "b2_b3": _one(al, b["bank"], s["thr_b2"], pil),
        }
    return res
