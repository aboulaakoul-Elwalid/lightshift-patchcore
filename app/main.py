"""LightShift-PatchCore demo API.

CPU-only PatchCore inference (timm wide_resnet50_2 features + exported memory
banks) with a live lighting-shift generator and conformal (B3) verdicts.

Run: uvicorn main:app --host 127.0.0.1 --port 8021
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import timm
import torchvision.transforms as T
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from lightshift.shift import SHIFT_FAMILIES, apply_shift  # noqa: E402

BANKS = Path(__file__).resolve().parent / "banks"
CATEGORIES = ["bottle", "hazelnut", "leather"]

app = FastAPI(title="LightShift-PatchCore demo", version="0.2")
app.mount("/static", StaticFiles(directory=Path(__file__).resolve().parent / "static"), name="static")

_tr = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_backbone = None
_banks: dict[str, dict] = {}


def backbone():
    global _backbone
    if _backbone is None:
        _backbone = timm.create_model(
            "wide_resnet50_2", pretrained=True, features_only=True, out_indices=(2, 3)
        ).eval()
    return _backbone


def get_bank(cat: str) -> dict:
    if cat not in _banks:
        _banks[cat] = torch.load(BANKS / f"{cat}.pt", map_location="cpu", weights_only=False)
    return _banks[cat]


@torch.no_grad()
def infer(pil: Image.Image, cat: str, fam: str | None, sev: int):
    arr0 = np.asarray(pil.convert("RGB")).astype(np.float32)
    arr = apply_shift(arr0, fam, sev, rng_seed=0) if fam else arr0
    img = Image.fromarray(arr.astype(np.uint8))
    x = _tr(img).unsqueeze(0)
    f2, f3 = backbone()(x)
    f2 = F.interpolate(f2, scale_factor=0.5, mode="bilinear")
    f = torch.cat([f2.flatten(2), f3.flatten(2)], dim=1).squeeze(0).permute(1, 0)
    f = F.normalize(f, dim=1)
    bank = get_bank(cat)["bank"].float()
    d = (2.0 - 2.0 * f @ bank.T).min(dim=1).values  # (P,)
    thr = float(get_bank(cat)["threshold"])
    side = int(round(d.numel() ** 0.5))
    # grid is (H/16)x(W/16); for square input it's square
    amap = d.reshape(side, side).numpy()
    # upsample to 256 and colorize
    lo, hi = float(np.percentile(amap, 5)), float(amap.max() - float(np.percentile(amap, 5)) + 1e-6)
    heat = np.clip((amap - lo) / hi, 0, 1)
    heat_img = heat_to_png(heat)
    preview = to_png(img.resize((384, 384)))
    overlay = overlay_png(pil, heat)
    score = float(d.max())
    return {
        "score": score,
        "threshold": thr,
        "verdict": "anomaly" if score > thr else "normal",
        "margin": score / max(thr, 1e-6),
        "heat_png": heat_img,
        "preview_png": preview,
        "overlay_png": overlay,
        "shifted_png": to_png(img.resize((384, 384))),
    }


def _png_bytes(img: Image.Image) -> str:
    import base64

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def heat_to_png(heat: np.ndarray) -> str:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import cm

    h = (heat * 255).astype(np.uint8)
    img = Image.fromarray(h).resize((384, 384), Image.BILINEAR)
    rgba = cm.inferno(np.asarray(img) / 255.0)
    return _png_bytes(Image.fromarray((rgba[..., :3] * 255).astype(np.uint8)))


def overlay_png(pil: Image.Image, heat: np.ndarray) -> str:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import cm

    base = pil.convert("RGB").resize((384, 384))
    base_np = np.asarray(base).astype(np.float32)
    h = np.asarray(Image.fromarray((heat * 255).astype(np.uint8)).resize(
        (base_np.shape[1], base_np.shape[0]), Image.BILINEAR)) / 255.0
    cmap = cm.inferno(h)[..., :3] * 255.0
    out = 0.55 * base_np + 0.45 * cmap
    return _png_bytes(Image.fromarray(out.astype(np.uint8)))


def to_png(img: Image.Image) -> str:
    return _png_bytes(img)


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(Path(__file__).resolve().parent / "static" / "index.html")


@app.get("/api/meta")
def meta():
    return {"categories": CATEGORIES, "families": list(SHIFT_FAMILIES), "severities": [1, 2, 3, 4, 5]}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), category: str = Form("bottle"),
                  family: str = Form("none"), severity: int = Form(0)):
    pil = Image.open(io.BytesIO(await file.read()))
    if category not in CATEGORIES:
        return {"error": f"unknown category {category}"}
    fam = None if family in (None, "none", "clean") else family
    if fam is not None and fam not in SHIFT_FAMILIES:
        return {"error": f"unknown family {family}"}
    sev = int(severity) if fam else 0
    if fam and not 1 <= sev <= 5:
        return {"error": "severity must be 1..5"}
    return infer(pil, category, fam, sev)
