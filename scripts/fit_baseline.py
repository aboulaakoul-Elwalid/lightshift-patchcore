"""Fit the vanilla PatchCore baseline for the reproduction gate (GO/NO-GO 1).

Usage:
  .venv/bin/python scripts/fit_baseline.py can [--coreset 0.25] [--seed 0]

Model: wide_resnet50_2, layers 2+3, pre-trained ImageNet weights, coreset
sampling ratio configurable. Config + receipt (fingerprint, metrics, git SHA,
elapsed) written under receipts/<bag_id>/patchcore/. Resumable: skips a
category if an identical-fingerprint receipt already exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore

REPO = Path(__file__).resolve().parents[1]
RECEIPTS = REPO / "receipts"


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def fingerprint(category: str, cfg: dict) -> str:
    blob = json.dumps({"task": "fit_baseline", "category": category, "cfg": cfg}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def run(category: str, coreset: float, seed: int, device: str = "auto", overwrite: bool = False) -> Path:
    cfg = {
        "backbone": "wide_resnet50_2",
        "layers": ["layer2", "layer3"],
        "coreset_sampling_ratio": coreset,
        "num_neighbors": 9,
        "image_size": 256,
        "seed": seed,
    }
    fp = fingerprint(category, cfg)
    bag = f"{category}-pcore-v1"
    receipt_dir = RECEIPTS / bag / "patchcore"
    receipt = receipt_dir / f"{fp}.json"
    if receipt.exists() and not overwrite:
        print(f"skip {category} {fp}: receipt exists")
        return receipt

    t0 = time.time()
    base = REPO / "data" / "fs" / category
    dm = Folder(
        name=category,
        root=str(base),
        normal_dir="normal",
        normal_test_dir="normal_test",
        abnormal_dir="abnormal",
        mask_dir="mask",
        train_batch_size=16,
        eval_batch_size=16,
        num_workers=4,
        val_split_mode="from_test",
        val_split_ratio=0.1,
        test_split_mode="from_dir",
        test_split_ratio=0.2,
        seed=seed,
    )
    model = Patchcore(
        backbone=cfg["backbone"],
        layers=tuple(cfg["layers"]),
        coreset_sampling_ratio=coreset,
        num_neighbors=cfg["num_neighbors"],
        pre_trained=True,
    )
    if device == "auto":
        device = "gpu"
    engine = Engine(accelerator=device, devices=1)
    engine.fit(model=model, datamodule=dm)
    engine.test(model=model, datamodule=dm)

    test_metrics = {k: v for k, v in engine.trainer.callback_metrics.items()}
    metrics = {}
    for k, v in test_metrics.items():
        try:
            metrics[k] = float(v)
        except (TypeError, ValueError):
            pass

    asset_dir = receipt_dir / f"{fp}-ckpt"
    asset_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), asset_dir / f"{bag}-{fp}.pt")

    # manual probe: score separation sanity check on the test loader
    dm.setup("test")
    model.eval()
    scores, labels = [], []
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in dm.test_dataloader():
            out = model.predict_step(batch, batch_idx=0)
            s = out.pred_score if isinstance(out, dict) is False else out["pred_score"]
            if hasattr(s, "detach"):
                s = s.detach().cpu().numpy()
            gs = out.gt_label if not isinstance(out, dict) else out["gt_label"]
            if hasattr(gs, "detach"):
                gs = gs.detach().cpu().long().numpy()
            scores.append(s); labels.append(gs)
    scores = np.concatenate(scores); labels = np.concatenate(labels)
    from sklearn.metrics import roc_auc_score
    manual_auroc = float(roc_auc_score(labels, scores))
    probe = {
        "n_test": int(labels.size),
        "n_bad": int(labels.sum()),
        "manual_image_auroc": manual_auroc,
        "score_mean_good": float(scores[labels == 0].mean()),
        "score_mean_bad": float(scores[labels == 1].mean()),
        "good_scores": [float(x) for x in scores[labels == 0][:16]],
        "bad_scores": [float(x) for x in scores[labels == 1][:16]],
    }
    print("PROBE:", json.dumps(probe, indent=2))

    receipt.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bag_id": bag,
        "fingerprint": fp,
        "category": category,
        "config": cfg,
        "git_sha": _git_sha(),
        "bus": {"device": device, "anomalib": __import__("anomalib").__version__},
        "elapsed_s": round(time.time() - t0, 1),
        "metrics": metrics,
        "probe": probe if "probe" in dir() else None,
    }
    payload["probe"] = probe
    receipt.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"receipt written: {receipt}")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return receipt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("categories", nargs="+", default=["can"], help="categories")
    ap.add_argument("--coreset", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    for c in args.categories:
        run(c, args.coreset, args.seed, args.device, args.overwrite)
