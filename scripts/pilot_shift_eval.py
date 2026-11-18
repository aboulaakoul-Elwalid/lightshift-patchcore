"""Pilot study (QR1/QR2): degradation of vanilla PatchCore under controlled
lighting shifts, with B2 (feature-stat alignment) and B3 (conformal threshold).

Inference-only, GPU-light (one process, 256px input). Protocol (seeded):
  * per category: train/good -> bank 70%, conformal calibration 15%,
    shift-target normals 15% (disjoint);
  * image score = max patch NN distance, L2-normalized layer2(+down)+layer3;
  * banking coreset: strided 25% subsample (documented deviation from greedy);
  * B2a: subtract per-channel mean gap (mu_shifted_target - mu_clean_target)
    from unnormalized query features, then renormalize;
  * B2b: additionally scale per-channel std toward the clean-target std;
  * B3: conformal image threshold from clean calibration scores, alpha=0.1.
Outputs: results/pilot/<cat>_pilot.json + results/pilot/pilot_summary.csv
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import timm
from PIL import Image
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lightshift.shift import SHIFT_FAMILIES, apply_shift  # noqa: E402
from lightshift.b3 import conformal_threshold  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MVT = Path("/mnt/data-ssd/mvtec_ad_full")
OUT = REPO / "results" / "pilot"
OUT.mkdir(parents=True, exist_ok=True)
ALPHA = 0.1
SEED = 0
CORESET_FRAC = 0.25

TR = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_model = None


def backbone():
    global _model
    if _model is None:
        _model = timm.create_model(
            "wide_resnet50_2", pretrained=True, features_only=True, out_indices=(2, 3)
        ).to(DEVICE).eval()
    return _model


@torch.no_grad()
def features(img: Image.Image, norm: bool = True) -> torch.Tensor:
    x = TR(img).unsqueeze(0).to(DEVICE)
    f2, f3 = backbone()(x)
    f2 = F.interpolate(f2, scale_factor=0.5, mode="bilinear")
    f = torch.cat([f2.flatten(2), f3.flatten(2)], dim=1).squeeze(0).permute(1, 0)  # (P, C)
    return F.normalize(f, dim=1) if norm else f


@torch.no_grad()
def image_score(query: torch.Tensor, bank: torch.Tensor) -> float:
    return float((2.0 - 2.0 * query @ bank.T).min(dim=1).values.max())


def shifted_image(img: Image.Image, fam: str, sev: int) -> Image.Image:
    arr = apply_shift(img, fam, sev, rng_seed=SEED)
    return Image.fromarray(arr.astype(np.uint8))


def load_split(cat: str):
    files = sorted((MVT / cat / "train" / "good").glob("*.png"))
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(files))
    b = int(0.70 * len(files))
    c = int(0.85 * len(files))
    bank_files = [files[i] for i in idx[:b]]
    cal_files = [files[i] for i in idx[b:c]]
    tgt_files = [files[i] for i in idx[c:]]
    test_good = sorted((MVT / cat / "test" / "good").glob("*.png"))
    test_bad = [p for p in (MVT / cat / "test").glob("*/*.png") if p.parent.name != "good"]
    return bank_files, cal_files, tgt_files, test_good, test_bad


def build_bank(bank_files) -> torch.Tensor:
    feats = torch.cat([features(Image.open(p).convert("RGB")) for p in bank_files])
    n_keep = max(1, int(CORESET_FRAC * feats.shape[0]))
    sel = torch.linspace(0, feats.shape[0] - 1, n_keep, dtype=torch.long)
    return feats[sel].contiguous()


def b2_stats(tgt_files, fam, sev):
    """Per-channel statistics of clean vs shifted target normal features."""
    mus_c, mus_s, stds_s = [], [], []
    for p in tgt_files:
        img = Image.open(p).convert("RGB")
        f_c = features(img, norm=False)
        f_s = features(shifted_image(img, fam, sev), norm=False)
        mus_c.append(f_c.mean(0))
        mus_s.append(f_s.mean(0))
        stds_s.append(f_s.std(0))
    mu_c = torch.stack(mus_c).mean(0)
    mu_s = torch.stack(mus_s).mean(0)
    std_s = torch.stack(stds_s).mean(0)
    return mu_c, mu_s, std_s


def run_cat(cat: str) -> dict:
    t0 = time.time()
    bank_files, cal_files, tgt_files, test_good, test_bad = load_split(cat)
    bank = build_bank(bank_files)
    print(f"[{cat}] bank {tuple(bank.shape)} good={len(test_good)} bad={len(test_bad)}", flush=True)

    cal_scores = np.array([image_score(features(Image.open(p).convert("RGB")), bank) for p in cal_files])
    thr = conformal_threshold(cal_scores, ALPHA)
    print(f"[{cat}] B3 threshold (alpha={ALPHA}, n={len(cal_scores)}): {thr:.4f}", flush=True)

    clean_good = [image_score(features(Image.open(p).convert("RGB")), bank) for p in test_good]
    clean_bad = [image_score(features(Image.open(p).convert("RGB")), bank) for p in test_bad]
    auroc_clean = roc_auc_score([0] * len(clean_good) + [1] * len(clean_bad), clean_good + clean_bad)
    fpr_clean = float(np.mean(np.array(clean_good) > thr))

    out = {
        "category": cat,
        "seed": SEED,
        "alpha": ALPHA,
        "threshold_clean": thr,
        "clean": {
            "image_auroc": float(auroc_clean),
            "fpr_at_thr": fpr_clean,
            "n_good": len(clean_good),
            "n_bad": len(clean_bad),
        },
        "families": {},
    }

    for fam in SHIFT_FAMILIES:
        for sev in range(1, 6):
            key = f"{fam}-s{sev}"
            mu_c, mu_s, std_s = b2_stats(tgt_files, fam, sev)
            delta_mu = (mu_s - mu_c).to(DEVICE)
            deltas_std_c = None  # B2b uses target-shifted std only (no clean std needed)

            rows: dict[str, list[float]] = {"vanilla": [], "b2a": [], "b2b": []}
            labels: list[int] = []

            def score_one(p: Path):
                img = Image.open(p).convert("RGB")
                f_raw = features(shifted_image(img, fam, sev), norm=False)
                rows["vanilla"].append(image_score(F.normalize(f_raw, dim=1), bank))
                rows["b2a"].append(image_score(F.normalize(f_raw - delta_mu, dim=1), bank))
                scale = torch.clamp(std_s.to(DEVICE), min=1e-3)
                rows["b2b"].append(image_score(F.normalize((f_raw - delta_mu) * scale, dim=1), bank))
                labels.append(0)

            for p in test_good:
                score_one(p)
            labels = [0] * len(test_good)
            for p in test_bad:
                img = Image.open(p).convert("RGB")
                f_raw = features(shifted_image(img, fam, sev), norm=False)
                rows["vanilla"].append(image_score(F.normalize(f_raw, dim=1), bank))
                rows["b2a"].append(image_score(F.normalize(f_raw - delta_mu, dim=1), bank))
                scale = torch.clamp(std_s.to(DEVICE), min=1e-3)
                rows["b2b"].append(image_score(F.normalize((f_raw - delta_mu) * scale, dim=1), bank))
                labels.append(1)

            labels_a = np.array(labels)
            famres = {}
            for variant, scores in rows.items():
                s = np.array(scores)
                good_s = s[labels_a == 0]
                famres[variant] = {
                    "image_auroc": float(roc_auc_score(labels_a, s)),
                    "fpr_at_thr": float(np.mean(good_s > thr)),
                    "mean_score_good": float(good_s.mean()),
                    "mean_score_bad": float(s[labels_a == 1].mean()),
                }
            famres["labels_n"] = {"good": int((labels_a == 0).sum()), "bad": int((labels_a == 1).sum())}
            out["families"][key] = famres
            print(
                f"[{cat}] {key}: " + ", ".join(
                    f"{v} auroc={famres[v]['image_auroc']:.3f} fpr={famres[v]['fpr_at_thr']:.2f}"
                    for v in ("vanilla", "b2a", "b2b")
                ),
                flush=True,
            )

    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


if __name__ == "__main__":
    cats = sys.argv[1:] or ["bottle", "hazelnut", "leather"]
    summaries = []
    for cat in cats:
        res = run_cat(cat)
        p = OUT / f"{cat}_pilot.json"
        p.write_text(json.dumps(res, indent=2, sort_keys=True))
        print(f"wrote {p}")
        summaries.append(res)

    with (OUT / "pilot_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "shift", "variant", "image_auroc", "fpr_at_thr", "mean_good", "mean_bad"])
        for res in summaries:
            w.writerow([res["category"], "clean", "vanilla",
                        f"{res['clean']['image_auroc']:.4f}", f"{res['clean']['fpr_at_thr']:.4f}", "", ""])
            for key, famres in res["families"].items():
                for variant in ("vanilla", "b2a", "b2b"):
                    r = famres[variant]
                    w.writerow([res["category"], key, variant,
                                f"{r['image_auroc']:.4f}", f"{r['fpr_at_thr']:.4f}",
                                f"{r['mean_score_good']:.2f}", f"{r['mean_score_bad']:.2f}"])
    print("done")
