"""Build a MVTec Anomaly Detection 2 (MAD 2) submission package.

Runs our PatchCore-style inference (WRN-50 layer2+down+layer3, 256 px,
strided coreset memory bank built from <category>/train/good) over
test_private and test_private_mixed, and writes float16 .tiff anomaly maps
at model output size into the exact required directory structure:

  submission_folder/
    anomaly_images/<category>/{test_private,test_private_mixed}/<name>.tiff

Only categories present under data/mvtec-ad-2/ are included (use --categories
to override). Filenames mirror the input image names. The optional
thresholded folder is skipped (it is not required).

Usage:
  .venv/bin/python scripts/build_submission.py [--device cpu] [--out OUT]
  [--tar]  # also produce the .tar.gz (checker script should be run first)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import timm
import tifffile
import torchvision.transforms as T
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from pilot_shift_eval import build_bank  # noqa: E402

DATA = REPO / "data" / "mvtec-ad-2"
SOURCE_ROOT = Path("/mnt/data-ssd/mvtec-ad-2")

TR = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

MODEL_SIZE = 256  # maps saved at model output size (Tip 1: server zooms up)


@torch.no_grad()
def map_for_image(model, x, bank) -> np.ndarray:
    f2, f3 = model(x)
    f2 = F.interpolate(f2, scale_factor=0.5, mode="bilinear")
    f = torch.cat([f2.flatten(2), f3.flatten(2)], dim=1)[0].permute(1, 0)
    f = F.normalize(f, dim=1)
    d = (2.0 - 2.0 * f @ bank.T).min(dim=1).values
    side = int(round(d.numel() ** 0.5))
    amap = d.reshape(side, side).cpu().numpy()
    amap = np.asarray(Image.fromarray(amap).resize((MODEL_SIZE, MODEL_SIZE), Image.BILINEAR),
                      dtype=np.float32)
    return amap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--out", default=str(REPO / "results" / "mad2_submission" / "submission_folder"))
    ap.add_argument("--tar", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: only N images per split")
    ap.add_argument("--categories", nargs="*", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    model = timm.create_model("wide_resnet50_2", pretrained=True,
                              features_only=True, out_indices=(2, 3)).eval().to(device)
    cats = args.categories
    if not cats:
        cats = [d.name for d in DATA.iterdir() if d.is_dir()]

    out_root = Path(args.out)
    for cat in cats:
        train_dir = SOURCE_ROOT / cat / "train" / "good"
        files = sorted(train_dir.glob("*"))
        if not files:
            print(f"SKIP {cat}: no train images")
            continue
        bank = build_bank(files).to(device).float()
        for split in ("test_private", "test_private_mixed"):
            src_split = SOURCE_ROOT / cat / split
            dst = out_root / "anomaly_images" / cat / split
            dst.mkdir(parents=True, exist_ok=True)
            imgs = sorted(p for p in src_split.iterdir() if p.is_file())
            if args.limit:
                imgs = imgs[: args.limit]
            for i in range(0, len(imgs), 16):
                chunk = imgs[i:i + 16]
                x = torch.stack([TR(Image.open(p).convert("RGB")) for p in chunk]).to(device)
                for j, p in enumerate(chunk):
                    amap = map_for_image(model, x[j: j + 1], bank)
                    tname = p.name.rsplit(".", 1)[0] + ".tiff"
                    tifffile.imwrite(dst / tname, amap.astype(np.float16))
                if (i // 16) % 20 == 0:
                    print(f"{cat}/{split}: {i + len(chunk)}/{len(imgs)}", flush=True)
        print(f"DONE {cat}")
    print("output:", out_root)


if __name__ == "__main__":
    main()
