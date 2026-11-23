"""Minimal PatchCore: frozen WideResNet-50 features, coreset memory bank, NN scoring.

Matches the pilot pipeline (layer2 downsampled x0.5 + layer3, 256px, L2-normalised
patch vectors, image score = max over patches of nearest-bank distance) but with a
real greedy (k-center) coreset instead of the pilot's strided subsample.

Raw (unnormalised) features are exposed because B2 aligns statistics *before* the
L2 normalisation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Extractor:
    def __init__(self, device: torch.device | None = None, size: int = 256):
        import timm
        import torchvision.transforms as T

        self.device = device or default_device()
        self.size = size
        self.model = timm.create_model(
            "wide_resnet50_2", pretrained=True, features_only=True, out_indices=(2, 3)
        ).to(self.device).eval()
        self.tr = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    @torch.no_grad()
    def raw(self, img) -> torch.Tensor:
        """PIL image -> (P, C) unnormalised patch features on self.device."""
        x = self.tr(img).unsqueeze(0).to(self.device)
        f2, f3 = self.model(x)
        f2 = F.interpolate(f2, scale_factor=0.5, mode="bilinear")
        return torch.cat([f2.flatten(2), f3.flatten(2)], dim=1).squeeze(0).permute(1, 0).contiguous()


def l2norm(f: torch.Tensor) -> torch.Tensor:
    return F.normalize(f, dim=-1)


@torch.no_grad()
def greedy_coreset(feats: torch.Tensor, frac: float, seed: int = 0, proj_dim: int = 128) -> torch.Tensor:
    """Greedy k-center coreset on a random projection; returns selected rows of feats."""
    n = feats.shape[0]
    m = max(1, int(frac * n))
    if m >= n:
        return feats
    g = torch.Generator(device="cpu").manual_seed(seed)
    proj = torch.randn(feats.shape[1], proj_dim, generator=g).to(feats.device) / proj_dim**0.5
    z = feats @ proj
    start = int(torch.randint(0, n, (1,), generator=g))
    sel = torch.empty(m, dtype=torch.long, device=feats.device)
    sel[0] = start
    mind = ((z - z[start]) ** 2).sum(1)
    for i in range(1, m):
        j = int(torch.argmax(mind))
        sel[i] = j
        mind = torch.minimum(mind, ((z - z[j]) ** 2).sum(1))
    return feats[sel].contiguous()


def build_bank(raw_feats: list[torch.Tensor], frac: float = 0.25, seed: int = 0) -> torch.Tensor:
    """Bank from per-image raw features (list of (P, C)); L2-normalised then coreset-subsampled."""
    allf = l2norm(torch.cat(raw_feats))
    return greedy_coreset(allf, frac, seed)


@torch.no_grad()
def image_scores(raw: torch.Tensor, bank: torch.Tensor) -> np.ndarray:
    """raw: (N, P, C) *already aligned* features (normalised here); returns (N,) image scores."""
    out = np.empty(raw.shape[0], dtype=np.float64)
    for i in range(raw.shape[0]):
        q = l2norm(raw[i])
        d = (2.0 - 2.0 * q @ bank.T).min(dim=1).values
        out[i] = float(d.max())
    return out
