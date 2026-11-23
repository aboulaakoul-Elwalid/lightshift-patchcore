"""B2 — feature-statistic alignment (AdaBN-style), deployable form.

All statistics are per-channel over patches. The *reference* moments come from the
images the memory bank was built from (reference lighting). The *target* moments come
from k normal images captured under the new lighting. No clean counterpart of the
target images is ever needed (the pilot's paired "oracle" estimate did need it).

Modes
  none     identity (vanilla)
  mean     f - (mu_t - mu_r)                      (B2a)
  meanstd  (f - mu_t) / sd_t * sd_r + mu_r        (B2b, corrected: ratio of stds)

Note: the pilot's B2b multiplied by the *target std* instead of sd_r / sd_t, which
is not an AdaBN-style rescale; that is why it looked uniformly harmful there.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

EPS = 1e-6


@dataclass
class Moments:
    mean: torch.Tensor  # (C,)
    var: torch.Tensor  # (C,)


def image_moments(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(P, C) -> per-channel mean and mean-of-squares for one image."""
    return raw.mean(0), (raw**2).mean(0)


def pool(per_image: list[tuple[torch.Tensor, torch.Tensor]]) -> Moments:
    """Pool per-image (mean, mean-of-squares) -> dataset moments (equal patch count per image)."""
    m = torch.stack([a for a, _ in per_image]).mean(0)
    s = torch.stack([b for _, b in per_image]).mean(0)
    return Moments(m, torch.clamp(s - m**2, min=0.0))


def align(raw: torch.Tensor, mode: str, target: Moments | None = None, ref: Moments | None = None) -> torch.Tensor:
    """raw: (..., C) unnormalised features -> aligned features (still unnormalised)."""
    if mode == "none":
        return raw
    if target is None or ref is None:
        raise ValueError(f"mode {mode!r} needs target and ref moments")
    if mode == "mean":
        return raw - (target.mean - ref.mean)
    if mode == "meanstd":
        sd_t = torch.sqrt(target.var + EPS)
        sd_r = torch.sqrt(ref.var + EPS)
        return (raw - target.mean) / sd_t * sd_r + ref.mean
    raise ValueError(f"unknown B2 mode {mode!r}")
