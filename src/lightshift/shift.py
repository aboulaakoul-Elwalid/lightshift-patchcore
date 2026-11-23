"""Controlled lighting-shift generator.

Families (from the sujet, seeded and invertible where possible):
gamma, brightness, contrast, colour temperature, directional shading.
Severity ladder: the sujet asks 3-5 severities; we implement an integer
level in {1..5} mapped per family to fixed, pre-registered parameter ranges
(range constants only; ranges registered in docs/hypotheses.md sibling note).
"""

from __future__ import annotations

import numpy as np

SHIFT_FAMILIES = ("gamma", "brightness", "contrast", "color_temp", "shading")

# Pre-registered parameters (level-indexed from 1). Fixed, do not tune
# retroactively; new ranges = new registration entry.
GAMMA_VALUES = {1: 0.85, 2: 0.75, 3: 0.6, 4: 0.45, 5: 0.3}  # <1 BRIGHTENS (x**g with g<1 lifts midtones)
BRIGHTNESS_DELS = {1: -12, 2: -22, 3: -35, 4: -48, 5: -62}  # 0-255 units
CONTRAST_SCALES = {1: 0.85, 2: 0.72, 3: 0.58, 4: 0.45, 5: 0.35}
COLOR_TEMP_GAINS = {  # (gain_r, gain_b); warm→cool shift, linear in channels
    1: (1.06, 0.95),
    2: (1.12, 0.9),
    3: (1.2, 0.82),
    4: (1.3, 0.72),
    5: (1.42, 0.62),
}
SHADING_ANGLES = {1: 0.0, 2: 20.0, 3: 35.0, 4: 50.0, 5: 65.0}
SHADING_DEPTH = {1: 0.25, 2: 0.4, 3: 0.55, 4: 0.7, 5: 0.85}


def _to_numpy(img):
    from PIL import Image

    if isinstance(img, Image.Image):
        return np.asarray(img, dtype=np.float32)
    return np.asarray(img, dtype=np.float32)


def apply_gamma(img, severity: int):
    x = _to_numpy(img) / 255.0
    g = GAMMA_VALUES[severity]
    return np.clip(np.power(x, g), 0.0, 1.0) * 255.0


def apply_brightness(img, severity: int):
    x = _to_numpy(img)
    return np.clip(x + BRIGHTNESS_DELS[severity], 0.0, 255.0)


def apply_contrast(img, severity: int):
    x = _to_numpy(img)
    s = CONTRAST_SCALES[severity]
    return np.clip((x - 128.0) * s + 128.0, 0.0, 255.0)


def apply_color_temp(img, severity: int):
    x = _to_numpy(img)
    if x.shape[-1] < 3:
        return x
    g = COLOR_TEMP_GAINS[severity]
    out = x.copy()
    out[..., 0] *= g[0]
    out[..., 2] *= g[1]
    return np.clip(out, 0.0, 255.0)


def apply_shading(img, severity: int, angle: float | None = None, seed: int = 0):
    """Additive directional shading gradient (one-sided)."""
    x = _to_numpy(img)
    h, w = x.shape[:2]
    rng = np.random.default_rng(seed)
    a = SHADING_ANGLES[severity] if angle is None else angle
    radial = np.radians(a)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = (xx / max(w - 1, 1) - 0.5) * np.cos(radial) + (yy / max(h - 1, 1) - 0.5) * np.sin(radial)
    # map to [-0.5, 0.5]; depth scales amplitude; smooth via tanh to avoid a hard edge
    grad = np.tanh(2.2 * (u - rng.uniform(0.0, 0.6)))  # random ramp placement
    delta = (grad - grad.min()) / (grad.max() - grad.min() + 1e-9)
    delta = (delta - 0.5) * 2.0  # [-1, 1]
    delta = delta[..., None] * (255.0 * SHADING_DEPTH[severity] * 0.5)
    return np.clip(x + delta, 0.0, 255.0)


APPLY = {
    "gamma": apply_gamma,
    "brightness": apply_brightness,
    "contrast": apply_contrast,
    "color_temp": apply_color_temp,
}


def apply_shift(img, family: str, severity: int, *, rng_seed: int = 0):
    if family not in SHIFT_FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    if family == "shading":
        return apply_shading(img, severity, seed=rng_seed)
    return APPLY[family](img, severity)


def shifted_pil(img, family: str, severity: int, *, rng_seed: int = 0):
    """PIL in -> PIL out (uint8), convenience wrapper around apply_shift."""
    from PIL import Image

    return Image.fromarray(apply_shift(img, family, severity, rng_seed=rng_seed).astype(np.uint8))
