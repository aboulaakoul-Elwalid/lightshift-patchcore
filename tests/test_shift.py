import numpy as np
import pytest
from PIL import Image

from lightshift.shift import SHIFT_FAMILIES, apply_shift

base = np.tile(np.linspace(60, 200, 64, dtype=np.float32), (64, 1))[..., None] * np.ones((1, 1, 3))


def img(arr):
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def test_all_families_run_all_severities():
    im = img(base)
    for fam in SHIFT_FAMILIES:
        for sev in range(1, 6):
            out = apply_shift(im, fam, sev, rng_seed=0)
            arr = np.asarray(out, dtype=np.int32)
            assert arr.min() >= 0 and arr.max() <= 255
            assert arr.shape == base.shape


def test_severity_orders_monotonically_for_gamma():
    im = img(base)
    outs = [np.asarray(apply_shift(im, "gamma", s, rng_seed=0), dtype=np.float32) for s in range(1, 6)]
    means = [o.mean() for o in outs]
    assert means == sorted(means, reverse=False)  # darker with severity


def test_shift_is_deterministic_given_seed():
    im = img(base)
    a = np.asarray(apply_shift(im, "shading", 3, rng_seed=7))
    b = np.asarray(apply_shift(im, "shading", 3, rng_seed=7))
    assert (a == b).all()


def test_known_corner_intensities_stay_in_range():
    x = np.zeros((16, 16, 3), dtype=np.float32)
    x[:8, :8] = 255.0
    im = Image.fromarray(x.astype(np.uint8), "RGB")
    for fam in SHIFT_FAMILIES:
        for sev in range(1, 6):
            out = np.asarray(apply_shift(im, fam, sev, rng_seed=1))
            assert out.min() >= 0 and out.max() <= 255
