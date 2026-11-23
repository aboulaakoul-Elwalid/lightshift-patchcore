import torch

from lightshift.b2 import Moments, align, image_moments, pool


def _moments(x):
    return pool([image_moments(i) for i in x])


def test_identity_when_no_shift():
    torch.manual_seed(0)
    x = torch.randn(10, 64, 8) * 2 + 1
    m = _moments(x)
    assert torch.allclose(align(x, "mean", m, m), x, atol=1e-5)
    assert torch.allclose(align(x, "meanstd", m, m), x, atol=1e-4)


def test_mean_alignment_removes_pure_offset():
    torch.manual_seed(0)
    ref = torch.randn(20, 64, 8)
    off = torch.linspace(-1, 1, 8)
    tgt = ref + off
    out = align(tgt, "mean", _moments(tgt), _moments(ref))
    assert torch.allclose(out.mean((0, 1)), ref.mean((0, 1)), atol=1e-5)


def test_meanstd_matches_reference_moments_on_affine_shift():
    torch.manual_seed(0)
    ref = torch.randn(20, 64, 8) * 1.5 + 0.3
    tgt = ref * 0.5 + 2.0  # contrast + brightness style affine drift
    out = align(tgt, "meanstd", _moments(tgt), _moments(ref))
    r, o = _moments(ref), _moments(out)
    assert torch.allclose(o.mean, r.mean, atol=1e-4)
    assert torch.allclose(o.var, r.var, rtol=1e-3, atol=1e-4)


def test_unknown_mode_and_missing_moments():
    x = torch.zeros(4, 2)
    for bad in (lambda: align(x, "nope"), lambda: align(x, "mean")):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")
