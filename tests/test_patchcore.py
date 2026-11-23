import math

import torch

from lightshift.patchcore import greedy_coreset, image_scores


def _unit(angles_deg):
    a = torch.deg2rad(torch.tensor(angles_deg, dtype=torch.float32))
    return torch.stack([torch.cos(a), torch.sin(a)], dim=1)


def test_worked_example_scores_match_report():
    bank = _unit([0.0, 90.0, 180.0])
    raw = _unit([10.0, 80.0, 135.0]).unsqueeze(0)  # (1 image, 3 patches, 2 dims)
    s = image_scores(raw, bank)[0]
    assert math.isclose(s, 2 - 2 * math.cos(math.radians(45)), rel_tol=1e-5)  # 0.5858


def test_scale_invariance_of_l2_normalisation():
    bank = _unit([0.0, 90.0])
    raw = _unit([20.0, 70.0]).unsqueeze(0)
    assert math.isclose(image_scores(raw, bank)[0], image_scores(raw * 7.0, bank)[0], rel_tol=1e-5)


def test_greedy_coreset_covers_both_clusters():
    pts = torch.tensor([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]])
    c = greedy_coreset(pts, frac=0.5, seed=0, proj_dim=1)
    assert c.shape[0] == 3
    radius = torch.cdist(pts, c).min(dim=1).values.max()
    assert radius <= 2.0 + 1e-5  # random-start greedy still covers both clusters within 2


def test_coreset_returns_all_when_frac_one():
    pts = torch.randn(20, 5)
    assert greedy_coreset(pts, 1.0).shape == pts.shape
