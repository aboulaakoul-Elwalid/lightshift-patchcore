import numpy as np
import pytest

from lightshift.b3 import conformal_k, conformal_threshold, fpr_tolerance_band, realized_fpr_distribution


def test_k_matches_corrected_formula():
    # the sujet's "k-th largest" typo would demand k from the other tail
    assert conformal_k(50, 0.1) == 46  # ceil(51*0.9)=46 -> 46th smallest
    assert conformal_k(9, 0.1) == 9    # ceil(10*0.9)=9
    assert conformal_k(1, 0.2) == 1    # clamp: ceil(2*0.8)=2 > n -> n


def test_threshold_is_kth_smallest_not_largest():
    r = np.arange(1, 11, dtype=float)  # 1..10, n=10, alpha=0.1
    # k = ceil(11*0.9) = 10 -> max value as kth smallest. With alpha=0.2: k=9.
    assert conformal_threshold(r, 0.2) == 9.0
    assert conformal_threshold(r, 0.1) == 10.0


def test_kth_smallest_rejects_about_alpha_fraction():
    # the correct rule (k-th smallest) rejects ~alpha of the good calibration scores;
    # the sujet's "k-th largest" would reject ~90%. n=50, alpha=0.1: k=46, 4/50 = 8% above threshold.
    r = np.arange(1, 51, dtype=float)
    threshold = conformal_threshold(r, 0.1)
    assert (r > threshold).mean() == pytest.approx(0.08, abs=0.02)


def test_beta_band_bounds_fpr():
    a, b = realized_fpr_distribution(20, 0.1)
    assert (a, b) == (2.0, 19.0)
    band = fpr_tolerance_band(20, 0.1)
    assert 0.1 <= band <= 0.35  # small-sample inflation, documented
