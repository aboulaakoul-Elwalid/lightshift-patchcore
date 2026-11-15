"""B3 conformal threshold, with the corrected order statistics.

The sujet says "k-th largest" — that is wrong and rejects ~90% of good parts
at n=50, alpha=0.1. The operative rule is:

    k = ceil((n + 1) * (1 - alpha))
    threshold = k-th SMALLEST of the n calibration residuals
    k fails -> fall back to the maximum (conservative split-conformal edge)

Realized FPR of this construction is distribution-free ~ mean of
Beta(n + 1 - k, k).
"""

from __future__ import annotations

import math

import numpy as np


def conformal_k(n: int, alpha: float) -> int:
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    k = math.ceil((n + 1) * (1.0 - alpha))
    return min(k, n)


def conformal_threshold(residuals: np.ndarray, alpha: float) -> float:
    r = np.asarray(residuals, dtype=np.float64).ravel()
    n = r.size
    k = conformal_k(n, alpha)
    return float(np.sort(r)[k - 1])


def realized_fpr_distribution(n: int, alpha: float):
    """Parameters of the Beta small-sample FPR law: fpr ~ Beta(n+1-k, k)."""
    k = conformal_k(n, alpha)
    return float(n + 1 - k), float(k)


def fpr_tolerance_band(n: int, alpha: float, upper: float = 0.99):
    from scipy.stats import beta

    a, b = realized_fpr_distribution(n, alpha)
    return float(beta.ppf(upper, a, b))
