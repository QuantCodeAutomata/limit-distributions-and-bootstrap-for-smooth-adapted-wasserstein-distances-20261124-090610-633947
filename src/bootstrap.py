"""Full-path multinomial bootstrap for the four-path CRR model (Section 3.3).

"Each dataset consists of multinomial path counts drawn with probabilities w
..., and each bootstrap draw resamples multinomial counts from the empirical
weights." Resampling whole paths (not marginal coordinates) is what retains
the cross-date dependence that the adapted Wasserstein distance is sensitive
to (Section 3.3, implementation notes).
"""

from __future__ import annotations

import numpy as np


def bootstrap_resample(empirical_weights: np.ndarray, n: int, b_resamples: int, rng: np.random.Generator) -> np.ndarray:
    """``b_resamples`` multinomial(n, empirical_weights) bootstrap weight vectors.

    Returns an array of shape (b_resamples, 4).
    """
    counts = rng.multinomial(n, empirical_weights, size=b_resamples)
    return counts / n


def order_statistic_quantile(values: np.ndarray, level: float) -> float:
    """The ceil(level * B)-th order statistic of ``values`` (Section 3.3).

    For B = 249 and level = 0.95, this is the 237th order statistic, matching
    "q_hat_{n,0.95}* is taken as the ceil(0.95 x 249) = 237th order statistic
    of the scaled bootstrap distances."
    """
    b = len(values)
    k = int(np.ceil(level * b))
    k = min(max(k, 1), b)
    return float(np.sort(values)[k - 1])
