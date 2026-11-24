"""Unit tests for src.bootstrap (full-path multinomial bootstrap, Section 3.3)."""

from __future__ import annotations

import numpy as np

from src.bootstrap import bootstrap_resample, order_statistic_quantile


def test_bootstrap_resample_shape_and_simplex():
    rng = np.random.default_rng(0)
    w = np.array([0.1, 0.2, 0.3, 0.4])
    out = bootstrap_resample(w, n=100, b_resamples=50, rng=rng)
    assert out.shape == (50, 4)
    assert np.allclose(np.sum(out, axis=1), 1.0)
    assert np.all(out >= 0.0)


def test_bootstrap_resample_mean_close_to_empirical_weights():
    rng = np.random.default_rng(1)
    w = np.array([0.1, 0.2, 0.3, 0.4])
    out = bootstrap_resample(w, n=2000, b_resamples=5000, rng=rng)
    assert np.max(np.abs(np.mean(out, axis=0) - w)) < 0.02


def test_order_statistic_quantile_is_the_ceil_level_b_th_order_statistic():
    values = np.arange(1, 250, dtype=float)  # 1..249, mimics B=249 bootstrap draws
    q = order_statistic_quantile(values, 0.95)
    k = int(np.ceil(0.95 * 249))
    assert k == 237
    assert q == float(np.sort(values)[k - 1])


def test_order_statistic_quantile_edge_cases():
    values = np.array([5.0])
    assert order_statistic_quantile(values, 0.95) == 5.0
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert order_statistic_quantile(values, 1.0) == 4.0
