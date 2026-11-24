"""Unit tests for src.gaussian_obstruction (Proposition 2.8, Section 2.2)."""

from __future__ import annotations

import numpy as np

from src import config as cfg
from src.gaussian_obstruction import (
    critical_exponent,
    invert_smoothed_empirical_cdf,
    population_quantile,
    smoothed_empirical_cdf,
    tau_squared,
    wasserstein_p_quantile_coupling,
)


def test_critical_exponent_sign_matches_configured_regimes():
    zeta_divergent = critical_exponent(cfg.EXP3_P_DIVERGENT, cfg.EXP3_S, cfg.EXP3_SIGMA_DIVERGENT)
    zeta_well_behaved = critical_exponent(cfg.EXP3_P_DIVERGENT, cfg.EXP3_S, cfg.EXP3_SIGMA_WELLBEHAVED)
    assert zeta_divergent > 0.0
    assert zeta_well_behaved <= 0.0


def test_critical_exponent_boundary_at_sigma_sq_equal_to_p_minus_2_s_sq():
    s = 1.0
    p = 4.0
    sigma_boundary = np.sqrt((p - 2) * s**2)
    zeta = critical_exponent(p, s, sigma_boundary)
    assert abs(zeta) < 1e-10


def test_tau_squared_and_population_quantile():
    s, sigma = 1.0, 0.5
    assert abs(tau_squared(s, sigma) - (s**2 + sigma**2)) < 1e-12
    assert abs(population_quantile(np.array(0.5), s, sigma)) < 1e-10
    q_low = population_quantile(np.array(0.1), s, sigma)
    q_high = population_quantile(np.array(0.9), s, sigma)
    assert q_low < 0.0 < q_high


def test_smoothed_empirical_cdf_is_monotone():
    rng = np.random.default_rng(0)
    samples = rng.normal(0.0, 1.0, size=50)
    x = np.linspace(-6, 6, 100)
    cdf = smoothed_empirical_cdf(x, samples, sigma=0.5)
    assert np.all(np.diff(cdf) >= -1e-12)
    assert cdf[0] < 1e-3 and cdf[-1] > 1 - 1e-3


def test_invert_smoothed_empirical_cdf_round_trip():
    rng = np.random.default_rng(1)
    samples = rng.normal(0.0, 1.0, size=50)
    sigma = 0.5
    u = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    x = invert_smoothed_empirical_cdf(u, samples, sigma, grid_points=4000)
    back = smoothed_empirical_cdf(x, samples, sigma)
    assert np.max(np.abs(back - u)) < 1e-3


def test_wasserstein_p_quantile_coupling_is_nonnegative_and_shrinks_with_n():
    s, sigma, p = 1.0, 2.0, 4.0
    rng = np.random.default_rng(42)
    w_small = []
    w_large = []
    for _ in range(40):
        samples_small = rng.normal(0.0, s, size=50)
        samples_large = rng.normal(0.0, s, size=2000)
        w_small.append(
            wasserstein_p_quantile_coupling(samples_small, s, sigma, p, n_nodes=100, eps_trunc=1e-8, grid_points=3000)
        )
        w_large.append(
            wasserstein_p_quantile_coupling(samples_large, s, sigma, p, n_nodes=100, eps_trunc=1e-8, grid_points=3000)
        )
    assert np.all(np.array(w_small) >= 0.0)
    assert np.all(np.array(w_large) >= 0.0)
    # In the well-behaved regime (sigma^2 >= (p-2)*s^2), W_p shrinks with n.
    assert np.median(w_large) < np.median(w_small)
