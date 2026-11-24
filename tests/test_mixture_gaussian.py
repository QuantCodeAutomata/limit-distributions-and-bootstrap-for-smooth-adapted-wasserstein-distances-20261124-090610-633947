"""Unit tests for src.mixture_gaussian (Gauss-Legendre quadrature and CDF inversion)."""

from __future__ import annotations

import numpy as np

from src.mixture_gaussian import (
    gauss_legendre_interval,
    gauss_legendre_unit,
    invert_mixture_cdf,
    merge_components,
    mixture_cdf,
    mixture_pdf,
)


def test_gauss_legendre_unit_nodes_in_range_and_weights_sum_to_one():
    nodes, weights = gauss_legendre_unit(24)
    assert np.all(nodes > 0.0) and np.all(nodes < 1.0)
    assert abs(np.sum(weights) - 1.0) < 1e-12


def test_gauss_legendre_interval_integrates_constant_exactly():
    nodes, weights = gauss_legendre_interval(16, -2.0, 3.0)
    assert abs(np.sum(weights) - 5.0) < 1e-10
    assert np.all((nodes >= -2.0) & (nodes <= 3.0))


def test_mixture_cdf_monotone_and_bounded():
    weights = np.array([0.3, 0.7])
    means = np.array([-1.0, 1.0])
    x = np.linspace(-10, 10, 200)
    cdf = mixture_cdf(x, weights, means, sigma=0.5)
    assert np.all(np.diff(cdf) >= -1e-12)
    assert cdf[0] < 1e-6 and cdf[-1] > 1 - 1e-6


def test_mixture_pdf_matches_cdf_derivative():
    weights = np.array([0.5, 0.5])
    means = np.array([0.0, 2.0])
    sigma = 0.4
    x0 = 1.3
    h = 1e-5
    cdf_deriv = (
        mixture_cdf(np.array([x0 + h]), weights, means, sigma)[0]
        - mixture_cdf(np.array([x0 - h]), weights, means, sigma)[0]
    ) / (2 * h)
    pdf_val = mixture_pdf(np.array([x0]), weights, means, sigma)[0]
    assert abs(cdf_deriv - pdf_val) < 1e-4


def test_invert_mixture_cdf_round_trip():
    weights = np.array([0.2, 0.3, 0.5])
    means = np.array([-2.0, 0.0, 3.0])
    sigma = 0.6
    u = np.linspace(0.01, 0.99, 25)
    w_b = np.broadcast_to(weights, (len(u), 3))
    m_b = np.broadcast_to(means, (len(u), 3))
    x = invert_mixture_cdf(u, w_b, m_b, sigma, cdf_tol=1e-13, bracket_tol=1e-12, max_iter=80)
    back = mixture_cdf(x, w_b, m_b, sigma)
    assert np.max(np.abs(back - u)) < 1e-8


def test_invert_mixture_cdf_single_gaussian_matches_closed_form():
    from scipy.stats import norm

    weights = np.array([1.0])
    means = np.array([2.0])
    sigma = 1.5
    u = np.array([0.1, 0.5, 0.9])
    w_b = np.broadcast_to(weights, (3, 1))
    m_b = np.broadcast_to(means, (3, 1))
    x = invert_mixture_cdf(u, w_b, m_b, sigma, cdf_tol=1e-13, bracket_tol=1e-12, max_iter=80)
    expected = means[0] + sigma * norm.ppf(u)
    assert np.max(np.abs(x - expected)) < 1e-6


def test_merge_components_combines_shared_locations():
    weights = np.array([0.2, 0.3, 0.5])
    means = np.array([1.0, 1.0, 2.0])
    w_out, m_out = merge_components(weights, means)
    assert w_out.shape[-1] == 2
    total_at_one = float(w_out[np.isclose(m_out, 1.0)].sum())
    assert abs(total_at_one - 0.5) < 1e-10
    assert abs(float(np.sum(w_out)) - 1.0) < 1e-10
