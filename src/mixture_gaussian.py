"""Gaussian-mixture CDFs and their inverse, used for the 4-point CRR model.

Implements the bracketed Newton inversion of Appendix A.3: "Each conditional
CDF is a mixture of at most three Gaussian CDFs, which we invert by
bracketed Newton iteration to CDF residual 1e-13 or bracket width 1e-12,
evaluating (A.1) by tensor-product Gauss-Legendre quadrature with 48 nodes
per coordinate."

All operations below are vectorized with numpy: a whole quadrature grid is
inverted in a handful of array-valued Newton steps instead of a Python loop
over nodes, which keeps the Section 3.3 experiment tractable.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def gauss_legendre_unit(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on (0, 1), summing to 1.

    Numpy's own ``numpy.polynomial.legendre.leggauss`` provides the nodes and
    weights on [-1, 1]; they are affinely rescaled to (0, 1) here as specified
    in Appendix A.3 ("tensor-product Gauss-Legendre quadrature").
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    unit_nodes = 0.5 * (nodes + 1.0)
    unit_weights = 0.5 * weights
    return unit_nodes, unit_weights


def gauss_legendre_interval(n_nodes: int, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on the finite interval [lo, hi]."""
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    scale = 0.5 * (hi - lo)
    shift = 0.5 * (hi + lo)
    return scale * nodes + shift, scale * weights


def mixture_cdf(x: np.ndarray, weights: np.ndarray, means: np.ndarray, sigma: float) -> np.ndarray:
    """CDF of sum_k weights[k] * N(means[k], sigma^2), broadcasting over x.

    ``weights`` and ``means`` carry their mixture axis on the last axis so
    that ``x`` of shape (...,) broadcasts against components of shape
    (..., K) once expanded with ``x[..., None]``.
    """
    z = (x[..., None] - means) / sigma
    return np.sum(weights * norm.cdf(z), axis=-1)


def mixture_pdf(x: np.ndarray, weights: np.ndarray, means: np.ndarray, sigma: float) -> np.ndarray:
    """Density of sum_k weights[k] * N(means[k], sigma^2), broadcasting over x."""
    z = (x[..., None] - means) / sigma
    return np.sum(weights * norm.pdf(z) / sigma, axis=-1)


def invert_mixture_cdf(
    u: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    sigma: float,
    cdf_tol: float,
    bracket_tol: float,
    max_iter: int,
) -> np.ndarray:
    """Bracketed Newton inversion of a Gaussian-mixture CDF (Appendix A.3).

    ``u`` has shape (...,) and broadcasts against the mixture axis of
    ``weights``/``means`` of shape (..., K); the result has the broadcast
    shape of ``u`` against the leading axes of ``weights``.
    """
    u = np.asarray(u, dtype=float)
    weights = np.asarray(weights, dtype=float)
    means = np.asarray(means, dtype=float)

    lead_shape = np.broadcast_shapes(u.shape, weights.shape[:-1])
    u_b = np.broadcast_to(u, lead_shape).astype(float)
    w_b = np.broadcast_to(weights, lead_shape + weights.shape[-1:])
    m_b = np.broadcast_to(means, lead_shape + means.shape[-1:])

    span = 10.0 * sigma + np.max(np.abs(means)) + 1.0
    lo = np.full(lead_shape, -span)
    hi = np.full(lead_shape, span)
    # Initial guess: the quantile of a single Gaussian centered at the
    # weighted mean, which is close enough to converge from either side.
    mean_guess = np.sum(w_b * m_b, axis=-1)
    x = mean_guess + sigma * norm.ppf(np.clip(u_b, 1e-12, 1 - 1e-12))
    x = np.clip(x, lo, hi)

    for _ in range(max_iter):
        f = mixture_cdf(x, w_b, m_b, sigma) - u_b
        lo = np.where(f < 0, x, lo)
        hi = np.where(f > 0, x, hi)
        done = (np.abs(f) < cdf_tol) | ((hi - lo) < bracket_tol)
        if np.all(done):
            break
        dens = mixture_pdf(x, w_b, m_b, sigma)
        step = np.where(dens > 1e-300, f / np.maximum(dens, 1e-300), 0.0)
        x_newton = x - step
        # Fall back to bisection whenever Newton leaves the bracket.
        x_bisect = 0.5 * (lo + hi)
        in_bracket = (x_newton > lo) & (x_newton < hi)
        x = np.where(done, x, np.where(in_bracket, x_newton, x_bisect))
    return x


def merge_components(weights: np.ndarray, means: np.ndarray, atol: float = 1e-12):
    """Merge mixture components that share a location, along the last axis.

    ``weights``/``means`` have identical shape (..., K_in); returns arrays of
    shape (..., K_out) with K_out <= K_in, padded with zero-weight dummy
    components at an unused location so all rows keep the same K_out.
    """
    distinct = np.unique(np.round(means / atol) * atol)
    out_w = np.zeros(weights.shape[:-1] + (len(distinct),))
    out_m = np.broadcast_to(distinct, weights.shape[:-1] + (len(distinct),)).copy()
    for k, loc in enumerate(distinct):
        mask = np.isclose(means, loc, atol=atol)
        out_w[..., k] = np.sum(np.where(mask, weights, 0.0), axis=-1)
    return out_w, out_m
