"""Root-n tightness failure for smooth Wasserstein distances (Proposition 2.8).

For d = T = 1 the adapted Wasserstein distance AW_p reduces to the ordinary
Wasserstein distance W_p on R (Section 2, Proposition 2.8). With
mu = N(0, s^2), Gaussian smoothing bandwidth sigma, and P = mu * gamma_sigma =
N(0, tau^2) with tau^2 = s^2 + sigma^2, Proposition 2.8 shows that
sqrt(n) * W_p(P_n, P) diverges to +infinity in probability whenever
sigma^2 < (p-2) * s^2, through the critical exponent

    zeta_p = ((p-2)*s^2 - sigma^2) / (2*(s^2+sigma^2)*(2*s^2+sigma^2))  (Sec. 2.2)

which is strictly positive exactly in the divergent regime.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from src.mixture_gaussian import gauss_legendre_interval


def tau_squared(s: float, sigma: float) -> float:
    """tau^2 = s^2 + sigma^2, variance of the smoothed population law P."""
    return s**2 + sigma**2


def critical_exponent(p: float, s: float, sigma: float) -> float:
    """zeta_p = ((p-2)*s^2 - sigma^2) / (2*(s^2+sigma^2)*(2*s^2+sigma^2)) (Sec. 2.2).

    zeta_p > 0 iff sigma^2 < (p-2)*s^2: the obstruction condition of
    Proposition 2.8 under which sqrt(n)*W_p(P_n, P) is not tight.
    """
    return ((p - 2) * s**2 - sigma**2) / (2 * (s**2 + sigma**2) * (2 * s**2 + sigma**2))


def population_quantile(u: np.ndarray, s: float, sigma: float) -> np.ndarray:
    """F^{-1}(u) = tau * Phi^{-1}(u), the quantile of P = N(0, tau^2)."""
    tau = np.sqrt(tau_squared(s, sigma))
    return tau * norm.ppf(u)


_SAMPLE_CHUNK = 2000  # caps the temporary (len(x), chunk) array at a few hundred MB


def smoothed_empirical_cdf(x: np.ndarray, samples: np.ndarray, sigma: float) -> np.ndarray:
    """F_n(x) = (1/n) * sum_i Phi((x - A^{(i)})/sigma).

    ``samples`` has shape (n,); ``x`` may be any broadcastable array. The sum
    over ``samples`` is accumulated in chunks so that the temporary
    (len(x), chunk) array stays small regardless of n, avoiding the O(n *
    len(x)) memory blow-up of a single broadcast for large n.
    """
    x = np.asarray(x)
    n = samples.shape[0]
    total = np.zeros(x.shape, dtype=np.float64)
    for start in range(0, n, _SAMPLE_CHUNK):
        block = samples[start:start + _SAMPLE_CHUNK]
        total += np.sum(norm.cdf((x[..., None] - block[None, :]) / sigma), axis=-1)
    return total / n


def invert_smoothed_empirical_cdf(
    u: np.ndarray, samples: np.ndarray, sigma: float, grid_points: int, x_pad: float = 12.0
) -> np.ndarray:
    """F_n^{-1}(u) by linear interpolation of F_n on a fine grid.

    The grid spans [-x_pad*scale, x_pad*scale] where ``scale`` is the sample
    standard deviation plus sigma, wide enough to cover the extreme quantiles
    used by the quantile-coupling integral.
    """
    scale = float(np.std(samples)) + sigma
    grid = np.linspace(-x_pad * scale, x_pad * scale, grid_points)
    cdf_grid = smoothed_empirical_cdf(grid, samples, sigma)
    # F_n is strictly increasing (a mixture of continuous Gaussian CDFs); clip
    # to the grid's achieved CDF range before interpolating to avoid
    # extrapolation at the extreme quantiles.
    cdf_grid = np.clip(cdf_grid, 1e-300, 1 - 1e-16)
    cdf_grid = np.maximum.accumulate(cdf_grid)
    u_clipped = np.clip(u, cdf_grid[0], cdf_grid[-1])
    return np.interp(u_clipped, cdf_grid, grid)


def wasserstein_p_quantile_coupling(
    samples: np.ndarray, s: float, sigma: float, p: float, n_nodes: int, eps_trunc: float, grid_points: int
) -> float:
    """W_p(P_n, P) via W_p(P_n,P)^p = integral_0^1 |F_n^{-1}(u)-F^{-1}(u)|^p du.

    Evaluated by Gauss-Legendre quadrature on [eps_trunc, 1-eps_trunc]
    (Section 4.3).
    """
    nodes, weights = gauss_legendre_interval(n_nodes, eps_trunc, 1.0 - eps_trunc)
    f_pop = population_quantile(nodes, s, sigma)
    f_emp = invert_smoothed_empirical_cdf(nodes, samples, sigma, grid_points)
    integrand = np.abs(f_emp - f_pop) ** p
    value = float(np.sum(weights * integrand))
    return value ** (1.0 / p)
