"""Generic Rosenblatt-quantile computation of AW_p for finite-support 2-D laws.

Shared by the CRR Bermudan model (Section 3.3) and the four-point product
law (Appendix A.1): both are Gaussian-smoothed mixtures on a fixed finite set
of (first-coordinate, second-coordinate) locations where the common-uniform
Rosenblatt coupling is optimal for AW_p because the smoothed second-period
conditionals are stochastically increasing in the first coordinate
(Appendix A.3). The adapted Wasserstein distance between two weight vectors
on the same support is then computed by one-dimensional quantile quadrature,
formula (A.1):

    AW_p(P_w, P_w~)^p = int_{(0,1)^2} sum_t |R_{w,t}(u) - R_{w~,t}(u)|^p du.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from src.mixture_gaussian import gauss_legendre_unit, invert_mixture_cdf, merge_components


def aw_p_general(
    c: np.ndarray,
    b: np.ndarray,
    weights_w: np.ndarray,
    weights_wtilde: np.ndarray,
    sigma: float,
    p_values: tuple[int, ...],
    n_nodes: int,
    newton_cdf_tol: float,
    newton_bracket_tol: float,
    newton_max_iter: int,
) -> dict[int, float]:
    """AW_p(P_w, P_w~) for every p in p_values, on the K-point support (c_i, b_i).

    ``c`` and ``b`` (shape (K,)) are the fixed first/second-coordinate
    locations shared by both weight vectors; only the probability weights
    differ. Returns {p: AW_p value} (not AW_p^p).
    """
    u1_nodes, u1_w = gauss_legendre_unit(n_nodes)
    u2_nodes, u2_w = gauss_legendre_unit(n_nodes)

    def first_coord_quantile(weights: np.ndarray) -> np.ndarray:
        w2, m2 = merge_components(weights, c)
        w2 = np.broadcast_to(w2, (n_nodes,) + w2.shape)
        m2 = np.broadcast_to(m2, (n_nodes,) + m2.shape)
        return invert_mixture_cdf(
            u1_nodes, w2, m2, sigma, newton_cdf_tol, newton_bracket_tol, newton_max_iter
        )

    def second_coord_quantile(weights: np.ndarray, x1: np.ndarray) -> np.ndarray:
        z = (x1[:, None] - c[None, :]) / sigma
        post = weights[None, :] * norm.pdf(z)
        post = post / np.sum(post, axis=-1, keepdims=True)
        w3, m3 = merge_components(post, np.broadcast_to(b, post.shape))
        w3 = np.broadcast_to(w3[:, None, :], (n_nodes, n_nodes, w3.shape[-1]))
        m3 = np.broadcast_to(m3[:, None, :], (n_nodes, n_nodes, m3.shape[-1]))
        u2_grid = np.broadcast_to(u2_nodes[None, :], (n_nodes, n_nodes))
        return invert_mixture_cdf(
            u2_grid, w3, m3, sigma, newton_cdf_tol, newton_bracket_tol, newton_max_iter
        )

    x1_w = first_coord_quantile(weights_w)
    x1_wt = first_coord_quantile(weights_wtilde)
    z_w = second_coord_quantile(weights_w, x1_w)
    z_wt = second_coord_quantile(weights_wtilde, x1_wt)

    cost1 = np.abs(x1_w - x1_wt)
    cost2 = np.abs(z_w - z_wt)
    outer_w = np.outer(u1_w, u2_w)

    result = {}
    for p in p_values:
        total_p = np.sum(outer_w * (cost1[:, None] ** p + cost2**p))
        result[p] = float(total_p ** (1.0 / p))
    return result
