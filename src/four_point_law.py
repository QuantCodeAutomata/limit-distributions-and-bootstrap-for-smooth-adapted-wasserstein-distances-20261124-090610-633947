"""Four-point product law mu = Unif{-a,a} (x) Unif{-b,b} on R^2 (Appendix A.1).

Implements the closed-form building blocks of Appendix A.1: the smoothed
marginal density f_c, the (even, strictly negative) CDF-difference D_c, the
odd log-odds tangent s_c, the quadrature constants C_{c,p} and kappa_{c,k},
and the scalar
tangent energy Q_{P,p}(h) of formula (3.1) specialised to a perturbation
h = rho * [u*s_a(x) + v*s_b(z) + w*s_a(x)*s_b(z)].

By the multivariate CLT, sqrt(n)*(u_n, v_n, w_n) -> (Z1, Z2, Z3) ~ N(0, I_3)
where u_n, v_n, w_n are the empirical means of the Rademacher signs and their
product (Appendix A.1). Because Q_{P,p} is jointly p-homogeneous in (u, v, w),
evaluating Q_{P,p} at the rescaled triple (sqrt(n)*u_n, sqrt(n)*v_n,
sqrt(n)*w_n) is identical to n^{p/2} * Q_{P,p}(u_n, v_n, w_n), so no separate
rescaling step is needed once the inputs are scaled by sqrt(n).
"""

from __future__ import annotations

import numpy as np
from scipy import integrate
from scipy.stats import norm

from src.mixture_gaussian import gauss_legendre_interval


def f_c(x: np.ndarray, c: float, sigma: float) -> np.ndarray:
    """Smoothed density of Unif{-c, c} * N(0, sigma^2) (Appendix A.1)."""
    return 0.5 * (norm.pdf((x - c) / sigma) + norm.pdf((x + c) / sigma)) / sigma


def cap_d_c(x: np.ndarray, c: float, sigma: float) -> np.ndarray:
    """D_c(x) = (Phi_sigma(x-c) - Phi_sigma(x+c)) / 2 (Appendix A.1)."""
    return 0.5 * (norm.cdf((x - c) / sigma) - norm.cdf((x + c) / sigma))


def s_c(x: np.ndarray, c: float, sigma: float) -> np.ndarray:
    """s_c(x) = tanh(c*x/sigma^2) (Appendix A.1)."""
    return np.tanh(c * x / sigma**2)


def constant_c_cp(c: float, sigma: float, p: int) -> float:
    """C_{c,p} = integral |D_c(x)|^p / f_c(x)^{p-1} dx over R (Appendix A.1)."""

    def integrand(x: float) -> float:
        density = f_c(x, c, sigma)
        d_val = cap_d_c(x, c, sigma)
        if density <= 0.0 or d_val == 0.0:
            # Far tails: both D_c and f_c underflow to zero at the same
            # Gaussian rate; the ratio's contribution there is negligible.
            return 0.0
        # Written as |D_c/f_c|^{p-1} * |D_c| rather than |D_c|^p / f_c^{p-1}
        # to avoid separately underflowing f_c^{p-1} in the far tails.
        ratio = d_val / density
        return np.abs(ratio) ** (p - 1) * np.abs(d_val)

    value, _ = integrate.quad(integrand, -np.inf, np.inf, limit=400)
    return float(value)


def constant_kappa_ck(c: float, sigma: float, k: int) -> float:
    """kappa_{c,k} = integral s_c(x)^k * f_c(x) dx over R (Appendix A.1)."""

    def integrand(x: float) -> float:
        return s_c(x, c, sigma) ** k * f_c(x, c, sigma)

    value, _ = integrate.quad(integrand, -np.inf, np.inf, limit=400)
    return float(value)


class FourPointLawBasis:
    """Precomputed quadrature grid for the x-integral in Q_{P,p} (formula 3.1).

    Q_{P,p}(u, v, w) = C_{a,p}*|u|^p
        + C_{b,p} * integral |v + w*s_a(x)|^p * f_a(x) dx.
    The second integral is evaluated on a fixed Gauss-Legendre grid in x so
    that it can be applied to large batches of Monte Carlo (u, v, w) triples
    at once.
    """

    def __init__(self, a: float, b: float, sigma: float, p_values: tuple[int, ...], n_nodes: int, x_pad_sigma: float = 10.0):
        self.a, self.b, self.sigma = a, b, sigma
        self.p_values = p_values
        lo, hi = -(a + x_pad_sigma * sigma), a + x_pad_sigma * sigma
        self.x_nodes, self.x_weights = gauss_legendre_interval(n_nodes, lo, hi)
        self.s_a_nodes = s_c(self.x_nodes, a, sigma)
        self.f_a_nodes = f_c(self.x_nodes, a, sigma)
        self.c_ap = {p: constant_c_cp(a, sigma, p) for p in p_values}
        self.c_bp = {p: constant_c_cp(b, sigma, p) for p in p_values}
        self.kappa_a2 = constant_kappa_ck(a, sigma, 2)
        self.kappa_a4 = constant_kappa_ck(a, sigma, 4)

    def tangent_energy(self, u: np.ndarray, v: np.ndarray, w: np.ndarray, p: int) -> np.ndarray:
        """Q_{P,p}(u, v, w) for arrays of equal shape, by quadrature (formula 3.1)."""
        u, v, w = np.asarray(u, dtype=float), np.asarray(v, dtype=float), np.asarray(w, dtype=float)
        term1 = self.c_ap[p] * np.abs(u) ** p
        integrand = np.abs(v[..., None] + w[..., None] * self.s_a_nodes) ** p * self.f_a_nodes
        term2 = self.c_bp[p] * np.sum(integrand * self.x_weights, axis=-1)
        return term1 + term2

    def closed_form_p2(self, z1: np.ndarray, z2: np.ndarray, z3: np.ndarray) -> np.ndarray:
        """C_{a,2}*Z1^2 + C_{b,2}*Z2^2 + kappa_{a,2}*C_{b,2}*Z3^2 (Appendix A.1)."""
        return self.c_ap[2] * z1**2 + self.c_bp[2] * z2**2 + self.kappa_a2 * self.c_bp[2] * z3**2

    def closed_form_p4(self, z1: np.ndarray, z2: np.ndarray, z3: np.ndarray) -> np.ndarray:
        """S_4 = C_{a,4}*Z1^4 + C_{b,4}*(Z2^4 + 6*kappa_{a,2}*Z2^2*Z3^2 + kappa_{a,4}*Z3^4)."""
        return self.c_ap[4] * z1**4 + self.c_bp[4] * (
            z2**4 + 6.0 * self.kappa_a2 * z2**2 * z3**2 + self.kappa_a4 * z3**4
        )


def empirical_weights(eps: np.ndarray, zeta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """u_n, v_n, w_n from batches of Rademacher pairs of shape (b, n) (Appendix A.1)."""
    u_n = np.mean(eps, axis=-1)
    v_n = np.mean(zeta, axis=-1)
    w_n = np.mean(eps * zeta, axis=-1)
    return u_n, v_n, w_n
