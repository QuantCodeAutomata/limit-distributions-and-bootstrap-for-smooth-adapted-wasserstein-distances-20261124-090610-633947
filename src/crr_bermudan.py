"""Two-date CRR binomial model, its Gaussian smoothing, AW_p, and V(.).

Implements Section 3.3 ("A semi-synthetic Bermudan experiment") and its
numerical details in Appendix A.3.

The latent law mu has four paths in log-relative coordinates
a_1 = (-a,-2a), a_2 = (-a,0), a_3 = (a,0), a_4 = (a,2a) (eq. 3.2). For any
weight vector w on these four paths, P_w = sum_i w_i N(a_i, sigma^2 I_2) is
its Gaussian smoothing. Because Gaussian smoothing makes the second-period
conditionals of P_w stochastically increasing in the first coordinate
(Appendix A.3), the common-uniform Rosenblatt coupling is optimal for AW_p,
so AW_p(P_w, P_w~) is computed by one-dimensional quantile quadrature (A.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import integrate
from scipy.stats import norm

from src.rosenblatt_awp import aw_p_general

# Fixed path geometry shared by every weight vector (eq. 3.2).
PATH_C = np.array([-1.0, -1.0, 1.0, 1.0])  # first-coordinate sign, times "a"
PATH_B = np.array([-2.0, 0.0, 0.0, 2.0])   # second-coordinate sign, times "a"


@dataclass(frozen=True)
class CRRModel:
    """Parameters and derived quantities of the two-date CRR model (Sec. 3.3)."""

    nu: float
    s0: float
    k_strike: float
    delta: float
    r: float
    sigma_mult: float

    @property
    def a(self) -> float:
        return self.nu * np.sqrt(self.delta)

    @property
    def sigma(self) -> float:
        return self.sigma_mult * self.a

    @property
    def pi_plus(self) -> float:
        a = self.a
        u, d = np.exp(a), np.exp(-a)
        return (np.exp(self.r * self.delta) - d) / (u - d)

    @property
    def weights_mu(self) -> np.ndarray:
        """Latent path probabilities w = ((1-pi)^2, (1-pi)pi, pi(1-pi), pi^2)."""
        pi = self.pi_plus
        return np.array([(1 - pi) ** 2, (1 - pi) * pi, pi * (1 - pi), pi**2])

    def path_means(self) -> np.ndarray:
        """The four (c_i, b_i) centres scaled by a, shape (4, 2)."""
        a = self.a
        return np.column_stack([PATH_C * a, PATH_B * a])


def aw_p_distance(
    model: CRRModel,
    weights_w: np.ndarray,
    weights_wtilde: np.ndarray,
    p_values: tuple[int, ...],
    n_nodes: int,
    newton_cdf_tol: float,
    newton_bracket_tol: float,
    newton_max_iter: int,
) -> dict[int, float]:
    """AW_p(P_w, P_wtilde) for every p in p_values, via quantile quadrature (A.1).

    Both weight vectors share the same fixed path geometry (eq. 3.2); only the
    probability weights differ (e.g. population vs. empirical vs. bootstrap).
    Returns {p: AW_p value} (not AW_p^p).
    """
    a = model.a
    return aw_p_general(
        PATH_C * a,
        PATH_B * a,
        weights_w,
        weights_wtilde,
        model.sigma,
        p_values,
        n_nodes,
        newton_cdf_tol,
        newton_bracket_tol,
        newton_max_iter,
    )


def _h_bermudan(model: CRRModel, b_i: np.ndarray) -> np.ndarray:
    """H_i = Gaussian expectation of (K - S0*e^{X2})_+ at second-period mean b_i."""
    k = np.log(model.k_strike / model.s0)
    sigma = model.sigma
    term1 = model.k_strike * norm.cdf((k - b_i) / sigma)
    term2 = model.s0 * np.exp(b_i + 0.5 * sigma**2) * norm.cdf((k - b_i - sigma**2) / sigma)
    return term1 - term2


def smoothed_stopping_value(model: CRRModel, weights: np.ndarray) -> float:
    """V(P_w) by adaptive quadrature of the continuation/exercise envelope (Sec. 3.3)."""
    a = model.a
    sigma = model.sigma
    c = PATH_C * a
    b = PATH_B * a
    h_i = _h_bermudan(model, b)

    def integrand(x: float) -> float:
        rho1 = np.sum(weights * norm.pdf((x - c) / sigma) / sigma)
        exercise = np.exp(-model.r * model.delta) * max(model.k_strike - model.s0 * np.exp(x), 0.0) * rho1
        continuation = np.exp(-2 * model.r * model.delta) * np.sum(
            weights * norm.pdf((x - c) / sigma) / sigma * h_i
        )
        return max(exercise, continuation)

    lo = -a - 10 * sigma
    hi = a + 10 * sigma
    value, _ = integrate.quad(integrand, lo, hi, epsabs=1e-9, epsrel=1e-9, limit=200)
    return float(value)


def unsmoothed_stopping_value(model: CRRModel) -> float:
    """V(mu) by backward induction on the 2-step binomial tree (Sec. 3.3).

    Payoffs g_t(x) = e^{-r t Delta}(K - S0 e^x)_+ are already discounted to
    time 0, so the backward recursion V_t = max(g_t, E[V_{t+1}|F_t]) applies
    no further discount factor at each step (Section 3.2).
    """
    a = model.a
    pi = model.pi_plus
    disc1 = np.exp(-model.r * model.delta)
    disc2 = np.exp(-model.r * 2 * model.delta)
    # Terminal log-relative levels after two steps: -2a, 0, 0, 2a with the
    # standard binomial recombination (down/down, down/up, up/down, up/up).
    s2 = model.s0 * np.exp(np.array([-2 * a, 0.0, 0.0, 2 * a]))
    payoff2 = disc2 * np.maximum(model.k_strike - s2, 0.0)
    down_node = pi * payoff2[1] + (1 - pi) * payoff2[0]  # continuation from s1 = S0*e^{-a}
    up_node = pi * payoff2[3] + (1 - pi) * payoff2[2]    # continuation from s1 = S0*e^{a}
    s1_down = model.s0 * np.exp(-a)
    s1_up = model.s0 * np.exp(a)
    exercise1_down = disc1 * max(model.k_strike - s1_down, 0.0)
    exercise1_up = disc1 * max(model.k_strike - s1_up, 0.0)
    v1_down = max(exercise1_down, down_node)
    v1_up = max(exercise1_up, up_node)
    v0 = pi * v1_up + (1 - pi) * v1_down
    return float(v0)
