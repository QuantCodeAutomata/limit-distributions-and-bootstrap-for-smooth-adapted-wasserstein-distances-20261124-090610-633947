"""Unit tests for src.four_point_law (Appendix A.1)."""

from __future__ import annotations

import numpy as np
from scipy import integrate

from src.four_point_law import (
    FourPointLawBasis,
    cap_d_c,
    constant_c_cp,
    constant_kappa_ck,
    empirical_weights,
    f_c,
    s_c,
)


def test_f_c_integrates_to_one():
    value, _ = integrate.quad(lambda x: f_c(np.array(x), 1.0, 0.65), -np.inf, np.inf)
    assert abs(value - 1.0) < 1e-8


def test_s_c_is_bounded_by_one_and_antisymmetric():
    x = np.linspace(-5, 5, 101)
    s = s_c(x, 1.0, 0.65)
    assert np.all(np.abs(s) <= 1.0)
    assert np.max(np.abs(s + s_c(-x, 1.0, 0.65))) < 1e-12


def test_cap_d_c_is_even_and_negative():
    # D_c(x) = (Phi_sigma(x-c) - Phi_sigma(x+c))/2 satisfies D_c(-x) = D_c(x)
    # (substitute Phi_sigma(-y) = 1 - Phi_sigma(y) in both terms), and is
    # strictly negative since Phi_sigma(x-c) < Phi_sigma(x+c) for c > 0.
    x = np.array([0.3, 1.1, -0.3, -1.1])
    d = cap_d_c(x, 1.0, 0.65)
    assert abs(d[0] - d[2]) < 1e-12
    assert abs(d[1] - d[3]) < 1e-12
    assert np.all(d < 0.0)


def test_constant_c_cp_is_positive():
    assert constant_c_cp(1.0, 0.65, 2) > 0.0
    assert constant_c_cp(1.0, 0.65, 4) > 0.0


def test_constant_kappa_ck_odd_moment_vanishes_by_symmetry():
    # s_c is an odd function of x and f_c is even, so integral s_c^1 * f_c = 0.
    kappa_1 = constant_kappa_ck(1.0, 0.65, 1)
    assert abs(kappa_1) < 1e-8


def test_constant_kappa_ck_even_moments_positive_and_bounded():
    kappa_2 = constant_kappa_ck(1.0, 0.65, 2)
    kappa_4 = constant_kappa_ck(1.0, 0.65, 4)
    assert 0.0 < kappa_4 < kappa_2 < 1.0


def test_tangent_energy_vanishes_at_origin_and_is_nonnegative():
    basis = FourPointLawBasis(a=1.0, b=0.6, sigma=0.65, p_values=(2, 4), n_nodes=48)
    zero = np.array([0.0])
    for p in (2, 4):
        assert basis.tangent_energy(zero, zero, zero, p)[0] < 1e-10
    u = np.array([0.5, -0.3])
    v = np.array([0.2, 0.1])
    w = np.array([0.1, -0.2])
    for p in (2, 4):
        assert np.all(basis.tangent_energy(u, v, w, p) >= 0.0)


def test_closed_form_quantities_nonnegative():
    basis = FourPointLawBasis(a=1.0, b=0.6, sigma=0.65, p_values=(2, 4), n_nodes=48)
    rng = np.random.default_rng(0)
    z1, z2, z3 = rng.standard_normal(1000), rng.standard_normal(1000), rng.standard_normal(1000)
    assert np.all(basis.closed_form_p2(z1, z2, z3) >= 0.0)
    assert np.all(basis.closed_form_p4(z1, z2, z3) >= 0.0)


def test_empirical_weights_from_rademacher_pairs():
    eps = np.array([[1.0, 1.0, -1.0, -1.0]])
    zeta = np.array([[1.0, -1.0, 1.0, -1.0]])
    u_n, v_n, w_n = empirical_weights(eps, zeta)
    assert abs(u_n[0]) < 1e-12
    assert abs(v_n[0]) < 1e-12
    assert abs(w_n[0]) < 1e-12
