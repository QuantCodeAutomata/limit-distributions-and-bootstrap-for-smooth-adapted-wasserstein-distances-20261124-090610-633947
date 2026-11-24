"""Unit tests for src.rosenblatt_awp (formula A.1 quantile quadrature)."""

from __future__ import annotations

import numpy as np

from src.rosenblatt_awp import aw_p_general

PATH_C = np.array([-1.0, -1.0, 1.0, 1.0])
PATH_B = np.array([-2.0, 0.0, 0.0, 2.0])


def test_aw_p_general_zero_for_identical_weights():
    a, sigma = 0.1, 0.07
    w = np.array([0.3, 0.2, 0.2, 0.3])
    dist = aw_p_general(PATH_C * a, PATH_B * a, w, w, sigma, (2, 4), n_nodes=24,
                         newton_cdf_tol=1e-12, newton_bracket_tol=1e-11, newton_max_iter=60)
    assert dist[2] < 1e-6
    assert dist[4] < 1e-6


def test_aw_p_general_positive_and_increasing_with_total_variation():
    a, sigma = 0.1, 0.07
    w0 = np.array([0.25, 0.25, 0.25, 0.25])
    w_small = np.array([0.3, 0.2, 0.25, 0.25])
    w_large = np.array([0.5, 0.1, 0.2, 0.2])
    d_small = aw_p_general(PATH_C * a, PATH_B * a, w0, w_small, sigma, (2,), n_nodes=24,
                            newton_cdf_tol=1e-12, newton_bracket_tol=1e-11, newton_max_iter=60)[2]
    d_large = aw_p_general(PATH_C * a, PATH_B * a, w0, w_large, sigma, (2,), n_nodes=24,
                            newton_cdf_tol=1e-12, newton_bracket_tol=1e-11, newton_max_iter=60)[2]
    assert d_small > 0.0
    assert d_large > d_small


def test_aw_p_general_symmetric_in_its_two_arguments():
    a, sigma = 0.1, 0.07
    w1 = np.array([0.4, 0.2, 0.2, 0.2])
    w2 = np.array([0.2, 0.3, 0.3, 0.2])
    d12 = aw_p_general(PATH_C * a, PATH_B * a, w1, w2, sigma, (2, 4), n_nodes=24,
                        newton_cdf_tol=1e-12, newton_bracket_tol=1e-11, newton_max_iter=60)
    d21 = aw_p_general(PATH_C * a, PATH_B * a, w2, w1, sigma, (2, 4), n_nodes=24,
                        newton_cdf_tol=1e-12, newton_bracket_tol=1e-11, newton_max_iter=60)
    assert abs(d12[2] - d21[2]) < 1e-8
    assert abs(d12[4] - d21[4]) < 1e-8
