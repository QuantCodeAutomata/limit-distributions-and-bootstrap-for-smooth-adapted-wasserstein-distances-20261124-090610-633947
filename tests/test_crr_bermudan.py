"""Unit tests for src.crr_bermudan (Section 3.3, Appendix A.3)."""

from __future__ import annotations

import numpy as np

from src import config as cfg
from src.crr_bermudan import CRRModel, aw_p_distance, smoothed_stopping_value, unsmoothed_stopping_value


def make_model() -> CRRModel:
    return CRRModel(cfg.EXP1_NU, cfg.EXP1_S0, cfg.EXP1_K, cfg.EXP1_DELTA, cfg.EXP1_R, cfg.EXP1_SIGMA_MULT)


def test_weights_mu_are_a_probability_vector():
    model = make_model()
    w = model.weights_mu
    assert w.shape == (4,)
    assert np.all(w >= 0.0)
    assert abs(np.sum(w) - 1.0) < 1e-12


def test_a_sigma_and_pi_plus_match_definitions():
    model = make_model()
    expected_a = cfg.EXP1_NU * np.sqrt(cfg.EXP1_DELTA)
    assert abs(model.a - expected_a) < 1e-12
    assert abs(model.sigma - cfg.EXP1_SIGMA_MULT * expected_a) < 1e-12
    u, d = np.exp(model.a), np.exp(-model.a)
    expected_pi = (np.exp(cfg.EXP1_R * cfg.EXP1_DELTA) - d) / (u - d)
    assert abs(model.pi_plus - expected_pi) < 1e-12
    assert 0.0 < model.pi_plus < 1.0


def test_aw_p_distance_vanishes_for_identical_weights():
    model = make_model()
    w = model.weights_mu
    dist = aw_p_distance(model, w, w, (2, 4), n_nodes=24,
                          newton_cdf_tol=1e-12, newton_bracket_tol=1e-11, newton_max_iter=60)
    assert dist[2] < 1e-6
    assert dist[4] < 1e-6


def test_aw_p_distance_positive_for_different_weights():
    model = make_model()
    w = model.weights_mu
    w_other = np.array([0.4, 0.2, 0.2, 0.2])
    dist = aw_p_distance(model, w, w_other, (2, 4), n_nodes=24,
                          newton_cdf_tol=1e-12, newton_bracket_tol=1e-11, newton_max_iter=60)
    assert dist[2] > 1e-3
    assert dist[4] > 1e-3


def test_unsmoothed_stopping_value_is_a_valid_put_price():
    model = make_model()
    v_mu = unsmoothed_stopping_value(model)
    # A Bermudan put is worth at least its intrinsic value at time 0 and at
    # most the discounted strike (Section 3.3).
    intrinsic = max(model.k_strike - model.s0, 0.0)
    assert intrinsic - 1e-8 <= v_mu <= model.k_strike


def test_unsmoothed_value_matches_paper(regtest=None):
    model = make_model()
    v_mu = unsmoothed_stopping_value(model)
    assert abs(v_mu - cfg.EXP1_V_MU_PAPER) < 1e-4


def test_smoothed_value_close_to_paper():
    model = make_model()
    v_p = smoothed_stopping_value(model, model.weights_mu)
    assert abs(v_p - cfg.EXP1_V_P_PAPER) < 1e-3


def test_smoothing_bias_matches_paper():
    model = make_model()
    v_mu = unsmoothed_stopping_value(model)
    v_p = smoothed_stopping_value(model, model.weights_mu)
    bias = abs(v_p - v_mu)
    assert abs(bias - cfg.EXP1_SMOOTHING_BIAS_PAPER) < 1e-3
