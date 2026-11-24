"""Unit tests for src.exp1_limit_law (scalar energy formula 3.1, Appendix A.3)."""

from __future__ import annotations

import numpy as np

from src import config as cfg
from src.crr_bermudan import CRRModel
from src.exp1_limit_law import LimitLawBasis, covariance_sigma_w, limit_quantile_p2, limit_quantile_p4


def make_model() -> CRRModel:
    return CRRModel(cfg.EXP1_NU, cfg.EXP1_S0, cfg.EXP1_K, cfg.EXP1_DELTA, cfg.EXP1_R, cfg.EXP1_SIGMA_MULT)


def test_covariance_sigma_w_is_symmetric_and_psd():
    model = make_model()
    sigma_w = covariance_sigma_w(model.weights_mu)
    assert sigma_w.shape == (3, 3)
    assert np.max(np.abs(sigma_w - sigma_w.T)) < 1e-12
    eigvals = np.linalg.eigvalsh(sigma_w)
    assert np.all(eigvals >= -1e-10)


def test_energy_matrix_p2_is_symmetric_and_psd():
    model = make_model()
    basis = LimitLawBasis(model, n_nodes=48)
    m = basis.energy_matrix_p2()
    assert m.shape == (3, 3)
    assert np.max(np.abs(m - m.T)) < 1e-6
    eigvals = np.linalg.eigvalsh(m)
    assert np.all(eigvals >= -1e-6)


def test_limit_quantile_p2_eigenvalues_are_positive_and_order_of_magnitude_matches_paper():
    model = make_model()
    rng = np.random.default_rng(0)
    q, eigvals = limit_quantile_p2(model, n_nodes=48, level=0.95, n_mc=20_000, rng=rng)
    assert np.all(eigvals > 0.0)
    assert q > 0.0
    # Appendix A.3 reports eigenvalues of order 1e-2; with the reduced node
    # count used here (48 instead of 192) only the order of magnitude is
    # checked, the exact match is verified in the full experiment run.
    paper_eig = np.array(cfg.EXP1_EIGENVALUES_PAPER)
    assert np.max(eigvals) < 1.0
    assert np.min(eigvals) > 1e-4
    assert abs(np.log10(np.sort(eigvals)[-1]) - np.log10(np.sort(paper_eig)[-1])) < 1.0


def test_limit_quantile_p4_is_positive():
    model = make_model()
    rng = np.random.default_rng(1)
    q = limit_quantile_p4(model, n_nodes=48, level=0.95, n_mc=5000, rng=rng)
    assert q > 0.0
