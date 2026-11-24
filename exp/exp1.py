"""Experiment 1: semi-synthetic Bermudan experiment (Section 3.3, Appendix A.3).

Reproduces the comparison between the sampling distribution of
sqrt(n)*AW_p(P_n, P) and the full-path bootstrap distribution of
sqrt(n)*AW_p(P_n*, P_n) for the two-date CRR binomial model, the convergence
of their 0.95-quantiles to the limiting value q_{0.95} of formula (3.1), the
empirical coverage of the resulting distance bound, and the bootstrap
calibration of error bounds around the smoothed value V(P) and the unsmoothed
value V(mu) (Proposition 3.1).

Run as ``python -m exp.exp1`` for the reduced configuration that fits the
available computation time, or ``python -m exp.exp1 --full`` for the paper's
full configuration (Section 3.3: n in {50, 200, 800, 3200, 12800}, 600
datasets, 249 bootstrap resamples, 500000 reference draws).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config as cfg
from src.bootstrap import bootstrap_resample, order_statistic_quantile
from src.crr_bermudan import CRRModel, aw_p_distance, smoothed_stopping_value, unsmoothed_stopping_value
from src.exp1_limit_law import limit_quantile_p2, limit_quantile_p4
from data.generate_synthetic import iter_crr_path_counts

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
LIPSCHITZ_L = cfg.EXP1_K  # payoff g_t(x) = e^{-rtDelta}(K - S0 e^x)_+ has Lipschitz constant K (Sec 3.3)


def run_one_size(
    model: CRRModel,
    n: int,
    m_datasets: int,
    b_bootstrap: int,
    v_p: float,
    v_mu: float,
    bias: float,
    rng: np.random.Generator,
) -> dict:
    """All per-dataset statistics for one sample size n (Section 3.3, step 6-9)."""
    p_values = cfg.EXP1_P_VALUES
    sampling_scaled = {p: [] for p in p_values}
    bootstrap_q95_star = {p: [] for p in p_values}
    pooled_bootstrap_scaled = {p: [] for p in p_values}
    value_err_vP, value_err_vmu = [], []
    coverage_dist = {p: 0 for p in p_values}
    coverage_value_vP = {p: 0 for p in p_values}
    coverage_value_vmu = {p: 0 for p in p_values}
    bootstrap_radius = {p: [] for p in p_values}

    n_done = 0
    for batch in iter_crr_path_counts(model.weights_mu, n, m_datasets, rng, batch_size=20):
        for w_hat in batch:
            n_done += 1
            dist = aw_p_distance(
                model, model.weights_mu, w_hat, p_values, cfg.EXP1_GL_NODES_AWP,
                cfg.EXP1_NEWTON_CDF_TOL, cfg.EXP1_NEWTON_BRACKET_TOL, cfg.EXP1_NEWTON_MAX_ITER,
            )
            v_pn = smoothed_stopping_value(model, w_hat)
            err_vP = abs(v_pn - v_p)
            err_vmu = abs(v_pn - v_mu)
            value_err_vP.append(err_vP)
            value_err_vmu.append(err_vmu)

            boots = bootstrap_resample(w_hat, n, b_bootstrap, rng)
            boot_scaled = {p: [] for p in p_values}
            for w_star in boots:
                d_star = aw_p_distance(
                    model, w_star, w_hat, p_values, cfg.EXP1_GL_NODES_AWP,
                    cfg.EXP1_NEWTON_CDF_TOL, cfg.EXP1_NEWTON_BRACKET_TOL, cfg.EXP1_NEWTON_MAX_ITER,
                )
                for p in p_values:
                    boot_scaled[p].append(np.sqrt(n) * d_star[p])

            for p in p_values:
                scaled = np.sqrt(n) * dist[p]
                sampling_scaled[p].append(scaled)
                q_star = order_statistic_quantile(np.asarray(boot_scaled[p]), 0.95)
                bootstrap_q95_star[p].append(q_star)
                pooled_bootstrap_scaled[p].extend(boot_scaled[p])
                radius = LIPSCHITZ_L * q_star / np.sqrt(n)
                bootstrap_radius[p].append(radius)
                coverage_dist[p] += int(scaled <= q_star)
                coverage_value_vP[p] += int(err_vP <= radius)
                coverage_value_vmu[p] += int(err_vmu <= radius + bias)

    out = {"n": n, "m_datasets": n_done, "b_bootstrap": b_bootstrap}
    for p in p_values:
        out[f"p{p}"] = {
            "sampling_scaled": sampling_scaled[p],
            "bootstrap_q95_star": bootstrap_q95_star[p],
            "pooled_bootstrap_scaled_sample": list(
                np.random.default_rng(0).choice(pooled_bootstrap_scaled[p], size=min(2000, len(pooled_bootstrap_scaled[p])), replace=False)
            ),
            "bootstrap_radius": bootstrap_radius[p],
            "coverage_dist_pct": 100.0 * coverage_dist[p] / n_done,
            "coverage_value_vP_pct": 100.0 * coverage_value_vP[p] / n_done,
            "coverage_value_vmu_pct": 100.0 * coverage_value_vmu[p] / n_done,
            "sampling_q95_empirical": float(np.quantile(sampling_scaled[p], 0.95)),
            "bootstrap_q95_median": float(np.median(bootstrap_q95_star[p])),
            "value_err_vP_q95": float(np.quantile(value_err_vP, 0.95)),
            "bootstrap_radius_median": float(np.median(bootstrap_radius[p])),
        }
    return out


def main(full: bool) -> None:
    sample_sizes = cfg.EXP1_SAMPLE_SIZES_PAPER if full else cfg.EXP1_SAMPLE_SIZES_RUN
    m_datasets = cfg.EXP1_M_DATASETS_PAPER if full else cfg.EXP1_M_DATASETS_RUN
    b_bootstrap = cfg.EXP1_B_BOOTSTRAP_PAPER if full else cfg.EXP1_B_BOOTSTRAP_RUN
    ref_draws = cfg.EXP1_REF_DRAWS_PAPER if full else cfg.EXP1_REF_DRAWS_RUN

    model = CRRModel(cfg.EXP1_NU, cfg.EXP1_S0, cfg.EXP1_K, cfg.EXP1_DELTA, cfg.EXP1_R, cfg.EXP1_SIGMA_MULT)
    print(f"CRR model: a={model.a:.6f} sigma={model.sigma:.6f} pi+={model.pi_plus:.6f}")
    print(f"weights_mu={model.weights_mu}")

    v_mu = unsmoothed_stopping_value(model)
    v_p = smoothed_stopping_value(model, model.weights_mu)
    bias = abs(v_p - v_mu)
    print(f"V(mu)={v_mu:.5f} (paper {cfg.EXP1_V_MU_PAPER}); V(P)={v_p:.5f} (paper {cfg.EXP1_V_P_PAPER}); "
          f"bias={bias:.5f} (paper {cfg.EXP1_SMOOTHING_BIAS_PAPER})")

    q95_limit, eigenvalues_p2 = {}, None
    for p in cfg.EXP1_P_VALUES:
        rng_limit = np.random.default_rng(cfg.EXP1_SEED + 1000 + p)
        if p == 2:
            q, eig = limit_quantile_p2(model, cfg.EXP1_GL_NODES_LIMIT, 0.95, ref_draws, rng_limit)
            eigenvalues_p2 = eig.tolist()
        else:
            q = limit_quantile_p4(model, cfg.EXP1_GL_NODES_LIMIT, 0.95, ref_draws, rng_limit)
        q95_limit[p] = q
        print(f"limiting q_0.95 (p={p}) = {q:.4f} (paper {cfg.EXP1_Q95_LIMIT_PAPER[p]})")

    results = {
        "model": {"a": model.a, "sigma": model.sigma, "pi_plus": model.pi_plus,
                  "weights_mu": model.weights_mu.tolist()},
        "v_mu": v_mu, "v_p": v_p, "smoothing_bias": bias,
        "q95_limit": q95_limit, "eigenvalues_p2": eigenvalues_p2,
        "config": {"sample_sizes": list(sample_sizes), "m_datasets": m_datasets,
                   "b_bootstrap": b_bootstrap, "ref_draws": ref_draws, "full": full},
        "by_n": [],
    }

    for n in sample_sizes:
        t0 = time.time()
        rng = np.random.default_rng(cfg.EXP1_SEED + n)
        out = run_one_size(model, n, m_datasets, b_bootstrap, v_p, v_mu, bias, rng)
        dt = time.time() - t0
        print(f"n={n}: done in {dt:.1f}s; coverage(dist) p2={out['p2']['coverage_dist_pct']:.1f}% "
              f"p4={out['p4']['coverage_dist_pct']:.1f}%")
        results["by_n"].append(out)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "exp1_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")

    _make_table1(results)
    _make_figure1(results)
    _make_figure2(results)


def _make_table1(results: dict) -> None:
    lines = ["n,p,coverage_dist_pct,sampling_q95,bootstrap_q95_median,norm_sampling_q95,norm_bootstrap_q95,"
             "value_err_vP_q95,bootstrap_radius_median,coverage_value_vP_pct,coverage_value_vmu_pct"]
    for row in results["by_n"]:
        n = row["n"]
        for p in cfg.EXP1_P_VALUES:
            d = row[f"p{p}"]
            qlim = results["q95_limit"][p]
            lines.append(
                f"{n},{p},{d['coverage_dist_pct']:.2f},{d['sampling_q95_empirical']:.4f},"
                f"{d['bootstrap_q95_median']:.4f},{d['sampling_q95_empirical']/qlim:.4f},"
                f"{d['bootstrap_q95_median']/qlim:.4f},{d['value_err_vP_q95']:.4f},"
                f"{d['bootstrap_radius_median']:.4f},{d['coverage_value_vP_pct']:.1f},"
                f"{d['coverage_value_vmu_pct']:.1f}"
            )
    (RESULTS_DIR / "exp1_table1.csv").write_text("\n".join(lines) + "\n")
    print(f"Saved {RESULTS_DIR / 'exp1_table1.csv'}")


def _make_figure1(results: dict) -> None:
    """Sampling vs. pooled bootstrap distributions, normalized by q_{0.95} (Figure 1)."""
    fig, axes = plt.subplots(len(cfg.EXP1_P_VALUES), len(results["by_n"]), figsize=(4 * len(results["by_n"]), 7))
    for i, p in enumerate(cfg.EXP1_P_VALUES):
        qlim = results["q95_limit"][p]
        for j, row in enumerate(results["by_n"]):
            ax = axes[i, j]
            d = row[f"p{p}"]
            sampling = np.asarray(d["sampling_scaled"]) / qlim
            boot = np.asarray(d["pooled_bootstrap_scaled_sample"]) / qlim
            ax.hist(sampling, bins=30, density=True, alpha=0.5, label="sampling")
            ax.hist(boot, bins=30, density=True, alpha=0.5, label="bootstrap (pooled)")
            ax.axvline(1.0, color="k", linestyle="--", linewidth=1)
            ax.set_title(f"p={p}, n={row['n']}")
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle("Sampling vs. full-path bootstrap distributions of sqrt(n)*AW_p, normalized by q_0.95 (Fig. 1)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_figure1.png", dpi=130)
    plt.close(fig)
    print(f"Saved {RESULTS_DIR / 'exp1_figure1.png'}")


def _make_figure2(results: dict) -> None:
    """Stopping-value errors and bootstrap radii around V(P) and V(mu) (Figure 2)."""
    fig, axes = plt.subplots(1, len(cfg.EXP1_P_VALUES), figsize=(5 * len(cfg.EXP1_P_VALUES), 4))
    ns = [row["n"] for row in results["by_n"]]
    for i, p in enumerate(cfg.EXP1_P_VALUES):
        ax = axes[i] if len(cfg.EXP1_P_VALUES) > 1 else axes
        radii = [row[f"p{p}"]["bootstrap_radius_median"] for row in results["by_n"]]
        err_vP = [row[f"p{p}"]["value_err_vP_q95"] for row in results["by_n"]]
        ax.plot(ns, radii, marker="o", label="median bootstrap radius")
        ax.plot(ns, err_vP, marker="s", label="0.95-quantile |V(P_n)-V(P)|")
        ax.axhline(results["smoothing_bias"], color="grey", linestyle=":", label="smoothing bias")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n")
        ax.set_title(f"p={p}")
        ax.legend(fontsize=7)
    fig.suptitle("Bootstrap calibration of stopping-value error bounds (Fig. 2, Table 1)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_figure2.png", dpi=130)
    plt.close(fig)
    print(f"Saved {RESULTS_DIR / 'exp1_figure2.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                         help="Run the paper's full configuration (Section 3.3) instead of the reduced one.")
    args = parser.parse_args()
    main(args.full)
