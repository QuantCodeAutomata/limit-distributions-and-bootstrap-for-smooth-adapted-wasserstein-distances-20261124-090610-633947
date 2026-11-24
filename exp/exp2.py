"""Experiment 2: four-point product law limit distribution (Appendix A.1).

Reproduces the explicit nonquadratic limit distribution of
n^{p/2} * AW_p(P_n, P)^p for the four-point product law
mu = Unif{-a,a} (x) Unif{-b,b} on R^2, for p = 2 (a weighted sum of three
independent chi-square(1) variables) and p = 4 (a homogeneous quartic
polynomial in three independent standard Gaussians, Appendix A.1). It also
demonstrates that replacing the empirical measure by the product of its
empirical marginals makes the cross-period interaction term Z_3 vanish.

By Appendix A.1's joint p-homogeneity of the tangent energy Q_{P,p} in
(u, v, w), evaluating Q_{P,p} at (sqrt(n)*u_n, sqrt(n)*v_n, sqrt(n)*w_n) is
identical to n^{p/2} * Q_{P,p}(u_n, v_n, w_n); the Monte Carlo loop below uses
this identity directly instead of computing AW_p via optimal transport.

Run as ``python -m exp.exp2`` for the reduced configuration that fits the
available computation time, or ``python -m exp.exp2 --full`` for the paper's
Monte Carlo replication count.
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

from data.generate_synthetic import iter_rademacher_pairs
from src import config as cfg
from src.four_point_law import FourPointLawBasis, empirical_weights

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def mc_tangent_energy(
    basis: FourPointLawBasis, n: int, m_replications: int, rng: np.random.Generator, product_marginal: bool
) -> dict[int, np.ndarray]:
    """Monte Carlo draws of n^{p/2}*Q_{P,p}(u_n, v_n, w_n) for every p in basis.p_values.

    ``product_marginal=True`` replaces w_n by u_n*v_n, the empirical measure's
    marginal product, demonstrating that the cross-period term Z_3 vanishes
    (Appendix A.1, step 9).
    """
    out = {p: [] for p in basis.p_values}
    sqrt_n = np.sqrt(n)
    for eps, zeta in iter_rademacher_pairs(n, m_replications, rng, batch_size=200):
        u_n, v_n, w_n = empirical_weights(eps, zeta)
        if product_marginal:
            w_n = u_n * v_n
        su, sv, sw = sqrt_n * u_n, sqrt_n * v_n, sqrt_n * w_n
        for p in basis.p_values:
            out[p].append(basis.tangent_energy(su, sv, sw, p))
    return {p: np.concatenate(vals) for p, vals in out.items()}


def reference_distribution(basis: FourPointLawBasis, n_draws: int, rng: np.random.Generator) -> dict[int, np.ndarray]:
    """Reference draws of the limit law Q_{P,p}(Z_1, Z_2, Z_3), Z ~ N(0, I_3)."""
    z = rng.standard_normal((n_draws, 3))
    z1, z2, z3 = z[:, 0], z[:, 1], z[:, 2]
    return {2: basis.closed_form_p2(z1, z2, z3), 4: basis.closed_form_p4(z1, z2, z3)}


def main(full: bool) -> None:
    m_replications = cfg.EXP2_M_REPLICATIONS_PAPER if full else cfg.EXP2_M_REPLICATIONS_RUN
    ref_draws = cfg.EXP2_REF_DRAWS_PAPER if full else cfg.EXP2_REF_DRAWS_RUN
    sample_sizes = cfg.EXP2_SAMPLE_SIZES

    basis = FourPointLawBasis(cfg.EXP2_A, cfg.EXP2_B, cfg.EXP2_SIGMA, cfg.EXP2_P_VALUES, cfg.EXP1_GL_NODES_LIMIT)
    print(f"C_a,p={basis.c_ap}, C_b,p={basis.c_bp}, kappa_a2={basis.kappa_a2:.6f}, kappa_a4={basis.kappa_a4:.6f}")

    rng_ref = np.random.default_rng(cfg.EXP2_SEED + 1)
    reference = reference_distribution(basis, ref_draws, rng_ref)

    results = {
        "config": {"a": cfg.EXP2_A, "b": cfg.EXP2_B, "sigma": cfg.EXP2_SIGMA, "sample_sizes": list(sample_sizes),
                   "m_replications": m_replications, "ref_draws": ref_draws, "full": full},
        "constants": {"C_a,p": basis.c_ap, "C_b,p": basis.c_bp, "kappa_a2": basis.kappa_a2, "kappa_a4": basis.kappa_a4},
        "reference_quantiles": {p: {"median": float(np.median(reference[p])), "q95": float(np.quantile(reference[p], 0.95))}
                                 for p in basis.p_values},
        "by_n": [],
    }

    rows = []
    for n in sample_sizes:
        t0 = time.time()
        rng = np.random.default_rng(cfg.EXP2_SEED + n)
        mc = mc_tangent_energy(basis, n, m_replications, rng, product_marginal=False)
        rng_prod = np.random.default_rng(cfg.EXP2_SEED + n + 500_000)
        mc_prod = mc_tangent_energy(basis, n, m_replications, rng_prod, product_marginal=True)

        # Direct demonstration that sqrt(n)*w_n -> N(0,1) while sqrt(n)*(u_n*v_n) -> 0
        # (Appendix A.1, step 9): recompute the raw scaled weights for n and compare
        # their sample standard deviations.
        rng_w = np.random.default_rng(cfg.EXP2_SEED + n + 1_000_000)
        w_n_scaled, w_n_prod_scaled = [], []
        for eps, zeta in iter_rademacher_pairs(n, m_replications, rng_w, batch_size=200):
            u_n, v_n, w_n = empirical_weights(eps, zeta)
            w_n_scaled.append(np.sqrt(n) * w_n)
            w_n_prod_scaled.append(np.sqrt(n) * (u_n * v_n))
        w_n_scaled = np.concatenate(w_n_scaled)
        w_n_prod_scaled = np.concatenate(w_n_prod_scaled)

        row = {"n": n, "std_sqrt_n_w_n": float(np.std(w_n_scaled)), "std_sqrt_n_w_n_product": float(np.std(w_n_prod_scaled))}
        for p in basis.p_values:
            row[f"p{p}_mc_median"] = float(np.median(mc[p]))
            row[f"p{p}_mc_q95"] = float(np.quantile(mc[p], 0.95))
            row[f"p{p}_mc_product_median"] = float(np.median(mc_prod[p]))
            row[f"p{p}_mc_product_q95"] = float(np.quantile(mc_prod[p], 0.95))
        rows.append(row)
        results["by_n"].append({
            "n": n,
            "mc_sample": {p: mc[p][: min(2000, len(mc[p]))].tolist() for p in basis.p_values},
            "mc_product_sample": {p: mc_prod[p][: min(2000, len(mc_prod[p]))].tolist() for p in basis.p_values},
            **{k: v for k, v in row.items() if k != "n"},
        })
        print(f"n={n}: done in {time.time()-t0:.1f}s; std(sqrt(n)*w_n)={row['std_sqrt_n_w_n']:.4f} "
              f"(->1) std(sqrt(n)*u_n*v_n)={row['std_sqrt_n_w_n_product']:.4f} (->0)")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "exp2_results.json").write_text(json.dumps(results, indent=2))
    print(f"Saved {RESULTS_DIR / 'exp2_results.json'}")

    header = ["n", "std_sqrt_n_w_n", "std_sqrt_n_w_n_product"]
    for p in basis.p_values:
        header += [f"p{p}_mc_median", f"p{p}_mc_q95", f"p{p}_mc_product_median", f"p{p}_mc_product_q95"]
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(row[h]) for h in header))
    (RESULTS_DIR / "exp2_table.csv").write_text("\n".join(lines) + "\n")
    print(f"Saved {RESULTS_DIR / 'exp2_table.csv'}")

    _make_figure(results, basis)


def _make_figure(results: dict, basis: FourPointLawBasis) -> None:
    """Monte Carlo vs. reference limit distributions at the largest n, and the
    vanishing of the cross-period term under the marginal-product substitute."""
    by_n = results["by_n"]
    largest = by_n[-1]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, p in enumerate(basis.p_values):
        ax = axes[i]
        mc = np.asarray(largest["mc_sample"][p])
        ref_q95 = results["reference_quantiles"][p]["q95"]
        ax.hist(mc, bins=40, density=True, alpha=0.6, label=f"MC, n={largest['n']}")
        ax.axvline(ref_q95, color="k", linestyle="--", label="reference 0.95-quantile")
        ax.set_title(f"p={p}: n^{{p/2}}*Q_P,p(u_n,v_n,w_n)")
        ax.legend(fontsize=7)

    ax = axes[2]
    ns = [row["n"] for row in by_n]
    ax.plot(ns, [row["std_sqrt_n_w_n"] for row in by_n], marker="o", label="std(sqrt(n)*w_n) -> 1")
    ax.plot(ns, [row["std_sqrt_n_w_n_product"] for row in by_n], marker="s", label="std(sqrt(n)*u_n v_n) -> 0")
    ax.set_xscale("log")
    ax.set_xlabel("n")
    ax.set_title("Vanishing cross-period term under the marginal product")
    ax.legend(fontsize=7)

    fig.suptitle("Four-point product law: Monte Carlo limit distributions (Appendix A.1)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp2_figure.png", dpi=130)
    plt.close(fig)
    print(f"Saved {RESULTS_DIR / 'exp2_figure.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                         help="Run the paper's full Monte Carlo replication count instead of the reduced one.")
    args = parser.parse_args()
    main(args.full)
