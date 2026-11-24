"""Experiment 3: Gaussian obstruction to root-n tightness (Proposition 2.8).

For d = T = 1 the adapted Wasserstein distance AW_p reduces to the ordinary
Wasserstein distance W_p on R (Section 2). With mu = N(0, s^2) and smoothing
bandwidth sigma, this experiment contrasts two regimes of Proposition 2.8:

* divergent: sigma^2 < (p-2)*s^2, so the critical exponent
  zeta_p = ((p-2)*s^2 - sigma^2) / (2*(s^2+sigma^2)*(2*s^2+sigma^2)) is
  strictly positive and sqrt(n)*W_p(P_n, P) diverges to +infinity in
  probability as n grows (Section 2.2);
* well-behaved: sigma^2 >= (p-2)*s^2, so zeta_p <= 0 and sqrt(n)*W_p(P_n, P)
  stays tight.

Run as ``python -m exp.exp3`` for the reduced configuration that fits the
available computation time, or ``python -m exp.exp3 --full`` for the paper's
full sample-size grid and replication count.
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

from data.generate_synthetic import iter_gaussian_sample
from src import config as cfg
from src.gaussian_obstruction import critical_exponent, wasserstein_p_quantile_coupling

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def scaled_wp_distribution(
    s: float, sigma: float, p: float, n: int, m_replications: int, rng: np.random.Generator
) -> np.ndarray:
    """Monte Carlo draws of sqrt(n)*W_p(P_n, P) for one (s, sigma, p, n)."""
    values = []
    sqrt_n = np.sqrt(n)
    for batch in iter_gaussian_sample(s, n, m_replications, rng, batch_size=cfg.EXP3_BATCH_SIZE):
        for samples in batch:
            w_p = wasserstein_p_quantile_coupling(
                samples, s, sigma, p, cfg.EXP3_QUAD_NODES, cfg.EXP3_EPS_TRUNC, cfg.EXP3_GRID_POINTS
            )
            values.append(sqrt_n * w_p)
    return np.asarray(values)


def run_regime(label: str, s: float, sigma: float, p: float, n_values: tuple[int, ...], m_replications: int, seed: int) -> dict:
    zeta_p = critical_exponent(p, s, sigma)
    print(f"{label}: s={s} sigma={sigma} p={p} sigma^2={sigma**2:.4f} (p-2)s^2={(p-2)*s**2:.4f} zeta_p={zeta_p:.5f}")
    out = {"label": label, "s": s, "sigma": sigma, "p": p, "zeta_p": zeta_p, "by_n": []}
    for n in n_values:
        t0 = time.time()
        rng = np.random.default_rng(seed + n)
        dist = scaled_wp_distribution(s, sigma, p, n, m_replications, rng)
        row = {
            "n": n,
            "median": float(np.median(dist)),
            "q95": float(np.quantile(dist, 0.95)),
            "mean": float(np.mean(dist)),
            "sample": dist[: min(500, len(dist))].tolist(),
        }
        out["by_n"].append(row)
        print(f"  n={n}: median={row['median']:.4f} q95={row['q95']:.4f} ({time.time()-t0:.1f}s)")
    return out


def main(full: bool) -> None:
    n_values = cfg.EXP3_N_VALUES_PAPER if full else cfg.EXP3_N_VALUES_RUN
    m_replications = cfg.EXP3_M_REPLICATIONS_PAPER if full else cfg.EXP3_M_REPLICATIONS_RUN

    divergent = run_regime(
        "divergent (sigma^2 < (p-2)s^2)", cfg.EXP3_S, cfg.EXP3_SIGMA_DIVERGENT, cfg.EXP3_P_DIVERGENT,
        n_values, m_replications, cfg.EXP3_SEED,
    )
    well_behaved = run_regime(
        "well-behaved (sigma^2 >= (p-2)s^2)", cfg.EXP3_S, cfg.EXP3_SIGMA_WELLBEHAVED, cfg.EXP3_P_DIVERGENT,
        n_values, m_replications, cfg.EXP3_SEED + 777,
    )

    extra = []
    for p in cfg.EXP3_EXTRA_P_VALUES:
        extra.append(run_regime(
            f"divergent, p={p}", cfg.EXP3_S, cfg.EXP3_SIGMA_DIVERGENT, p, n_values, m_replications, cfg.EXP3_SEED + 10 * p,
        ))

    results = {
        "config": {"n_values": list(n_values), "m_replications": m_replications, "full": full},
        "divergent": divergent, "well_behaved": well_behaved, "extra_p": extra,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "exp3_results.json").write_text(json.dumps(results, indent=2))
    print(f"Saved {RESULTS_DIR / 'exp3_results.json'}")

    header = ["regime", "p", "zeta_p", "n", "median", "q95", "mean"]
    lines = [",".join(header)]
    for regime in [divergent, well_behaved, *extra]:
        for row in regime["by_n"]:
            lines.append(f"{regime['label']},{regime['p']},{regime['zeta_p']:.6f},{row['n']},"
                          f"{row['median']:.6f},{row['q95']:.6f},{row['mean']:.6f}")
    (RESULTS_DIR / "exp3_table.csv").write_text("\n".join(lines) + "\n")
    print(f"Saved {RESULTS_DIR / 'exp3_table.csv'}")

    _make_figure(divergent, well_behaved)


def _make_figure(divergent: dict, well_behaved: dict) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    for regime, marker in [(divergent, "o"), (well_behaved, "s")]:
        ns = [row["n"] for row in regime["by_n"]]
        medians = [row["median"] for row in regime["by_n"]]
        q95s = [row["q95"] for row in regime["by_n"]]
        ax.plot(ns, medians, marker=marker, label=f"{regime['label']}: median")
        ax.plot(ns, q95s, marker=marker, linestyle="--", label=f"{regime['label']}: 0.95-quantile")
    ax.set_xscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("sqrt(n) * W_p(P_n, P)")
    ax.set_title("Root-n tightness failure under the Gaussian obstruction (Prop. 2.8)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp3_figure.png", dpi=130)
    plt.close(fig)
    print(f"Saved {RESULTS_DIR / 'exp3_figure.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                         help="Run the paper's full sample-size grid and replication count instead of the reduced one.")
    args = parser.parse_args()
    main(args.full)
