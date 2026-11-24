# Limit distributions and bootstrap for smooth adapted Wasserstein distances

Replication of selected numerical experiments from:

Xiaozhen Wang, **"Limit distributions and bootstrap for smooth adapted
Wasserstein distances"**, arXiv:2611.28468v1, 15 November 2026.

All three experiments below are purely numerical / simulation studies built
on Theorem 2.2 (limit distribution), Proposition 2.8 (Gaussian obstruction to
root-n tightness) and the scalar energy formula (3.1) of the paper. No market
data is used anywhere in this repository: every input is simulated from an
explicitly specified probability model, so `data/` contains only a synthetic
data generator and no market-data loader.

## Repository layout

```
src/        flat library modules (model, Rosenblatt coupling, AW_p distance,
            bootstrap, limit-law formulas, config)
exp/        experiment drivers, run as `python -m exp.<name>`
tests/      pytest unit tests of src/ components
data/       data/generate_synthetic.py generates every simulated input
results/    saved tables, figures and RESULTS.md with the paper comparison
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No API keys or external data are required.

## Tests

```bash
python -m pytest
```

## Experiments

### Experiment 1 — Semi-synthetic Bermudan experiment (Section 3.3, Appendix A.3)

Two-date Cox-Ross-Rubinstein binomial model with Gaussian smoothing, the
common-uniform Rosenblatt coupling for AW_p, 0.95-quantiles from the scalar
energy formula (3.1), full-path bootstrap, and calibration of error bounds for
the Bermudan put optimal-stopping value.

```bash
python -m exp.exp1            # reduced configuration (see results/RESULTS.md)
python -m exp.exp1 --full     # paper configuration: 600 datasets, 249 bootstrap draws
```

Produces `results/exp1_results.json`, `results/exp1_table1.csv`,
`results/exp1_figure1.png` (sampling vs. bootstrap distributions, Figure 1)
and `results/exp1_figure2.png` (stopping-value errors and bootstrap radii,
Figure 2).

### Experiment 2 — Four-point product law (Appendix A.1)

Explicit nonquadratic limit distribution of the smoothed adapted Wasserstein
distance for the four-point product law
mu = Unif{-a,a} (x) Unif{-b,b}: a weighted chi-square sum at p=2 and a
homogeneous quartic polynomial in three independent Gaussians at p=4, with the
cross-period term vanishing once the empirical measure is replaced by the
product of its marginal empirical measures.

```bash
python -m exp.exp2
python -m exp.exp2 --full     # paper-scale Monte Carlo replication count
```

Produces `results/exp2_results.json`, `results/exp2_table.csv` and
`results/exp2_figure.png`.

### Experiment 3 — Gaussian obstruction to root-n tightness (Section 2.2, Proposition 2.8, Section 4.3)

For mu = N(0, s^2), p > 2 and d = T = 1 (so AW_p reduces to the ordinary
Wasserstein distance W_p), verifies that sqrt(n) * W_p(P_n, P) is not tight
when sigma^2 < (p-2) s^2, and that root-n tightness is restored when
sigma^2 >= (p-2) s^2.

```bash
python -m exp.exp3
python -m exp.exp3 --full     # paper-scale replication count and CDF-inversion grid
```

Produces `results/exp3_results.json`, `results/exp3_table.csv` and
`results/exp3_figure.png`.

## Results

See `results/RESULTS.md` for the full comparison against the paper's reported
values, the execution configuration actually used, and every deviation from
the paper.
