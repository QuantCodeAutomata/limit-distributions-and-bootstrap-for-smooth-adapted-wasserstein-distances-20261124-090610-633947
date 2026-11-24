# AGENTS.md

Guide for coding agents and contributors working in this repository.

## What this repository implements

A replication of three numerical experiments from Xiaozhen Wang,
"Limit distributions and bootstrap for smooth adapted Wasserstein distances"
(arXiv:2611.28468v1):

1. Experiment 1 — the two-date CRR Bermudan model: limit distribution,
   full-path bootstrap, and optimal-stopping calibration (Section 3.3,
   Appendix A.3).
2. Experiment 2 — the four-point product law: explicit nonquadratic limit
   distribution (Appendix A.1).
3. Experiment 3 — the Gaussian obstruction to root-n tightness for p > 2 at
   small bandwidths (Section 2.2, Proposition 2.8, Section 4.3).

All three experiments are self-contained simulation studies; no market data
is used anywhere in this repository.

## Layout

```
src/        flat library modules, no subdirectories
  config.py            every experiment parameter, module-level UPPER_CASE
  crr_bermudan.py       CRR model, Bermudan backward induction, V(P) quadrature
  mixture_gaussian.py   Gaussian-mixture CDF/quantile utilities (bracketed Newton inversion)
  rosenblatt_awp.py     Rosenblatt coupling and AW_p by quantile quadrature (Experiment 1)
  exp1_limit_law.py     scalar energy formula (3.1), energy matrix, limit quantiles
  bootstrap.py          full-path multinomial bootstrap of path counts
  four_point_law.py     Experiment 2: C_{c,p}, kappa_{c,k}, tangent energy, Monte Carlo
  gaussian_obstruction.py  Experiment 3: smoothed CDF inversion, W_p via quantile coupling
exp/        exp1.py, exp2.py, exp3.py — run with `python -m exp.<name>`
data/       generate_synthetic.py — every simulated input used by the experiments
tests/      pytest unit tests of the src/ modules
results/    tables, figures, and RESULTS.md with the paper comparison
```

## Commands

```bash
pip install -r requirements.txt
python -m pytest -m "not market_data"   # no market_data tests exist in this repo
python -m exp.exp1            # reduced configuration
python -m exp.exp1 --full     # paper configuration (M=600, B=249, 500k ref draws)
python -m exp.exp2
python -m exp.exp2 --full     # paper-scale Monte Carlo replication count
python -m exp.exp3
python -m exp.exp3 --full     # paper-scale replication count and CDF-inversion grid
```

`reproduce.sh` runs the test suite and all three experiments at their
reduced (default) configuration.

## Conventions

- Every experiment parameter lives in `src/config.py` as a module-level
  UPPER_CASE variable with a comment stating its meaning, unit, and source
  (a paper section/equation or "not specified in the paper"). No classes or
  dictionaries of parameters.
- All simulated input data is produced only by `data/generate_synthetic.py`;
  no experiment script generates its own data or duplicates a computation
  already in `src/`.
- `exp/*.py` only calls the public functions of `src/*.py` with the
  signatures as written; it does not invent extra keyword arguments.
- Tests use small, fast fixtures (synthetic draws, coarse quadrature grids)
  and check the paper's qualitative properties (signs, monotonicity, the
  absence of look-ahead, known closed-form special cases), not the full-scale
  reported numbers.
- Re-running an experiment command overwrites its files in `results/`; after
  any such rerun, re-check the numbers and verdicts in `results/RESULTS.md`
  by hand and update them if they changed.
- No API keys are needed anywhere in this repository.
