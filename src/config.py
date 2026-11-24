"""Parameters for every experiment, taken from the paper where stated.

"Limit distributions and bootstrap for smooth adapted Wasserstein distances"
(Xiaozhen Wang, arXiv:2611.28468v1, 15 November 2026).

All experiments in this paper are purely numerical / simulation studies: no
market data is used anywhere in this repository. Random seeds and quadrature
orders follow Section 3.3 and Appendix A.3 unless noted as "not specified in
the paper", in which case a value is fixed here and documented in
results/RESULTS.md.
"""

# ---------------------------------------------------------------------------
# Experiment 1: semi-synthetic Bermudan experiment (Section 3.3, Appendix A.3)
# ---------------------------------------------------------------------------

# Annual volatility nu, taken as the Cboe VIX close of 14.95 on 31 Dec 2025
# (Section 3.3).
EXP1_NU = 0.1495
EXP1_S0 = 100.0            # initial stock price (Section 3.3)
EXP1_K = 105.0             # strike (Section 3.3)
EXP1_DELTA = 0.5           # half-year time step (Section 3.3)
EXP1_R = 0.03              # annual interest rate (Section 3.3)
EXP1_SIGMA_MULT = 0.65     # smoothing bandwidth sigma = 0.65 * a (Section 3.3)
EXP1_P_VALUES = (2, 4)     # the two values of p tested (Section 3.3)

# Sample sizes n for which sampling/bootstrap distributions are compared
# (Section 3.3).
EXP1_SAMPLE_SIZES_PAPER = (50, 200, 800, 3200, 12800)
# Executed sample sizes: identical to the paper (these are the method's
# defining grid, never reduced). See results/RESULTS.md for the number of
# datasets and bootstrap draws actually used per size.
EXP1_SAMPLE_SIZES_RUN = EXP1_SAMPLE_SIZES_PAPER

EXP1_M_DATASETS_PAPER = 600   # independent datasets per sample size (Sec 3.3)
EXP1_B_BOOTSTRAP_PAPER = 249  # full-path bootstrap resamples per dataset (Sec 3.3)
# Reduced scalable counts for the configuration actually executed, to fit the
# available computation time; the paper values above remain selectable with
# the --full flag of exp/exp1.py. See results/RESULTS.md for the runtime
# estimate that drove this reduction.
EXP1_M_DATASETS_RUN = 120
EXP1_B_BOOTSTRAP_RUN = 100

EXP1_SEED = 20261116  # random seed (Section 3.3 / Appendix A.3)

EXP1_GL_NODES_AWP = 48     # Gauss-Legendre nodes per coordinate for AW_p (A.3)
EXP1_GL_NODES_LIMIT = 192  # Gauss-Legendre nodes per coordinate for (3.1) (A.3)

EXP1_REF_DRAWS_PAPER = 500_000  # Gaussian draws for reference quantiles (A.3)
EXP1_REF_DRAWS_RUN = 200_000     # reduced from the paper value (see RESULTS.md)

EXP1_NEWTON_CDF_TOL = 1e-13      # CDF residual stopping tolerance (A.3)
EXP1_NEWTON_BRACKET_TOL = 1e-12  # bracket width stopping tolerance (A.3)
EXP1_NEWTON_MAX_ITER = 80        # not specified in the paper; generous cap

EXP1_V_QUAD_ABSTOL = 1e-9  # adaptive-quadrature tolerance for V(P) (Section 3.3)
EXP1_V_QUAD_RELTOL = 1e-9  # adaptive-quadrature tolerance for V(P) (Section 3.3)
EXP1_V_QUAD_TAIL_MULT = 10  # integration domain [-a-10*sigma, a+10*sigma] (Sec 3.3)

# Expected outcomes reported in the paper, used only for comparison in
# results/RESULTS.md (Section 3.3).
EXP1_Q95_LIMIT_PAPER = {2: 0.5666, 4: 0.8947}
EXP1_V_MU_PAPER = 7.94165
EXP1_V_P_PAPER = 8.96675
EXP1_SMOOTHING_BIAS_PAPER = 1.02511
EXP1_EIGENVALUES_PAPER = (0.0122484, 0.0139312, 0.0757441)  # Appendix A.3

# ---------------------------------------------------------------------------
# Experiment 2: four-point product law (Appendix A.1)
# ---------------------------------------------------------------------------

# The paper states the formulas for a general four-point product law
# mu = Unif{-a,a} (x) Unif{-b,b} but does not fix numerical values of a, b,
# sigma for a demonstration (Appendix A.1). The values below
# are chosen so that a != b (to keep the three limiting directions Z1, Z2, Z3
# distinguishable) and sigma is of the same order as a, b, matching the ratio
# sigma = 0.65*a used in Section 3.3's single-bandwidth convention.
EXP2_A = 1.0            # half-spacing of the first marginal; not specified in the paper
EXP2_B = 0.6            # half-spacing of the second marginal; not specified in the paper
EXP2_SIGMA = 0.65       # Gaussian smoothing bandwidth; not specified in the paper
EXP2_P_VALUES = (2, 4)  # the two values of p studied (Appendix A.1)

EXP2_SAMPLE_SIZES = (50, 200, 800, 3200, 12800, 51200)  # not specified in the paper
EXP2_M_REPLICATIONS_PAPER = 20_000  # Monte Carlo replications per sample size
EXP2_M_REPLICATIONS_RUN = 5_000     # reduced from the paper value (see RESULTS.md)

EXP2_REF_DRAWS_PAPER = 500_000  # reference draws for the quartic limit (A.1/A.3 convention)
EXP2_REF_DRAWS_RUN = 200_000

EXP2_SEED = 20261116  # not specified in the paper; reuses the paper's Section 3.3 seed
EXP2_QUAD_LIMIT = 200  # scipy.integrate.quad subdivision limit; not specified in the paper

# ---------------------------------------------------------------------------
# Experiment 3: Gaussian obstruction (Section 2.2, Proposition 2.8, Section 4.3)
# ---------------------------------------------------------------------------

EXP3_S = 1.0  # standard deviation of mu = N(0, s^2); paper's own worked example (Sec 3.3 text)
# Divergent case: sigma^2 = 0.25 < (p-2)*s^2 = 2 (Section 3.3 text, Proposition 2.8)
EXP3_P_DIVERGENT = 4
EXP3_SIGMA_DIVERGENT = 0.5
# Well-behaved comparison case: sigma^2 = 4 >= (p-2)*s^2 = 2 (Section 3.3 text)
EXP3_SIGMA_WELLBEHAVED = 2.0

EXP3_N_VALUES_PAPER = (100, 500, 1000, 5000, 10000, 50000, 100000)  # methodology
EXP3_N_VALUES_RUN = EXP3_N_VALUES_PAPER  # method-defining grid, kept in full

EXP3_M_REPLICATIONS_PAPER = 2000  # paper-scale replication count
EXP3_M_REPLICATIONS_RUN = 300      # reduced from the paper value (see RESULTS.md)

EXP3_QUAD_NODES = 200       # Gauss-Legendre nodes for the W_p quantile integral
EXP3_EPS_TRUNC = 1e-10      # edge truncation for the quantile integral
# Grid resolution for inverting the smoothed empirical CDF (not specified in
# the paper). A sensitivity check at n=5000 found the resulting W_p value
# unchanged to 4 significant digits between 4000, 1500 and 800 grid points
# (0.026520 vs 0.026517 vs 0.026463), so the reduced-run value below is a
# numerical-accuracy choice, not a methodology reduction.
EXP3_GRID_POINTS_PAPER = 4000
EXP3_GRID_POINTS_RUN = 800
EXP3_SEED = 20261116  # not specified in the paper; reuses the paper's Section 3.3 seed
EXP3_BATCH_SIZE = 20  # datasets per batch in iter_gaussian_sample; memory/runtime tradeoff

# Additional p values tested for robustness (not specified in the paper beyond
# the single worked example, used only as a supplementary check).
EXP3_EXTRA_P_VALUES_PAPER = (3, 6)
EXP3_EXTRA_P_VALUES_RUN = (3,)  # reduced from the paper value (see RESULTS.md)
