"""Synthetic sample generators for every experiment.

No market data is used anywhere in this paper's experiments: Section 3.3 is a
semi-synthetic Bermudan option model, Appendix A.1 is a four-point product
law, and Section 4.3 / Proposition 2.8 is a univariate Gaussian law. Each
function below draws the i.i.d. "observed sample" that stands in for data in
the corresponding experiment; everything downstream (bootstrap resampling,
distance computation, limit laws) lives in ``src/``.

All generators yield batches instead of materializing the full
(m_replications, n) array at once, so that peak memory stays bounded even at
the paper's largest configurations (e.g. Appendix A.3's n = 12800 with 600
datasets, or Section 4.3's n = 100000 with thousands of replications).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def iter_crr_path_counts(
    weights_mu: np.ndarray, n: int, m_datasets: int, rng: np.random.Generator, batch_size: int = 100
) -> Iterator[np.ndarray]:
    """Batches of empirical path-frequency vectors, Multinomial(n, weights_mu)/n.

    Each yielded array has shape (b, len(weights_mu)) with b <= batch_size,
    representing b independent datasets of n draws from the latent law mu
    (Section 3.3: "600 independent datasets are simulated ... as multinomial
    path counts with probabilities w").
    """
    remaining = m_datasets
    while remaining > 0:
        b = min(batch_size, remaining)
        counts = rng.multinomial(n, weights_mu, size=b)
        yield counts / n
        remaining -= b


def iter_rademacher_pairs(
    n: int, m_replications: int, rng: np.random.Generator, batch_size: int = 200
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Batches of n i.i.d. Rademacher pairs (epsilon, zeta) in {-1,+1}^2.

    Each yielded pair has shape (b, n) with b <= batch_size, representing b
    independent datasets of n draws for the four-point product law
    A^{(i)} = (a*epsilon_i, b*zeta_i) of Appendix A.1.
    """
    remaining = m_replications
    while remaining > 0:
        b = min(batch_size, remaining)
        eps = rng.choice(np.array([-1.0, 1.0]), size=(b, n))
        zeta = rng.choice(np.array([-1.0, 1.0]), size=(b, n))
        yield eps, zeta
        remaining -= b


def iter_gaussian_sample(
    s: float, n: int, m_replications: int, rng: np.random.Generator, batch_size: int = 20
) -> Iterator[np.ndarray]:
    """Batches of n i.i.d. draws from N(0, s^2) (Section 4.3, Proposition 2.8).

    Each yielded array has shape (b, n) with b <= batch_size, representing b
    independent datasets A^{(1)}, ..., A^{(n)} ~ N(0, s^2).
    """
    remaining = m_replications
    while remaining > 0:
        b = min(batch_size, remaining)
        yield rng.normal(loc=0.0, scale=s, size=(b, n))
        remaining -= b
