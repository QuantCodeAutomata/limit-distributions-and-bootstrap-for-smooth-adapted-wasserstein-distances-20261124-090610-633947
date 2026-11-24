"""Limiting distribution of sqrt(n)*AW_p(P_n, P) for the 4-path CRR model.

Implements the scalar energy formula (3.1) and Appendix A.3: the rescaled
empirical weight vector sqrt(n)*(w_hat_{1:3}-w_{1:3}) -> N(0, Sigma_w) with
Sigma_w = diag(w_{1:3}) - w_{1:3} w_{1:3}^T (multinomial CLT on 3 of the 4
cells), and the limit of sqrt(n)*AW_p(P_n,P) is Q_{P,p}(G)^{1/p} for
G = sum_{i=1}^3 Z_i psi_i, psi_i = e_{a_i}-e_{a_4}.

At p=2, Q_{P,2} is the quadratic form Z^T M Z for a 3x3 matrix M built from
formula (3.1); diagonalizing Sigma_w^{1/2} M Sigma_w^{1/2} gives weights
lambda_j so that Q_{P,2}(G) = sum_j lambda_j Y_j^2, Y_j iid N(0,1) (Remark
2.3's spectral form, "sum lambda_j Z_j^2"). At p=4, Q_{P,4}(G) is a
homogeneous quartic polynomial in Z, evaluated by quadrature at each Monte
Carlo draw of Z ~ N(0, Sigma_w).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from src.crr_bermudan import CRRModel, PATH_B, PATH_C
from src.mixture_gaussian import gauss_legendre_interval


def covariance_sigma_w(weights_mu: np.ndarray) -> np.ndarray:
    """Sigma_w = diag(w_{1:3}) - w_{1:3} w_{1:3}^T for the multinomial CLT."""
    w3 = weights_mu[:3]
    return np.diag(w3) - np.outer(w3, w3)


class LimitLawBasis:
    """Precomputed quadrature grids and basis fields for formula (3.1).

    Builds v1_i(x) and v2_i(x, z) for i = 1, 2, 3 (the three basis
    perturbations psi_i = e_{a_i} - e_{a_4}) on tensor-product Gauss-Legendre
    grids, following Appendix A.3's prescription of 192 nodes per coordinate.
    """

    def __init__(self, model: CRRModel, n_nodes: int, x_pad_sigma: float = 8.0):
        self.model = model
        a, sigma = model.a, model.sigma
        w_mu = model.weights_mu
        c = PATH_C * a
        b = PATH_B * a

        x_lo, x_hi = -a - x_pad_sigma * sigma, a + x_pad_sigma * sigma
        z_lo, z_hi = -2 * a - x_pad_sigma * sigma, 2 * a + x_pad_sigma * sigma
        self.x_nodes, self.x_weights = gauss_legendre_interval(n_nodes, x_lo, x_hi)
        self.z_nodes, self.z_weights = gauss_legendre_interval(n_nodes, z_lo, z_hi)

        phi_x = norm.pdf((self.x_nodes[:, None] - c[None, :]) / sigma) / sigma  # (Nx, 4)
        rho1 = phi_x @ w_mu  # (Nx,)
        dphi_x = -((self.x_nodes[:, None] - c[None, :]) / sigma**2) * phi_x  # phi_j'(x), (Nx,4)
        rho1_prime = dphi_x @ w_mu  # (Nx,)

        post = phi_x * w_mu[None, :] / rho1[:, None]  # post_j(x), (Nx, 4)
        post_prime = (
            dphi_x * w_mu[None, :] / rho1[:, None]
            - phi_x * w_mu[None, :] * rho1_prime[:, None] / rho1[:, None] ** 2
        )  # post_j'(x), (Nx, 4)

        cap_phi_z = norm.cdf((self.z_nodes[:, None] - b[None, :]) / sigma)  # Phi_sigma(z-b_j), (Nz,4)
        # K(x,z) = sum_j post_j(x) Phi_sigma(z-b_j); dK/dx = sum_j post_j'(x) Phi_sigma(z-b_j)
        self.cap_k = post @ cap_phi_z.T  # (Nx, Nz)
        self.dx_cap_k = post_prime @ cap_phi_z.T  # (Nx, Nz)
        self.rho1 = rho1  # (Nx,)
        # rho(x,z) = sum_j w_j phi_sigma(x-c_j) phi_sigma(z-b_j)
        phi_z_each = norm.pdf((self.z_nodes[:, None] - b[None, :]) / sigma) / sigma  # (Nz,4)
        self.rho = np.einsum("xj,zj,j->xz", phi_x, phi_z_each, w_mu)  # (Nx, Nz)

        v1_list, v2_list = [], []
        for i in range(3):
            h1_i = phi_x[:, i] - phi_x[:, 3]
            cap_h_i = norm.cdf((self.x_nodes - c[i]) / sigma) - norm.cdf((self.x_nodes - c[3]) / sigma)
            a_i = (
                phi_x[:, i][:, None] * cap_phi_z[:, i][None, :]
                - phi_x[:, 3][:, None] * cap_phi_z[:, 3][None, :]
            )  # (Nx, Nz)
            v1_i = -cap_h_i / rho1
            v2_i = (a_i - self.cap_k * h1_i[:, None] - cap_h_i[:, None] * self.dx_cap_k) / self.rho
            v1_list.append(v1_i)
            v2_list.append(v2_i)
        self.v1 = np.stack(v1_list, axis=0)  # (3, Nx)
        self.v2 = np.stack(v2_list, axis=0)  # (3, Nx, Nz)

    def energy_matrix_p2(self) -> np.ndarray:
        """3x3 matrix M with M_ij = <psi_i, psi_j>_{Q_{P,2}} (formula 3.1)."""
        m = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                term1 = np.sum(self.x_weights * self.v1[i] * self.v1[j] * self.rho1)
                term2 = np.sum(
                    (self.x_weights[:, None] * self.z_weights[None, :]) * self.v2[i] * self.v2[j] * self.rho
                )
                m[i, j] = term1 + term2
        return m

    def quartic_p4(self, z_draws: np.ndarray, batch_size: int = 2000) -> np.ndarray:
        """Q_{P,4}(sum_i Z_i psi_i) for each draw Z (shape (M, 3)) -> shape (M,).

        Processed in batches over the M draws: holding the full (M, Nx*Nz)
        array at once would exceed the available memory once n_nodes and the
        number of reference draws reach the paper's Appendix A.3 values.
        """
        v1_flat = self.v1.reshape(3, -1)  # (3, Nx)
        v2_flat = self.v2.reshape(3, -1)  # (3, Nx*Nz)
        weight_x = self.x_weights * self.rho1  # (Nx,)
        weight_grid = (self.x_weights[:, None] * self.z_weights[None, :] * self.rho).reshape(-1)

        m_total = z_draws.shape[0]
        out = np.empty(m_total)
        for start in range(0, m_total, batch_size):
            z_batch = z_draws[start:start + batch_size]
            v1_z = z_batch @ v1_flat  # (b, Nx)
            term1 = np.sum(weight_x[None, :] * np.abs(v1_z) ** 4, axis=1)
            v2_z = z_batch @ v2_flat  # (b, Nx*Nz)
            term2 = np.sum(weight_grid[None, :] * np.abs(v2_z) ** 4, axis=1)
            out[start:start + batch_size] = term1 + term2
        return out


def limit_quantile_p2(model: CRRModel, n_nodes: int, level: float, n_mc: int, rng: np.random.Generator) -> tuple[float, np.ndarray]:
    """Limiting quantile of [Q_{P,2}(G)]^{1/2} via the eigenvalues of Sigma_w^{1/2} M Sigma_w^{1/2}."""
    basis = LimitLawBasis(model, n_nodes)
    m = basis.energy_matrix_p2()
    sigma_w = covariance_sigma_w(model.weights_mu)
    sqrt_sigma_w = _sqrtm_psd(sigma_w)
    a_mat = sqrt_sigma_w @ m @ sqrt_sigma_w
    eigvals = np.linalg.eigvalsh(a_mat)
    eigvals = np.clip(eigvals, 0.0, None)
    y = rng.standard_normal((n_mc, 3))
    draws = y**2 @ eigvals
    q = float(np.quantile(draws, level))
    return float(np.sqrt(q)), eigvals


def limit_quantile_p4(model: CRRModel, n_nodes: int, level: float, n_mc: int, rng: np.random.Generator) -> float:
    """Limiting quantile of [Q_{P,4}(G)]^{1/4} via Monte Carlo over Z ~ N(0, Sigma_w)."""
    basis = LimitLawBasis(model, n_nodes)
    sigma_w = covariance_sigma_w(model.weights_mu)
    sqrt_sigma_w = _sqrtm_psd(sigma_w)
    y = rng.standard_normal((n_mc, 3))
    z_draws = y @ sqrt_sigma_w
    q4 = basis.quartic_p4(z_draws)
    q = float(np.quantile(q4, level))
    return float(q ** 0.25)


def _sqrtm_psd(mat: np.ndarray) -> np.ndarray:
    """Symmetric PSD square root via eigendecomposition."""
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T
