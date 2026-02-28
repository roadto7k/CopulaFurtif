"""
Bernstein copula (non-parametric) of order m with weight matrix P.

C(u,v) = sum_{i=0}^m sum_{j=0}^m P[i,j] * B_{i,m}(u) * B_{j,m}(v),
where B_{i,m}(t) = C(m,i) t^i (1-t)^(m-i).

Sampling:
  1) draw (I,J) ~ categorical(P)
  2) U ~ Beta(I+1, m-I+1), V ~ Beta(J+1, m-J+1)

Notes
-----
- P must be nonnegative, sum to 1, and (approximately) have uniform margins:
    row_sums = col_sums = 1/(m+1)
  for the result to be a valid copula (uniform marginals).
"""

from __future__ import annotations

import numpy as np
from numpy.random import default_rng
from scipy.stats import binom, kendalltau
from scipy.stats import beta as beta_dist

from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel, CopulaParameters
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


class BernsteinCopula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """Bernstein copula with fixed weight matrix P (no scalar parameters)."""

    def __init__(self, P: np.ndarray, *, validate: bool = True, tol: float = 1e-8):
        super().__init__()
        self.name = "Bernstein Copula"
        self.type = "bernstein"
        self.default_optim_method = "N/A"

        self.P = None
        self.m = None
        self.set_weights(P, validate=validate, tol=tol)

        # no free scalar parameters
        self.init_parameters(CopulaParameters(np.array([]), [], []))

    # ---------------------------------------------------------------------
    # Weight matrix handling
    # ---------------------------------------------------------------------
    def set_weights(self, P: np.ndarray, *, validate: bool = True, tol: float = 1e-8) -> None:
        P = np.asarray(P, dtype=float)
        if P.ndim != 2 or P.shape[0] != P.shape[1]:
            raise ValueError("P must be a square matrix of shape (m+1, m+1).")
        if validate:
            if np.any(P < 0):
                raise ValueError("All P entries must be nonnegative.")
            s = float(P.sum())
            if not np.isfinite(s) or abs(s - 1.0) > tol:
                raise ValueError(f"P must sum to 1 (got {s}).")

            m = P.shape[0] - 1
            target = 1.0 / (m + 1)
            r = P.sum(axis=1)
            c = P.sum(axis=0)
            if np.max(np.abs(r - target)) > 1e-3 or np.max(np.abs(c - target)) > 1e-3:
                # not fatal mathematically, but then marginals won't be uniform.
                raise ValueError(
                    "P must have (approximately) uniform margins: row_sums = col_sums = 1/(m+1). "
                    "If you build P from data, apply a Sinkhorn scaling step."
                )

        self.P = P
        self.m = P.shape[0] - 1

    # ---------------------------------------------------------------------
    # Bernstein basis and derivative (vectorized)
    # ---------------------------------------------------------------------
    @staticmethod
    def _clip01(x, eps: float = 1e-12):
        return np.clip(x, eps, 1.0 - eps)

    @staticmethod
    def _ensure_pairwise(u, v):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        if u.ndim == 0 and v.ndim == 0:
            return u, v, True
        if u.shape != v.shape:
            raise ValueError("Vectorized evaluation is pairwise: u and v must have the same shape.")
        return u, v, False

    def _basis(self, t: np.ndarray) -> np.ndarray:
        """Return B_{i,m}(t) for i=0..m as a matrix shape (m+1, n)."""
        t = self._clip01(np.asarray(t, dtype=float))
        t = np.atleast_1d(t)
        i = np.arange(self.m + 1)[:, None]
        return binom.pmf(i, self.m, t[None, :])

    def _dbasis(self, t: np.ndarray) -> np.ndarray:
        """
        Return d/dt B_{i,m}(t) for i=0..m as matrix shape (m+1, n).

        Identity:
          d/dt B_{i,m}(t) = m [B_{i-1,m-1}(t) - B_{i,m-1}(t)]
        with out-of-range terms = 0.
        """
        t = self._clip01(np.asarray(t, dtype=float))
        t = np.atleast_1d(t)

        if self.m == 0:
            return np.zeros((1, t.size), dtype=float)

        k = np.arange(self.m)[:, None]  # 0..m-1
        Bm1 = binom.pmf(k, self.m - 1, t[None, :])  # shape (m, n)

        dB = np.zeros((self.m + 1, t.size), dtype=float)
        dB[0, :] = -self.m * Bm1[0, :]
        dB[self.m, :] = self.m * Bm1[self.m - 1, :]
        dB[1:self.m, :] = self.m * (Bm1[0:self.m - 1, :] - Bm1[1:self.m, :])
        return dB

    def _beta_cdf_mat(self, t):
        t = np.asarray(t, float)
        t = np.clip(t, 0.0, 1.0)  # IMPORTANT: allow exact 0/1 for boundaries
        t = np.atleast_1d(t)
        i = np.arange(self.m + 1)
        a = i + 1.0
        b = (self.m - i) + 1.0
        # shape (m+1, n)
        return beta_dist.cdf(t[None, :], a[:, None], b[:, None])

    def _beta_pdf_mat(self, t):
        t = np.asarray(t, float)
        # for pdf avoid exact 0/1 to prevent inf; but keep boundary logic in CDF
        eps = 1e-12
        t = np.clip(t, eps, 1.0 - eps)
        t = np.atleast_1d(t)
        i = np.arange(self.m + 1)
        a = i + 1.0
        b = (self.m - i) + 1.0
        return beta_dist.pdf(t[None, :], a[:, None], b[:, None])

    # ---------------------------------------------------------------------
    # Core API
    # ---------------------------------------------------------------------
    def get_cdf(self, u, v, param=None):
        u, v, is_scalar = self._ensure_pairwise(u, v)
        Fu = self._beta_cdf_mat(u)  # (m+1, n)
        Fv = self._beta_cdf_mat(v)  # (m+1, n)
        out = np.einsum("ik,ij,jk->k", Fu, self.P, Fv)
        return float(out[0]) if is_scalar else out.reshape(np.asarray(u).shape)

    def get_pdf(self, u, v, param=None):
        u, v, is_scalar = self._ensure_pairwise(u, v)
        fu = self._beta_pdf_mat(u)
        fv = self._beta_pdf_mat(v)
        out = np.einsum("ik,ij,jk->k", fu, self.P, fv)
        out = np.maximum(out, 0.0)
        return float(out[0]) if is_scalar else out.reshape(np.asarray(u).shape)
    def partial_derivative_C_wrt_u(self, u, v, param=None):
        u, v, is_scalar = self._ensure_pairwise(u, v)
        fu = self._beta_pdf_mat(u)
        Fv = self._beta_cdf_mat(v)
        out = np.einsum("ik,ij,jk->k", fu, self.P, Fv)
        out = np.clip(out, 0.0, 1.0)
        return float(out[0]) if is_scalar else out.reshape(np.asarray(u).shape)

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        u, v, is_scalar = self._ensure_pairwise(u, v)
        Fu = self._beta_cdf_mat(u)
        fv = self._beta_pdf_mat(v)
        out = np.einsum("ik,ij,jk->k", Fu, self.P, fv)
        out = np.clip(out, 0.0, 1.0)
        return float(out[0]) if is_scalar else out.reshape(np.asarray(u).shape)

    # ---------------------------------------------------------------------
    # Dependence summaries / fitting helpers
    # ---------------------------------------------------------------------
    def blomqvist_beta(self, param=None) -> float:
        """Blomqvist beta: 4*C(1/2,1/2) - 1."""
        return 4.0 * float(self.get_cdf(0.5, 0.5)) - 1.0

    def kendall_tau(self, param=None) -> float:
        """Deterministic MC estimate of Kendall tau (no closed form here)."""
        rng = default_rng(0)
        data = self.sample(4000, rng=rng)
        return float(kendalltau(data[:, 0], data[:, 1]).correlation)

    def LTDC(self, param=None) -> float:
        return 0.0

    def UTDC(self, param=None) -> float:
        return 0.0

    # ---------------------------------------------------------------------
    # Sampling
    # ---------------------------------------------------------------------
    def sample(self, n: int, param=None, rng=None):
        """
        Sample n pairs from Bernstein copula via mixture of Betas:
          1) Choose (i,j) with prob P[i,j]
          2) U ~ Beta(i+1, m-i+1), V ~ Beta(j+1, m-j+1)
        """
        if rng is None:
            rng = default_rng()

        flat_P = self.P.ravel()
        idx = rng.choice(self.P.size, size=int(n), p=flat_P)

        i = idx // (self.m + 1)
        j = idx % (self.m + 1)

        # numpy Generator.beta supports vectorized a,b
        U = rng.beta(i + 1.0, (self.m - i) + 1.0)
        V = rng.beta(j + 1.0, (self.m - j) + 1.0)
        return np.column_stack([U, V])

    # ---------------------------------------------------------------------
    # Optional: update P from data (simple histogram + Sinkhorn scaling)
    # ---------------------------------------------------------------------
    @staticmethod
    def _sinkhorn_to_uniform(P, target, n_iter=500, eps=1e-15):
        P = np.asarray(P, float)
        P = P + eps  # avoid exact zeros that can break scaling
        P /= P.sum()

        for _ in range(n_iter):
            rs = P.sum(axis=1)
            P *= (target / rs)[:, None]
            cs = P.sum(axis=0)
            P *= (target / cs)[None, :]
        P /= P.sum()
        return P

    def init_from_data(self, u, v):
        """
        Rebuild P from pseudo-observations (u,v) using a (m+1)x(m+1) histogram
        then Sinkhorn-scale to enforce uniform margins.

        This keeps the model non-parametric and compatible with the CopulaModel API
        (no scalar parameters to return).
        """
        u = np.asarray(u, float).ravel()
        v = np.asarray(v, float).ravel()
        mask = np.isfinite(u) & np.isfinite(v)
        u = u[mask]
        v = v[mask]
        if u.size < 50:
            return self.get_parameters()

        u = self._clip01(u)
        v = self._clip01(v)

        edges = np.linspace(0.0, 1.0, self.m + 2)
        H, _, _ = np.histogram2d(u, v, bins=[edges, edges])
        P0 = H / max(1, u.size)

        target = np.full(self.m + 1, 1.0 / (self.m + 1))
        P1 = self._sinkhorn_to_uniform(P0, target=target, n_iter=500)

        self.set_weights(P1, validate=False)
        return self.get_parameters()