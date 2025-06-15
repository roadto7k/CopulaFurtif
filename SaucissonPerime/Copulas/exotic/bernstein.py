import numpy as np
import math
from scipy.stats import beta, kendalltau
from SaucissonPerime.Copulas.base import BaseCopula


class BernsteinCopula(BaseCopula):
    """
    Bernstein copula of order m with weight matrix P.

    C(u,v) = sum_{i=0}^m sum_{j=0}^m P[i,j] · B_{i,m}(u) · B_{j,m}(v),
    where B_{i,m}(t) = binomial(m, i) t^i (1−t)^(m−i).

    Sampling: choose (i,j) with prob P[i,j], then
      U ~ Beta(i+1, m−i+1), V ~ Beta(j+1, m−j+1).
    """

    def __init__(self, P):
        """
        Parameters
        ----------
        P : array-like, shape (m+1, m+1)
            Nonnegative weight matrix summing to 1, satisfying
            row‐ and column‐sums = 1/(m+1) for a valid copula.
        """
        super().__init__()
        self.P = np.asarray(P, float)
        if self.P.ndim != 2 or self.P.shape[0] != self.P.shape[1]:
            raise ValueError("P must be square matrix of shape (m+1, m+1)")
        if not np.all(self.P >= 0):
            raise ValueError("All P entries must be nonnegative")
        if abs(self.P.sum() - 1.0) > 1e-6:
            raise ValueError("P must sum to 1")
        self.m = self.P.shape[0] - 1
        self.type = "bernstein"
        self.name = f"Bernstein Copula (m={self.m})"
        self.bounds_param = []  # no free parameters
        self.parameters = np.array([])

    def _B(self, i, t):
        """Bernstein basis B_{i,m}(t)."""
        return math.comb(self.m, i) * (t**i) * ((1 - t)**(self.m - i))

    def _B_derivative(self, i, t):
        """Derivative d/dt B_{i,m}(t) = m [B_{i-1,m-1}(t) - B_{i,m-1}(t)]."""
        if self.m == 0:
            return 0.0
        b1 = self._B(i - 1, t) if 0 <= i - 1 <= self.m - 1 else 0.0
        b2 = self._B(i, t)     if 0 <= i     <= self.m - 1 else 0.0
        return self.m * (b1 - b2)

    def get_cdf(self, u, v, param=None):
        """
        C(u,v) = sum_{i,j} P[i,j] B_{i,m}(u) B_{j,m}(v).
        """
        u = np.atleast_1d(u)
        v = np.atleast_1d(v)
        C = np.zeros_like(u, float)
        for i in range(self.m + 1):
            Bi = np.array([self._B(i, uu) for uu in u])
            for j in range(self.m + 1):
                Bj = np.array([self._B(j, vv) for vv in v])
                C += self.P[i, j] * Bi * Bj
        return C if C.shape != () else C.item()

    def get_pdf(self, u, v, param=None):
        """
        PDF c(u,v) = sum_{i,j} P[i,j] B'_{i,m}(u) B'_{j,m}(v).
        """
        u = np.atleast_1d(u)
        v = np.atleast_1d(v)
        pdf = np.zeros_like(u, float)
        for i in range(self.m + 1):
            dBi = np.array([self._B_derivative(i, uu) for uu in u])
            for j in range(self.m + 1):
                dBj = np.array([self._B_derivative(j, vv) for vv in v])
                pdf += self.P[i, j] * dBi * dBj
        return pdf if pdf.shape != () else pdf.item()

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """
        ∂C/∂u = sum_{i,j} P[i,j] B'_{i,m}(u) B_{j,m}(v).
        """
        u = np.atleast_1d(u)
        v = np.atleast_1d(v)
        pd = np.zeros_like(u, float)
        for i in range(self.m + 1):
            dBi = np.array([self._B_derivative(i, uu) for uu in u])
            for j in range(self.m + 1):
                Bj = np.array([self._B(j, vv) for vv in v])
                pd += self.P[i, j] * dBi * Bj
        return pd if pd.shape != () else pd.item()

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """
        ∂C/∂v = sum_{i,j} P[i,j] B_{i,m}(u) B'_{j,m}(v).
        """
        u = np.atleast_1d(u)
        v = np.atleast_1d(v)
        pd = np.zeros_like(u, float)
        for i in range(self.m + 1):
            Bi = np.array([self._B(i, uu) for uu in u])
            for j in range(self.m + 1):
                dBj = np.array([self._B_derivative(j, vv) for vv in v])
                pd += self.P[i, j] * Bi * dBj
        return pd if pd.shape != () else pd.item()

    def conditional_cdf_v_given_u(self, u, v, param=None):
        """
        P(V ≤ v | U = u) = ∂C/∂u(u, v).
        """
        return self.partial_derivative_C_wrt_u(u, v)

    def conditional_cdf_u_given_v(self, u, v, param=None):
        """
        P(U ≤ u | V = v) = ∂C/∂v(u, v).
        """
        return self.partial_derivative_C_wrt_v(u, v)

    def sample(self, n, param=None):
        """
        Sample n pairs from the Bernstein copula via mixture of Betas:
          1. Choose (i,j) with probability P[i,j].
          2. Draw U ~ Beta(i+1, m−i+1), V ~ Beta(j+1, m−j+1).
        """
        flat_P = self.P.ravel()
        indices = np.random.choice(self.P.size, size=n, p=flat_P)
        U = np.empty(n)
        V = np.empty(n)
        for k, idx in enumerate(indices):
            i, j = divmod(idx, self.m + 1)
            U[k] = beta.rvs(i + 1, self.m - i + 1)
            V[k] = beta.rvs(j + 1, self.m - j + 1)
        return np.column_stack((U, V))

    def kendall_tau(self, param=None):
        """
        Estimate Kendall's τ by Monte Carlo:
          draw 2000 samples and compute sample τ.
        """
        sample = self.sample(2000)
        return kendalltau(sample[:, 0], sample[:, 1]).correlation

    def LTDC(self, param=None):
        """
        Lower‑tail dependence λ_L = 0 (no tail‐dependence).
        """
        return 0.0

    def UTDC(self, param=None):
        """
        Upper‑tail dependence λ_U = 0 (no tail‐dependence).
        """
        return 0.0

    def IAD(self, data):
        """
        Integrated Anderson–Darling not implemented for Bernstein copula.
        """
        raise NotImplementedError("IAD not available for Bernstein copula.")

    def AD(self, data):
        """
        Anderson–Darling not implemented for Bernstein copula.
        """
        raise NotImplementedError("AD not available for Bernstein copula.")
