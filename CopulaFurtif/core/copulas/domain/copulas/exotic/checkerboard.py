import numpy as np
from scipy.stats import kendalltau
from copulas.domain.copulas.base import BaseCopula


class CheckerboardCopula(BaseCopula):
    """
    Checkerboard copula (multilinear extension of the empirical copula).

    Given a weight matrix W of shape (m, m) summing to 1, the unit square [0,1]^2
    is partitioned into an m×m grid of cells of size 1/m.  Within each cell,
    the copula is defined by bilinear interpolation of the cell‑masses W[i,j].

    References
    ----------
    - Deheuvels (1979): multilinear extension of empirical copula :contentReference[oaicite:0]{index=0}.
    """

    def __init__(self, W):
        """
        Parameters
        ----------
        W : array-like, shape (m, m)
            Nonnegative weights summing to 1, giving the probability mass in each cell.
        """
        super().__init__()
        self.W = np.asarray(W, float)
        if self.W.ndim != 2 or self.W.shape[0] != self.W.shape[1]:
            raise ValueError("W must be a square matrix")
        if np.any(self.W < 0) or abs(self.W.sum() - 1.0) > 1e-6:
            raise ValueError("W must be nonnegative and sum to 1")
        self.m = self.W.shape[0]
        self.type = "checkerboard"
        self.name = f"Checkerboard Copula (m={self.m})"
        self.bounds_param = []       # no parameters to estimate
        self.parameters = np.array([])

    def get_cdf(self, u, v, param=None):
        """
        C(u,v) = ∑_{i<k, j<ℓ} W[i,j]
                + α * ∑_{j<ℓ} W[k,j]
                + β * ∑_{i<k} W[i,ℓ]
                + α·β·W[k,ℓ],

        where k = floor(u·m), ℓ = floor(v·m),
              α = u·m − k,  β = v·m − ℓ. :contentReference[oaicite:1]{index=1}
        """
        u_cl = np.clip(u, 0.0, 1.0)
        v_cl = np.clip(v, 0.0, 1.0)
        # handle scalars and arrays uniformly
        uc = np.atleast_1d(u_cl)
        vc = np.atleast_1d(v_cl)
        C = np.empty_like(uc, float)
        m = self.m

        for idx, (ui, vi) in enumerate(zip(uc, vc)):
            x = min(int(ui * m), m - 1)
            y = min(int(vi * m), m - 1)
            α = ui * m - x
            β = vi * m - y

            # sum over full cells
            block = self.W[:x, :y].sum() if x > 0 and y > 0 else 0.0
            # partial edges
            row = self.W[x, :y].sum() if y > 0 else 0.0
            col = self.W[:x, y].sum() if x > 0 else 0.0
            # fractional cell
            cell = self.W[x, y]

            C[idx] = block + α * row + β * col + α * β * cell

        return C.item() if np.isscalar(u) and np.isscalar(v) else C

    def get_pdf(self, u, v, param=None):
        """
        PDF c(u,v) = m^2 · W[k,ℓ] for u in ((k)/m, (k+1)/m), v in ((ℓ)/m, (ℓ+1)/m).
        """
        uc = np.atleast_1d(u)
        vc = np.atleast_1d(v)
        pdf = np.empty_like(uc, float)
        m2 = float(self.m**2)

        for idx, (ui, vi) in enumerate(zip(uc, vc)):
            k = min(int(np.clip(ui, 0.0, 1.0) * self.m), self.m - 1)
            ℓ = min(int(np.clip(vi, 0.0, 1.0) * self.m), self.m - 1)
            pdf[idx] = m2 * self.W[k, ℓ]

        return pdf.item() if np.isscalar(u) and np.isscalar(v) else pdf

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """
        ∂C/∂u = m·[ ∑_{j<ℓ} W[k,j] + β·W[k,ℓ] ] for u in cell k, with β = v·m−ℓ.
        """
        uc = np.atleast_1d(u)
        vc = np.atleast_1d(v)
        pd = np.empty_like(uc, float)

        for idx, (ui, vi) in enumerate(zip(uc, vc)):
            k = min(int(np.clip(ui, 0.0, 1.0) * self.m), self.m - 1)
            ℓ = min(int(np.clip(vi, 0.0, 1.0) * self.m), self.m - 1)
            β = vi * self.m - ℓ
            row = self.W[k, :ℓ].sum() if ℓ > 0 else 0.0
            pd[idx] = self.m * (row + β * self.W[k, ℓ])

        return pd.item() if np.isscalar(u) and np.isscalar(v) else pd

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """
        ∂C/∂v = m·[ ∑_{i<k} W[i,ℓ] + α·W[k,ℓ] ] for v in cell ℓ, with α = u·m−k.
        """
        uc = np.atleast_1d(u)
        vc = np.atleast_1d(v)
        pd = np.empty_like(uc, float)

        for idx, (ui, vi) in enumerate(zip(uc, vc)):
            k = min(int(np.clip(ui, 0.0, 1.0) * self.m), self.m - 1)
            ℓ = min(int(np.clip(vi, 0.0, 1.0) * self.m), self.m - 1)
            α = ui * self.m - k
            col = self.W[:k, ℓ].sum() if k > 0 else 0.0
            pd[idx] = self.m * (col + α * self.W[k, ℓ])

        return pd.item() if np.isscalar(u) and np.isscalar(v) else pd

    def conditional_cdf_v_given_u(self, u, v, param=None):
        """
        P(V ≤ v | U = u) = ∂C/∂u(u,v) / ∂C/∂u(u,1).
        """
        num = self.partial_derivative_C_wrt_u(u, v)
        # at v=1, ℓ=m-1, β=1
        denom = self.partial_derivative_C_wrt_u(u, 1.0)
        return num / denom

    def conditional_cdf_u_given_v(self, u, v, param=None):
        """
        P(U ≤ u | V = v) = ∂C/∂v(u,v) / ∂C/∂v(1,v).
        """
        num = self.partial_derivative_C_wrt_v(u, v)
        denom = self.partial_derivative_C_wrt_v(1.0, v)
        return num / denom

    def sample(self, n, param=None):
        """
        Sample n points by:
          1. choose cell (i,j) with probability W[i,j]
          2. sample U ∼ Uniform(i/m, (i+1)/m), V ∼ Uniform(j/m, (j+1)/m). :contentReference[oaicite:2]{index=2}
        """
        flat = self.W.ravel()
        idx = np.random.choice(len(flat), size=n, p=flat)
        U = np.empty(n)
        V = np.empty(n)
        for t, ind in enumerate(idx):
            i, j = divmod(ind, self.m)
            U[t] = np.random.rand() / self.m + i / self.m
            V[t] = np.random.rand() / self.m + j / self.m
        return np.column_stack((U, V))

    def kendall_tau(self, param=None):
        """
        Estimate Kendall's τ by Monte Carlo sampling.
        """
        samp = self.sample(2000)
        return kendalltau(samp[:, 0], samp[:, 1]).correlation

    def LTDC(self, param=None):
        """
        Lower‐tail dependence λ_L = 0 (no tail dependence).
        """
        return 0.0

    def UTDC(self, param=None):
        """
        Upper‐tail dependence λ_U = 0 (no tail dependence).
        """
        return 0.0

    def IAD(self, data):
        """
        Integrated Anderson–Darling not implemented.
        """
        print(f"[INFO] IAD not implemented for {self.name}.")
        return np.nan

    def AD(self, data):
        """
        Anderson–Darling not implemented.
        """
        print(f"[INFO] AD not implemented for {self.name}.")
        return np.nan
