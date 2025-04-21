import numpy as np
from Service.Copulas.base import BaseCopula


class EmpiricalCopula(BaseCopula):
    """
    Empirical Copula built from pseudo‐observations { (u_i, v_i) }.

    Attributes
    ----------
    family : str
        Identifier for the copula family. Here, "empirical".
    name : str
        Human-readable name for output/logging.
    U, V : np.ndarray
        Arrays of marginal uniforms in [0,1].
    n : int
        Number of observations.
    """

    def __init__(self, U, V):
        """
        Store the data and set up the copula.

        Parameters
        ----------
        U, V : array-like, shape (n,)
            Pseudo‐observations in [0,1].
        """
        super().__init__()
        self.type = "empirical"
        self.name = "Empirical Copula"
        self.U = np.asarray(U).ravel()
        self.V = np.asarray(V).ravel()
        if self.U.shape != self.V.shape:
            raise ValueError("U and V must have the same length")
        self.n = len(self.U)
        # no parameters to estimate
        self.bounds_param = []
        self.parameters = np.array([])

    def get_cdf(self, u, v, param=None):
        """
        Empirical CDF C_n(u,v) = (1/n) ∑_{i=1}^n 1{U_i ≤ u, V_i ≤ v}.

        Parameters
        ----------
        u, v : scalar or array‐like in [0,1]
            Points at which to evaluate the empirical copula.
        param : ignored

        Returns
        -------
        float or np.ndarray
            Empirical copula values.
        """
        u_arr = np.atleast_1d(u)
        v_arr = np.atleast_1d(v)
        # vectorized counting
        C = np.empty_like(u_arr, dtype=float)
        for idx, (uu, vv) in enumerate(zip(u_arr, v_arr)):
            C[idx] = np.count_nonzero((self.U <= uu) & (self.V <= vv)) / self.n
        return C if C.shape != () else C.item()

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """
        ∂C_n/∂u approximated by finite differences:
          [C(u+ε, v) − C(u−ε, v)] / (2ε).
        """
        eps = 1e-6
        return (self.get_cdf(u + eps, v) - self.get_cdf(u - eps, v)) / (2 * eps)

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """
        ∂C_n/∂v approximated by finite differences:
          [C(u, v+ε) − C(u, v−ε)] / (2ε).
        """
        eps = 1e-6
        return (self.get_cdf(u, v + eps) - self.get_cdf(u, v - eps)) / (2 * eps)

    def conditional_cdf_v_given_u(self, u, v, param=None):
        """
        Empirical P(V ≤ v | U = u) = C_n(u, v) / F_U(u),
        where F_U(u) = (1/n) ∑ 1{U_i ≤ u}.
        """
        Fu = np.count_nonzero(self.U <= u) / self.n
        return self.get_cdf(u, v) / Fu if Fu > 0 else 0.0

    def conditional_cdf_u_given_v(self, u, v, param=None):
        """
        Empirical P(U ≤ u | V = v) = C_n(u, v) / F_V(v).
        """
        Fv = np.count_nonzero(self.V <= v) / self.n
        return self.get_cdf(u, v) / Fv if Fv > 0 else 0.0

    def sample(self, m, param=None):
        """
        Sample m points by bootstrap‐resampling the observed pairs.

        Parameters
        ----------
        m : int
            Number of draws.

        Returns
        -------
        np.ndarray, shape (m,2)
            Pseudo‐observations drawn with replacement.
        """
        idx = np.random.randint(0, self.n, size=m)
        return np.column_stack((self.U[idx], self.V[idx]))

    def kendall_tau(self, param=None):
        """
        Empirical Kendall's τ:
          τ = [# concordant − # discordant] / [n(n−1)/2].
        Computed in O(n²), may be slow for large n.
        """
        n = self.n
        concordant = discordant = 0
        for i in range(n - 1):
            ui, vi = self.U[i], self.V[i]
            # compare to later points
            du = self.U[i+1:] - ui
            dv = self.V[i+1:] - vi
            prod = du * dv
            concordant += np.count_nonzero(prod > 0)
            discordant += np.count_nonzero(prod < 0)
        denom = n * (n - 1) / 2
        return (concordant - discordant) / denom

    def LTDC(self, param=None):
        """
        Empirical lower‐tail dependence:
          λ_L = lim_{u→0} C_n(u,u)/u.
        """
        # approximate at u = 1/n
        u = 1.0 / self.n
        return self.get_cdf(u, u) / u

    def UTDC(self, param=None):
        """
        Empirical upper‐tail dependence:
          λ_U = lim_{u→1} [1 − 2u + C_n(u,u)] / (1−u).
        """
        # approximate at u = (n−1)/n
        u = (self.n - 1.0) / self.n
        return (1 - 2*u + self.get_cdf(u, u)) / (1 - u)

    def IAD(self, data):
        """
        Integrated Anderson–Darling not implemented for empirical copula.
        """
        raise NotImplementedError("IAD not available for empirical copula.")

    def AD(self, data):
        """
        Anderson–Darling not implemented for empirical copula.
        """
        raise NotImplementedError("AD not available for empirical copula.")
