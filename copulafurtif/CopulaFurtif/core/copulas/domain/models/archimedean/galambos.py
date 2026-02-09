"""
Galambos Copula implementation.

The Galambos copula is an extreme value copula used to model upper tail dependence.
It is asymmetric and capable of capturing strong dependence in the upper tail,
making it suitable for risk management and dependence modeling in extremes.

Attributes:
    name (str): Human-readable name of the copula.
    type (str): Identifier for the copula type.
    bounds_param (list of tuple): Bounds for the copula parameter [theta] ∈ (0.01, 10.0).
    parameters (np.ndarray): Copula parameter [theta].
    default_optim_method (str): Default optimization method used during fitting.
"""

import numpy as np
from scipy.stats import kendalltau
from scipy.optimize import root_scalar

from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel, CopulaParameters
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence
from numpy.random import default_rng


class GalambosCopula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """Galambos Copula model."""

    def __init__(self):
        """Initialize the Galambos copula with default parameters and bounds."""
        super().__init__()
        self.name = "Galambos Copula"
        self.type = "galambos"
        self.default_optim_method = "SLSQP"
        self.init_parameters(CopulaParameters([1.5], [(1e-4, 50.0)], ["delta"]))

    # ---------- helpers ----------
    @staticmethod
    def _xyz(u, v, d, eps=1e-10):
        u = np.clip(np.asarray(u, float), eps, 1.0 - eps)
        v = np.clip(np.asarray(v, float), eps, 1.0 - eps)

        x = -np.log(u)
        y = -np.log(v)

        # avoid x=0 or y=0 leading to 0**(-d)
        x = np.maximum(x, eps)
        y = np.maximum(y, eps)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            S = x ** (-d) + y ** (-d)
        return x, y, S

    def get_cdf(self, u, v, param=None):
        """Compute the copula CDF C(u, v).

        Args:
            u (float or np.ndarray): First input in (0, 1).
            v (float or np.ndarray): Second input in (0, 1).
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float or np.ndarray: Value(s) of the CDF.
        """
        d = float(self.get_parameters()[0]) if param is None else float(param[0])
        _, _, S = self._xyz(u, v, d, eps=1e-10)

        u = np.clip(np.asarray(u, float), 1e-10, 1.0 - 1e-10)
        v = np.clip(np.asarray(v, float), 1e-10, 1.0 - 1e-10)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            A = S ** (-1.0 / d)
            out = u * v * np.exp(A)

        return np.clip(out, 0.0, 1.0)

    def get_pdf(self, u, v, param=None):
        """
        Compute the Galambos copula PDF c(u, v; δ).

        Parameters
        ----------
        u : array_like
            First marginal argument, must lie in (0,1).
        v : array_like
            Second marginal argument, must lie in (0,1).
        param : sequence, optional
            If provided, overrides self.get_parameters()[0] as δ.

        Returns
        -------
        pdf : ndarray
            The value(s) of the Galambos copula PDF at (u, v).
        """
        # dependence parameter δ
        d = float(self.get_parameters()[0]) if param is None else float(param[0])

        # guard against log(0) -- clip *very* lightly; RNG draws are in [0,1)
        eps = 1e-10
        u = np.clip(np.asarray(u, float), eps, 1.0 - eps)
        v = np.clip(np.asarray(v, float), eps, 1.0 - eps)

        x, y, S = self._xyz(u, v, d, eps=eps)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            A = S ** (-1.0 / d)
            term2 = S ** (-1.0 - 1.0 / d) * (x ** (-d - 1) + y ** (-d - 1))
            term3 = S ** (-2.0 - 1.0 / d) * (x * y) ** (-d - 1) * (1.0 + d + A)
            g = 1.0 - term2 + term3
            pdf = np.exp(A) * g

        return np.clip(pdf, 1e-300, np.inf)

    def sample(self,
               n: int,
               param=None,
               rng=None,
               eps: float = 1e-12,
               max_iter: int = 40) -> np.ndarray:
        """
        Draw *n* i.i.d. pairs (U, V) from the Galambos copula via conditional
        inversion.

        Algorithm
        ---------
        1.  U ~ Unif(0, 1)                    (draw once for all).
        2.  P ~ Unif(0, 1)                    (target prob each row).
        3.  For each U, solve F_{V|U}(v|U)=P  by vectorised bisection on v∈(0,1).

            The monotone function is  self.partial_derivative_C_wrt_u(U, v).

        Parameters
        ----------
        n        : int
            Number of samples.
        param    : sequence-like, optional
            Copula parameter ``[δ]``.  If *None* uses current parameters.
        rng      : numpy.random.Generator, optional
        eps      : float
            Hard clip to keep both margins in the open unit interval.
        max_iter : int
            Bisection iterations (2^{-max_iter} absolute tolerance).

        Returns
        -------
        ndarray, shape (n, 2)
            Columns are U and V.
        """
        # ------------------------------------------------------------------ RNG + δ
        if rng is None:
            rng = default_rng()

        δ = float(self.get_parameters()[0]) if param is None else float(param[0])
        if δ <= 0.0:
            raise ValueError("Galambos parameter δ must be positive.")

        # independence shortcut (δ→0) ---------------------------------------------
        if δ < 1e-8:
            uv = rng.random((n, 2))
            np.clip(uv, eps, 1.0 - eps, out=uv)
            return uv

        # -------------------------------------------------------------------- draws
        u = rng.random(n)
        p = rng.random(n)

        # ---------------------------------------------------------------- bisection
        lo = np.full_like(u, eps)  # lower bound v ≈ 0+
        hi = np.full_like(u, 1.0 - eps)  # upper bound v ≈ 1−

        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            cdf_mid = self.partial_derivative_C_wrt_u(u, mid, [δ])

            greater = cdf_mid > p  # still above target → need larger v
            hi[greater] = mid[greater]

            smaller = ~greater  # below target → elevate lower bound
            lo[smaller] = mid[smaller]

        v = 0.5 * (lo + hi)

        # ------------------------------------------------------------------ return
        np.clip(u, eps, 1.0 - eps, out=u)
        np.clip(v, eps, 1.0 - eps, out=v)

        return np.column_stack((u, v))

    def kendall_tau(self, param=None, method="mc", n=200000, rng=None):
        """
        Monte Carlo Kendall's tau for Galambos (no simple closed form).
        """
        from scipy.stats import kendalltau
        if param is None:
            d = float(self.get_parameters()[0])
        else:
            d = float(param[0])

        if rng is None:
            rng = default_rng()

        uv = self.sample(n, param=[d], rng=rng)
        tau, _ = kendalltau(uv[:, 0], uv[:, 1])
        return float(tau)


    def LTDC(self, param=None):
        """Lower tail dependence coefficient (always 0 for Galambos copula).

        Args:
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float: 0.0
        """
        return 0

    def UTDC(self, param=None):
        """Upper tail dependence coefficient for the Galambos copula.

        Args:
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float. λ_U = 2^{ -1/δ }
        """
        d = float(self.get_parameters()[0]) if param is None else float(param[0])
        return 2.0 ** (-1.0 / d)

    def IAD(self, data):
        """Integrated Absolute Deviation (disabled for Galambos copula).

        Args:
            data (array-like): Input data (unused).

        Returns:
            float: NaN.
        """
        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        """Anderson–Darling test statistic (disabled for Galambos copula).

        Args:
            data (array-like): Input data (unused).

        Returns:
            float: NaN.
        """
        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """
        ∂C(u,v)/∂u  ==  F_{V|U}(v | u)   for the Galambos copula.

        Formula (Joe 2014, §4.9.1):

            x = -log u,   y = -log v,
            r = (x / y)^δ,
            S = x^{-δ} + y^{-δ}

            ∂C/∂u = v · exp(S^{-1/δ}) · [ 1 − (1 + r)^{−1−1/δ} ]   ∈ (0,1)
        """
        d = float(self.get_parameters()[0]) if param is None else float(param[0])

        u = np.clip(np.asarray(u, float), 1e-10, 1.0 - 1e-10)
        v = np.clip(np.asarray(v, float), 1e-10, 1.0 - 1e-10)

        x, y, S = self._xyz(u, v, d, eps=1e-10)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            ratio = (x / y) ** d
            out = v * np.exp(S ** (-1.0 / d)) * (1.0 - (1.0 + ratio) ** (-1.0 - 1.0 / d))

        return np.clip(out, 1e-12, 1.0 - 1e-12)

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """Compute ∂C(u,v)/∂v via symmetry.

        Args:
            u (float or np.ndarray): U value(s).
            v (float or np.ndarray): V value(s).
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float or np.ndarray: Partial derivative values.
        """
        return self.partial_derivative_C_wrt_u(v, u, param)

    def blomqvist_beta(self, param=None):
        """
        Compute Blomqvist's beta (theoretical) for the Galambos copula.

        Formula
        -------
        β(delta) = (4/delta) * log( (2*exp(-delta/2) - 2*exp(-delta)) / (1 - exp(-delta)) ) - 1

        Reference
        ---------
        Nelsen (2006), "An Introduction to Copulas", Springer.

        Parameters
        ----------
        param : np.ndarray, optional
            Copula parameter [delta]. If None, uses the current parameters.

        Returns
        -------
        float
            Theoretical Blomqvist's beta.
        """
        if param is None:
            param = self.get_parameters()
        delta = float(param[0])
        if delta < 1e-8:
            return 0.0
        # β = 2^{ 1 / 2^{1/δ} } - 1  == exp( ln 2 / 2^{1/δ} ) - 1
        return float(np.exp(np.log(2.0) / (2.0 ** (1.0 / delta))) - 1.0)

    def init_from_data(self, u, v):
        """
        Robust initialization of Galambos copula parameter δ from data.

        Strategy
        --------
        - Compute empirical Kendall's tau (τ̂).
        - Compute empirical Blomqvist's beta (β̂).
        - If τ̂ is strong enough (>0.05) → use closed-form inversion of τ(δ).
        - Otherwise → invert β numerically via root-finding.
        - Clip δ into model bounds.

        Parameters
        ----------
        u : array-like
            Pseudo-observations in (0,1).
        v : array-like
            Pseudo-observations in (0,1).

        Returns
        -------
        float
            Initial guess δ₀ for MLE fitting.
        """

        u = np.asarray(u)
        v = np.asarray(v)

        # 1) empirical λ_U
        qs = (0.90, 0.92, 0.94, 0.96, 0.98)
        vals = []
        for q in qs:
            uq, vq = np.quantile(u, q), np.quantile(v, q)
            joint = np.mean((u > uq) & (v > vq))
            denom = max(1e-9, 1.0 - q)
            vals.append(joint / denom)
        lamU_emp = float(np.median(vals))
        lamU_emp = float(np.clip(lamU_emp, 1e-9, 0.999))  # guards

        # 2) try inversion from λ_U
        delta0 = None
        if 0 < lamU_emp < 1:
            delta0 = -1.0 / (np.log(lamU_emp) / np.log(2.0))

        # 3) fallback via β if δ0 not valid
        if delta0 is None or not np.isfinite(delta0) or delta0 <= 0:
            same_half = np.mean(((u > 0.5) & (v > 0.5)) |
                                ((u <= 0.5) & (v <= 0.5)))
            beta_emp = 2.0 * same_half - 1.0
            if beta_emp > -0.99:  # safe guard
                s = np.log2(max(1.0 + beta_emp, 1e-9))
                if s > 0:
                    delta0 = 1.0 / np.log2(1.0 / s)
                else:
                    delta0 = 1e-3
            else:
                delta0 = 1e-3

        # 4) clip to bounds
        low, high = self.get_bounds()[0]
        delta0 = float(np.clip(delta0, low, high))
        return np.array([delta0], dtype=float)
