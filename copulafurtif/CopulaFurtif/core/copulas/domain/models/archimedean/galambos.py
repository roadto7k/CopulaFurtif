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
        # self.bounds_param = [(0.01, 10.0)]  # [theta]
        # self.param_names = ["theta"]
        # self.parameters = [1.5]
        self.default_optim_method = "SLSQP"
        self.init_parameters(CopulaParameters([1.5], [(1e-4, 50.0)], ["delta"]))

    # ---------- helpers ----------
    @staticmethod
    def _xyz(u, v, d):
        x = -np.log(u)
        y = -np.log(v)
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
        _, _, S = self._xyz(u, v, d)
        return u * v * np.exp(S ** (-1.0 / d))

    def get_pdf(self, u, v, param=None):
        """Compute the copula PDF c(u, v).

        Args:
            u (float or np.ndarray): First input in (0, 1).
            v (float or np.ndarray): Second input in (0, 1).
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float or np.ndarray: Value(s) of the PDF.
        """
        d = float(self.get_parameters()[0]) if param is None else float(param[0])
        x = -np.log(u)
        y = -np.log(v)
        S = x ** (-d) + y ** (-d)

        C = u * v * np.exp(S ** (-1.0 / d))  # the copula itself
        g = (x * y) ** (-d) * S ** (-2.0 - 1.0 / d) * (d + 1) \
            - (x ** (-d - 1) + y ** (-d - 1)) * S ** (-1.0 - 1.0 / d) \
            + 1.0
        return C * g / (u * v)

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

    def kendall_tau(self, param=None):
        """Compute Kendall's tau for the Galambos copula.

        Args:
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float: Kendall's tau.
        """
        d = float(self.get_parameters()[0]) if param is None else float(param[0])
        return d / (d + 2.0)

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
        x, y, S = self._xyz(u, v, d)

        ratio = (x / y) ** d
        return v * np.exp(S ** (-1.0 / d)) * (1.0 - (1.0 + ratio) ** (-1.0 - 1.0 / d))

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

    def conditional_cdf_u_given_v(self, u, v, param=None):
        """Compute conditional CDF P(U ≤ u | V = v).

        Args:
            u (float or np.ndarray): U values.
            v (float or np.ndarray): V values.
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float or np.ndarray: Conditional CDF values.
        """
        return self.partial_derivative_C_wrt_v(u, v, param)

    def conditional_cdf_v_given_u(self, u, v, param=None):
        """Compute conditional CDF P(V ≤ v | U = u).

        Args:
            u (float or np.ndarray): U values.
            v (float or np.ndarray): V values.
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float or np.ndarray: Conditional CDF values.
        """
        return self.partial_derivative_C_wrt_u(u, v, param)
