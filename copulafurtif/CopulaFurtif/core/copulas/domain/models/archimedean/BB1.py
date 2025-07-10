import numpy as np
from scipy.special import beta
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel, CopulaParameters
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence
from numpy.random import default_rng


class BB1Copula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """
    BB1 Copula (Two-parameter Archimedean copula).

    The BB1 copula extends Clayton and Gumbel copulas with two parameters: 
    one for dependence strength (theta > 0) and one for tail dependence (delta >= 1).

    Attributes:
        name (str): Human-readable name of the copula.
        type (str): Identifier for the copula family.
        bounds_param (list of tuple): Bounds for parameters [theta, delta].
        parameters (np.ndarray): Current copula parameters.
        default_optim_method (str): Default optimization method.
    """

    def __init__(self):
        """Initialize BB1 copula with default parameters."""
        super().__init__()
        self.name = "BB1 Copula"
        self.type = "bb1"
        # self.bounds_param = [(1e-6, np.inf), (1.0, np.inf)]  # [theta, delta]
        # self.param_names = ["theta", "delta"]
        # self.parameters = [0.5, 1.5]
        self.default_optim_method = "Powell"
        self.init_parameters(CopulaParameters([0.5, 1.5],[(1e-6, np.inf), (1.0, np.inf)], ["theta", "delta"] ))

    def get_cdf(self, u, v, param=None):
        """
        Compute the BB1 copula CDF C(u, v).

        Args:
            u (float or np.ndarray): First input in (0,1).
            v (float or np.ndarray): Second input in (0,1).
            param (np.ndarray, optional): Copula parameters [theta, delta].

        Returns:
            float or np.ndarray: Copula CDF values.
        """
        if param is None:
            param = self.get_parameters()
        theta, delta = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        term1 = (u ** (-theta) - 1) ** delta
        term2 = (v ** (-theta) - 1) ** delta
        inner = term1 + term2
        return (1 + inner ** (1.0 / delta)) ** (-1.0 / theta)

    def get_pdf(self, u, v, param=None):
        """
        Compute the BB1 copula PDF c(u, v).

        Args:
            u (float or np.ndarray): First input in (0,1).
            v (float or np.ndarray): Second input in (0,1).
            param (np.ndarray, optional): Copula parameters [theta, delta].

        Returns:
            float or np.ndarray: Copula PDF values.
        """
        if param is None:
            param = self.get_parameters()
        theta, delta = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        x = (u ** (-theta) - 1) ** delta
        y = (v ** (-theta) - 1) ** delta
        S = x + y
        term1 = (1 + S ** (1.0 / delta)) ** (-1.0 / theta - 2)
        term2 = S ** (1.0 / delta - 2)
        term3 = theta * (delta - 1) + (theta * delta + 1) * S ** (1.0 / delta)
        term4 = (x * y) ** (1 - 1.0 / delta) * (u * v) ** (-theta - 1)
        return term1 * term2 * term3 * term4

    def kendall_tau(self, param=None):
        """
        Closed-form Kendall's tau for the BB1 copula:

        τ(θ, δ) = 1 − 2 / [ δ (θ + 2) ]      (Joe 1997, Eq. 4.32)
        """
        if param is None:
            param = self.get_parameters()
        theta, delta = map(float, param)
        return 1.0 - 2.0 / (delta * (theta + 2.0))

    def _invert_conditional_v(self, u, target_cdf, theta, delta,
                              eps=1e-12, max_iter=40):
        """
        Solve for v in (0,1) such that

            F_{V|U}(v | u) = ∂C(u,v)/∂u = target_cdf

        by vectorized bisection.
        """
        lo = np.full_like(target_cdf, eps)
        hi = np.full_like(target_cdf, 1.0 - eps)

        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            cdf_mid = self.partial_derivative_C_wrt_u(u, mid, [theta, delta])
            # when CDF(mid) > target, decrease upper bound
            hi = np.where(cdf_mid > target_cdf, mid, hi)
            # when CDF(mid) <= target, increase lower bound
            lo = np.where(cdf_mid <= target_cdf, mid, lo)

        return 0.5 * (lo + hi)

    def sample(self,
               n: int,
               param=None,
               rng=None,
               eps: float = 1e-12,
               max_iter: int = 40) -> np.ndarray:
        """
        Generate n i.i.d. pairs (U, V) from the BB1 copula by conditional inversion.

        Parameters
        ----------
        n        : int
            Number of samples.
        param    : sequence-like, optional
            Copula parameters [theta, delta].  If None, uses current parameters.
        rng      : numpy.random.Generator, optional
            Random number generator for reproducibility.
        eps      : float
            Small epsilon to keep values in (0,1).
        max_iter : int
            Maximum number of bisection iterations.

        Returns
        -------
        ndarray of shape (n, 2)
            Columns are U and V.
        """
        if rng is None:
            rng = default_rng()

        if param is None:
            theta, delta = map(float, self.get_parameters())
        else:
            theta, delta = map(float, param)

        # 1) draw U uniform and target probabilities P
        u = rng.random(n)
        p = rng.random(n)

        # 2) invert conditional CDF ∂C/∂u to obtain V
        v = self._invert_conditional_v(u, p, theta, delta,
                                       eps=eps, max_iter=max_iter)

        # 3) final clipping and pack
        np.clip(u, eps, 1.0 - eps, out=u)
        np.clip(v, eps, 1.0 - eps, out=v)
        return np.column_stack((u, v))

    def LTDC(self, param=None):
        """
        Compute lower tail dependence coefficient.

        Args:
            param (np.ndarray, optional): Copula parameters [theta, delta].

        Returns:
            float: Lower tail dependence.
        """
        if param is None:
            param = self.get_parameters()
        theta, delta = param
        return 2.0 ** (-1.0 / (delta * theta))

    def UTDC(self, param=None):
        """
        Compute upper tail dependence coefficient.

        Args:
            param (np.ndarray, optional): Copula parameters [theta, delta].

        Returns:
            float: Upper tail dependence.
        """
        if param is None:
            param = self.get_parameters()
        delta = param[1]
        return 2.0 - 2.0 ** (1.0 / delta)

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """
        Compute partial derivative ∂C/∂u.

        Args:
            u (float or np.ndarray): U values.
            v (float or np.ndarray): V values.
            param (np.ndarray, optional): Copula parameters [theta, delta].

        Returns:
            float or np.ndarray: Partial derivative values.
        """
        if param is None:
            param = self.get_parameters()
        theta, delta = param
        T = (u ** (-theta) - 1) ** delta + (v ** (-theta) - 1) ** delta
        factor = (1 + T ** (1.0 / delta)) ** (-1.0 / theta - 1)
        return factor * T ** (1.0 / delta - 1) * (u ** (-theta) - 1) ** (delta - 1) * u ** (-theta - 1)

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """
        Compute partial derivative ∂C/∂v.

        Args:
            u (float or np.ndarray): U values.
            v (float or np.ndarray): V values.
            param (np.ndarray, optional): Copula parameters [theta, delta].

        Returns:
            float or np.ndarray: Partial derivative values.
        """
        return self.partial_derivative_C_wrt_u(v, u, param)

    def IAD(self, data):
        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan
