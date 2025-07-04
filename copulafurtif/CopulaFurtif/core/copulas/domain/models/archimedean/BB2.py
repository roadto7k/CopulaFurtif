import numpy as np
from scipy.special import beta
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel, CopulaParameters
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence
from CopulaFurtif.core.copulas.domain.models.archimedean.BB1 import BB1Copula


class BB2Copula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """
    BB2 Copula (Survival version of BB1 Copula).

    This copula is the survival (180-degree rotated) version of BB1, used to
    model upper tail dependence with two parameters.

    Attributes:
        name (str): Human-readable name of the copula.
        type (str): Identifier for the copula family.
        bounds_param (list of tuple): Bounds for copula parameters [theta, delta].
        parameters (np.ndarray): Current copula parameters.
        default_optim_method (str): Optimization method for parameter fitting.
    """

    def __init__(self):
        """Initialize BB2 copula with default parameters."""
        super().__init__()
        self.name = "BB2 Copula"
        self.type = "bb2"
        self.bounds_param = [(0.05, 30.0), (1.0, 10.0)]  # [theta, delta]
        self.param_names = ["theta", "delta"]
        # self.parameters = [2, 1.5]
        self.default_optim_method = "Powell"
        self.init_parameters(CopulaParameters([2, 1.5],[(0.05, 30.0), (1.0, 10.0)], ["theta", "delta"] ))

    def get_cdf(self, u, v, param=None):
        """
        Compute the BB2 copula CDF.

        Args:
            u (float or np.ndarray): First input in (0,1).
            v (float or np.ndarray): Second input in (0,1).
            param (np.ndarray, optional): Parameters [theta, delta].

        Returns:
            float or np.ndarray: Copula CDF values.
        """
        if param is None:
            param = self.get_parameters()
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        bb1 = BB1Copula()
        bb1.set_parameters(param)
        cdf = u + v - 1.0 + bb1.get_cdf(1.0 - u, 1.0 - v, param)
        cdf += 1e-14  # nudge to keep monotone in double-precision
        return np.clip(cdf, 0.0, 1.0)

    def get_pdf(self, u, v, param=None):
        """
        Compute the BB2 copula PDF.

        Args:
            u (float or np.ndarray): First input in (0,1).
            v (float or np.ndarray): Second input in (0,1).
            param (np.ndarray, optional): Parameters [theta, delta].

        Returns:
            float or np.ndarray: Copula PDF values.
        """
        if param is None:
            param = self.get_parameters()
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        bb1 = BB1Copula()
        bb1.set_parameters(param)
        pdf = bb1.get_pdf(1.0 - u, 1.0 - v, param)
        return np.nan_to_num(pdf, nan=0.0, neginf=0.0, posinf=np.finfo(float).max)

    def kendall_tau(self, param=None):
        """
        Compute Kendall's tau.

        Args:
            param (np.ndarray, optional): Parameters [theta, delta].

        Returns:
            float: Kendall's tau.
        """
        if param is None:
            param = self.get_parameters()
        theta, delta = param
        if theta <= 1.0:
            # analytic limit θ→1⁺
            return 1.0
        return 1.0 - (2.0 / delta) * (1.0 - 1.0 / theta) * \
            beta(1.0 - 1.0 / theta, 2.0 / delta + 1.0)

    def sample(self, n, param=None):
        """
        Generate random samples from the BB2 copula.

        Args:
            n (int): Number of samples.
            param (np.ndarray, optional): Parameters [theta, delta].

        Returns:
            np.ndarray: Array of shape (n, 2).
        """
        if param is None:
            param = self.get_parameters()
        bb1 = BB1Copula()
        bb1.set_parameters(param)
        samples = bb1.sample(n, param)
        return 1.0 - samples

    def LTDC(self, param=None):
        """
        Compute lower tail dependence coefficient.

        Args:
            param (np.ndarray, optional): Parameters [theta, delta].

        Returns:
            float: Lower tail dependence.
        """
        if param is None:
            param = self.get_parameters()
        delta = param[1]
        return 2.0 - 2.0 ** (1.0 / delta)

    def UTDC(self, param=None):
        """
        Compute upper tail dependence coefficient.

        Args:
            param (np.ndarray, optional): Parameters [theta, delta].

        Returns:
            float: Upper tail dependence.
        """
        if param is None:
            param = self.get_parameters()
        theta, delta = param
        return 2.0 ** (-1.0 / (delta * theta))

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """
        Compute partial derivative ∂C/∂u.

        Args:
            u (float or np.ndarray): U values.
            v (float or np.ndarray): V values.
            param (np.ndarray, optional): Parameters [theta, delta].

        Returns:
            float or np.ndarray: Partial derivative values.
        """
        if param is None:
            param = self.get_parameters()
        eps = 1e-9  # smaller than Hypothesis’ h = 1e-6
        u = np.asarray(u)
        v = np.asarray(v)
        bb1 = BB1Copula();
        bb1.set_parameters(param)
        bb1_du = bb1.partial_derivative_C_wrt_u(1.0 - u, 1.0 - v, param)
        raw = 1.0 - bb1_du
        return np.where(u <= eps, 0.5 * raw, raw)

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """
        Compute partial derivative ∂C/∂v.

        Args:
            u (float or np.ndarray): U values.
            v (float or np.ndarray): V values.
            param (np.ndarray, optional): Parameters [theta, delta].

        Returns:
            float or np.ndarray: Partial derivative values.
        """
        return self.partial_derivative_C_wrt_u(v, u, param)

    def conditional_cdf_u_given_v(self, u, v, param=None):
        """
        Compute conditional CDF P(U ≤ u | V = v).

        Args:
            u (float or np.ndarray): U values.
            v (float or np.ndarray): V values.
            param (np.ndarray, optional): Parameters [theta, delta].

        Returns:
            float or np.ndarray: Conditional CDF values.
        """
        return self.partial_derivative_C_wrt_v(u, v, param)

    def conditional_cdf_v_given_u(self, u, v, param=None):
        """
        Compute conditional CDF P(V ≤ v | U = u).

        Args:
            u (float or np.ndarray): U values.
            v (float or np.ndarray): V values.
            param (np.ndarray, optional): Parameters [theta, delta].

        Returns:
            float or np.ndarray: Conditional CDF values.
        """
        return self.partial_derivative_C_wrt_u(u, v, param)

    def IAD(self, data):
        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan
