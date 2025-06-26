"""
Tawn Copula implementation.

The Tawn copula is an asymmetric copula used to model both upper tail dependence
and asymmetry in joint distributions. It generalizes the Gumbel copula by introducing
a second parameter (delta) that controls asymmetry.

Attributes:
    name (str): Human-readable name of the copula.
    type (str): Identifier for the copula type.
    bounds_param (list of tuple): Bounds for the copula parameters [theta, delta].
    parameters (np.ndarray): Copula parameters [theta, delta].
    default_optim_method (str): Optimization method used for fitting.
"""

import numpy as np
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel, CopulaParameters
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


class TawnCopula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """Tawn Copula model."""

    def __init__(self):
        """Initialize the Tawn copula with default parameters and bounds."""
        super().__init__()
        self.name = "Tawn Copula"
        self.type = "tawn"
        self.bounds_param = [(1.01, 5.0), (0.0, 1.0)]  # [theta, delta]
        self.param_names = ["theta", "delta"]
        # self.parameters = [2.0, 0.5]
        self.default_optim_method = "SLSQP"
        self.init_parameters(CopulaParameters([2.0, 0.5],[(1.01, 5.0), (0.0, 1.0)] , ["theta", "delta"]))

    def get_cdf(self, u, v, param=None):
        """Compute the copula CDF C(u, v).

        Args:
            u (float or np.ndarray): U values in (0,1).
            v (float or np.ndarray): V values in (0,1).
            param (np.ndarray, optional): Copula parameters [theta, delta].

        Returns:
            float or np.ndarray: CDF value(s).
        """
        if param is None:
            param = self.get_parameters()
        theta, delta = param
        x = -np.log(u)
        y = -np.log(v)
        s = x + y
        w = (1 - delta) * (x / s) ** theta + delta * (y / s) ** theta
        return np.exp(-s * w ** (1 / theta))

    def get_pdf(self, u, v, param=None):
        """Compute the copula PDF c(u, v).

        Note:
            Analytical expression for the PDF is complex and currently not implemented.

        Args:
            u (float or np.ndarray): U values.
            v (float or np.ndarray): V values.
            param (np.ndarray, optional): Copula parameters.

        Returns:
            float or np.ndarray: Placeholder value (1.0).
        """
        if param is None:
            param = self.get_parameters()
        return np.ones_like(u)

    def sample(self, n, param=None):
        """Generate random samples from the Tawn copula (placeholder).

        Args:
            n (int): Number of samples to generate.
            param (np.ndarray, optional): Copula parameters.

        Returns:
            np.ndarray: Samples of shape (n, 2).
        """
        if param is None:
            param = self.get_parameters()
        u = np.random.rand(n)
        v = np.random.rand(n)
        return np.column_stack((u, v))  # NOTE: Not exact sampling

    def kendall_tau(self, param=None):
        """Compute Kendall's tau for the Tawn copula.

        Args:
            param (np.ndarray, optional): Copula parameters [theta, delta].

        Returns:
            float: Kendall's tau.
        """
        if param is None:
            param = self.get_parameters()
        theta, delta = param
        return (theta * (1 - delta + delta)) / (theta + 2)

    def LTDC(self, param=None):
        """Lower tail dependence coefficient (0 for Tawn copula).

        Args:
            param (np.ndarray, optional): Copula parameters.

        Returns:
            float: 0.0
        """
        return 0.0

    def UTDC(self, param=None):
        """Upper tail dependence coefficient for the Tawn copula.

        Args:
            param (np.ndarray, optional): Copula parameters [theta, delta].

        Returns:
            float: UTDC value.
        """
        if param is None:
            param = self.get_parameters()
        theta, delta = param
        return 2 - 2 ** (1 / theta)

    def IAD(self, data):
        """Integrated Absolute Deviation (disabled for Tawn copula).

        Args:
            data (array-like): Input data (unused).

        Returns:
            float: NaN.
        """
        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        """Anderson–Darling test statistic (disabled for Tawn copula).

        Args:
            data (array-like): Input data (unused).

        Returns:
            float: NaN.
        """
        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """Compute ∂C(u,v)/∂u (placeholder implementation).

        Args:
            u (float or np.ndarray): U values.
            v (float or np.ndarray): V values.
            param (np.ndarray, optional): Copula parameters.

        Returns:
            float or np.ndarray: Placeholder (1.0).
        """
        if param is None:
            param = self.get_parameters()
        theta, delta = param
        return np.ones_like(u)

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """Compute ∂C(u,v)/∂v via symmetry (placeholder).

        Args:
            u (float or np.ndarray): U values.
            v (float or np.ndarray): V values.
            param (np.ndarray, optional): Copula parameters.

        Returns:
            float or np.ndarray: Placeholder (1.0).
        """
        return self.partial_derivative_C_wrt_u(v, u, param)

    def conditional_cdf_u_given_v(self, u, v, param=None):
        """Compute conditional CDF P(U ≤ u | V = v) (placeholder).

        Args:
            u (float or np.ndarray): U values.
            v (float or np.ndarray): V values.
            param (np.ndarray, optional): Copula parameters.

        Returns:
            float or np.ndarray: Placeholder (1.0).
        """
        return self.partial_derivative_C_wrt_v(u, v, param)

    def conditional_cdf_v_given_u(self, u, v, param=None):
        """Compute conditional CDF P(V ≤ v | U = u) (placeholder).

        Args:
            u (float or np.ndarray): U values.
            v (float or np.ndarray): V values.
            param (np.ndarray, optional): Copula parameters.

        Returns:
            float or np.ndarray: Placeholder (1.0).
        """
        return self.partial_derivative_C_wrt_u(u, v, param)
