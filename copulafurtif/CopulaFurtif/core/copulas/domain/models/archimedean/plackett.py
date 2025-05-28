"""
Plackett Copula implementation.

The Plackett copula is a symmetric copula that allows for modeling both positive 
and negative dependence while not exhibiting tail dependence. It is particularly 
useful due to its simplicity and flexibility with a single parameter.

Attributes:
    name (str): Human-readable name of the copula.
    type (str): Identifier for the copula type.
    bounds_param (list of tuple): Bounds for the copula parameter [theta] ∈ (0.01, 100.0).
    parameters (np.ndarray): Copula parameter [theta].
    default_optim_method (str): Optimization method used during fitting.
"""

import numpy as np
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


class PlackettCopula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """Plackett Copula model."""

    def __init__(self):
        """Initialize the Plackett copula with default parameters and bounds."""
        super().__init__()
        self.name = "Plackett Copula"
        self.type = "plackett"
        self.bounds_param = [(0.01, 100.0)]  # [theta]
        self.param_names = ["theta"]
        self.parameters = [2.0]
        self.default_optim_method = "SLSQP"

    def get_cdf(self, u, v, param=None):
        """Compute the copula CDF C(u, v).

        Args:
            u (float or np.ndarray): First input in (0, 1).
            v (float or np.ndarray): Second input in (0, 1).
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float or np.ndarray: CDF value(s).
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        a = theta - 1
        b = 1 + a * (u + v)
        c = np.sqrt(b ** 2 - 4 * theta * a * u * v)
        return (2 * theta * u * v) / (b + c)

    def get_pdf(self, u, v, param=None):
        """Compute the copula PDF c(u, v).

        Args:
            u (float or np.ndarray): First input in (0, 1).
            v (float or np.ndarray): Second input in (0, 1).
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float or np.ndarray: PDF value(s).
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        num = theta * (1 + (theta - 1) * (u + v - 2 * u * v))
        denom = ((1 + (theta - 1) * (u + v)) ** 2 - 4 * theta * (theta - 1) * u * v) ** 1.5
        return num / denom

    def sample(self, n, param=None):
        """Generate samples from the Plackett copula (placeholder implementation).

        Args:
            n (int): Number of samples to generate.
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            np.ndarray: Samples of shape (n, 2).
        """
        if param is None:
            param = self.parameters
        u = np.random.rand(n)
        v = np.random.rand(n)
        return np.column_stack((u, v))  # NOTE: Not an exact sampler

    def kendall_tau(self, param=None):
        """Compute Kendall's tau for the Plackett copula.

        Args:
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float: Kendall's tau.
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        return (theta - 1) / (theta + 1)

    def LTDC(self, param=None):
        """Lower tail dependence coefficient (0 for Plackett copula).

        Args:
            param (np.ndarray, optional): Copula parameter.

        Returns:
            float: 0.0
        """
        return 0.0

    def UTDC(self, param=None):
        """Upper tail dependence coefficient (0 for Plackett copula).

        Args:
            param (np.ndarray, optional): Copula parameter.

        Returns:
            float: 0.0
        """
        return 0.0

    def IAD(self, data):
        """Integrated Absolute Deviation (disabled for Plackett copula).

        Args:
            data (array-like): Input data (unused).

        Returns:
            float: NaN.
        """
        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        """Anderson–Darling test statistic (disabled for Plackett copula).

        Args:
            data (array-like): Input data (unused).

        Returns:
            float: NaN.
        """
        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """Compute ∂C(u,v)/∂u.

        Args:
            u (float or np.ndarray): U values.
            v (float or np.ndarray): V values.
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float or np.ndarray: Partial derivative values.
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        a = theta - 1
        b = 1 + a * (u + v)
        c = np.sqrt(b ** 2 - 4 * theta * a * u * v)
        return (2 * theta * v * (b + c - 2 * a * u)) / ((b + c) ** 2)

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """Compute ∂C(u,v)/∂v via symmetry.

        Args:
            u (float or np.ndarray): U values.
            v (float or np.ndarray): V values.
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
