"""
Joe Copula implementation.

The Joe copula is an asymmetric Archimedean copula that models strong upper tail dependence.
It is often used in finance and insurance to capture co-movements in the upper tails
of joint distributions.

Attributes:
    name (str): Human-readable name of the copula.
    type (str): Identifier for the copula family.
    bounds_param (list of tuple): Parameter bounds for theta ∈ (1.01, 30.0).
    parameters (np.ndarray): Copula parameter [theta].
    default_optim_method (str): Optimization method used for fitting.
"""

import numpy as np
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


class JoeCopula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """Joe Copula model."""

    def __init__(self):
        """Initialize the Joe copula with default parameters and bounds."""
        super().__init__()
        self.name = "Joe Copula"
        self.type = "joe"
        self.bounds_param = [(1.01, 30.0)]  # [theta]
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
            float or np.ndarray: Value(s) of the CDF.
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        term1 = (1 - (1 - u) ** theta)
        term2 = (1 - (1 - v) ** theta)
        return 1 - ((1 - term1 * term2) ** (1 / theta))

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
        a = (1 - u) ** theta
        b = (1 - v) ** theta
        ab = a * b
        one_minus_ab = 1 - (1 - ab) ** (1 / theta)
        term = (1 - ab) ** (1 / theta - 2) * (a * (1 - v) ** (theta - 1) + b * (1 - u) ** (theta - 1))
        pdf = (1 - ab) ** (1 / theta - 1) * theta * (1 - u) ** (theta - 1) * (1 - v) ** (theta - 1) * term / (u * v)
        return pdf

    def sample(self, n, param=None):
        """Generate samples from the Joe copula (placeholder implementation).

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
        """Compute Kendall's tau for the Joe copula.

        Args:
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float: Kendall's tau.
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        return 1 - 1 / theta

    def LTDC(self, param=None):
        """Lower tail dependence coefficient (0 for Joe copula).

        Args:
            param (np.ndarray, optional): Copula parameter.

        Returns:
            float: 0.0
        """
        return 0.0

    def UTDC(self, param=None):
        """Upper tail dependence coefficient for the Joe copula.

        Args:
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float: UTDC value.
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        return 2 - 2 ** (1 / theta)

    def IAD(self, data):
        """Integrated Absolute Deviation (disabled for Joe copula).

        Args:
            data (array-like): Input data (unused).

        Returns:
            float: NaN.
        """
        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        """Anderson–Darling test statistic (disabled for Joe copula).

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

        A = (1 - u) ** theta
        B = (1 - v) ** theta
        Z = A + B - A * B

        return (1 - u) ** (theta - 1) * (1 - B) * Z ** (1 / theta - 1)

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
