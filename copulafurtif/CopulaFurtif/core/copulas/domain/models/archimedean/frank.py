"""
Frank Copula implementation.

The Frank copula is an Archimedean copula that supports both positive and negative 
dependence. It is symmetric and does not exhibit tail dependence. This implementation 
includes methods for CDF, PDF, sampling, and conditional distributions.

Attributes:
    name (str): Human-readable name of the copula.
    type (str): Copula identifier.
    bounds_param (list of tuple): Bounds for the copula parameter [theta] ∈ (-35, 35).
    parameters (np.ndarray): Copula parameter [theta].
    default_optim_method (str): Optimization method used for parameter fitting.
"""

import numpy as np
from scipy.special import spence
from scipy.stats import uniform
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


class FrankCopula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """Frank Copula model."""

    def __init__(self):
        """Initialize the Frank copula with default parameters and bounds."""
        super().__init__()
        self.name = "Frank Copula"
        self.type = "frank"
        self.bounds_param = [(-35.0, 35.0)]
        self._parameters = np.array([5.0])
        self.default_optim_method = "SLSQP"

    @property
    def parameters(self):
        """Get the copula parameter.

        Returns:
            np.ndarray: Copula parameter [theta].
        """
        return self._parameters

    @parameters.setter
    def parameters(self, param):
        """Set and validate the copula parameter.

        Args:
            param (array-like): New parameter [theta].

        Raises:
            ValueError: If parameter is out of bounds.
        """
        param = np.asarray(param)
        if not (self.bounds_param[0][0] < param[0] < self.bounds_param[0][1]):
            raise ValueError("Parameter out of bounds")
        self._parameters = param

    def get_cdf(self, u, v, param=None):
        """Compute the copula CDF C(u, v).

        Args:
            u (float or np.ndarray): First input in (0,1).
            v (float or np.ndarray): Second input in (0,1).
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float or np.ndarray: CDF values at (u, v).
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        if np.isclose(theta, 0.0):
            return u * v
        num = (np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1)
        denom = np.exp(-theta) - 1
        return -1 / theta * np.log(1 + num / denom)

    def get_pdf(self, u, v, param=None):
        """Compute the copula PDF c(u, v).

        Args:
            u (float or np.ndarray): First input in (0,1).
            v (float or np.ndarray): Second input in (0,1).
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float or np.ndarray: PDF values at (u, v).
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        if np.isclose(theta, 0.0):
            return np.ones_like(u)
        e_theta_u = np.exp(-theta * u)
        e_theta_v = np.exp(-theta * v)
        num = theta * e_theta_u * e_theta_v * (1 - np.exp(-theta))
        denom = (1 - np.exp(-theta) + (e_theta_u - 1) * (e_theta_v - 1)) ** 2
        return num / denom

    def sample(self, n, param=None):
        """Generate samples from the Frank copula.

        Args:
            n (int): Number of samples to generate.
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            np.ndarray: Array of shape (n, 2) of pseudo-observations.
        """
        if param is None:
            theta = self.parameters[0]
        else:
            theta = float(param[0])

        if abs(theta) < 1e-8:
            u = uniform.rvs(size=n)
            v = uniform.rvs(size=n)
            return np.column_stack((u, v))

        u = uniform.rvs(size=n)
        w = uniform.rvs(size=n)
        exp_neg_t = np.exp(-theta)
        exp_neg_t_u = np.exp(-theta * u)
        numerator = np.log(1 - w * (1 - exp_neg_t))
        denominator = exp_neg_t_u - 1
        denominator = np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
        v = -1.0 / theta * np.log(1 + numerator / denominator)
        return np.column_stack((u, v))

    def kendall_tau(self, param=None):
        """Compute Kendall's tau for the Frank copula.

        Args:
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float: Kendall's tau value.
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        if np.isclose(theta, 0.0):
            return 0.0
        return 1 + 4 / theta * (1 - spence(1 - np.exp(-theta)) / np.exp(-theta))

    def LTDC(self, param=None):
        """Lower tail dependence coefficient (always 0 for Frank copula).

        Args:
            param (np.ndarray, optional): Copula parameter.

        Returns:
            float: 0.0
        """
        return 0.0

    def UTDC(self, param=None):
        """Upper tail dependence coefficient (always 0 for Frank copula).

        Args:
            param (np.ndarray, optional): Copula parameter.

        Returns:
            float: 0.0
        """
        return 0.0

    def IAD(self, data):
        """Integrated Absolute Deviation (disabled for Frank copula).

        Args:
            data (array-like): Input data (unused).

        Returns:
            float: NaN
        """
        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        """Anderson–Darling test statistic (disabled for Frank copula).

        Args:
            data (array-like): Input data (unused).

        Returns:
            float: NaN
        """
        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """Compute ∂C(u,v)/∂u.

        Args:
            u (float or np.ndarray): First input in (0,1).
            v (float or np.ndarray): Second input in (0,1).
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float or np.ndarray: Partial derivative values.
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        num = theta * np.exp(-theta * u) * (np.exp(-theta * v) - 1)
        denom = (np.exp(-theta) - 1 + (np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1))
        return num / denom

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
