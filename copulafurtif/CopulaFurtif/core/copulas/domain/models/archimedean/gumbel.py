"""
Gumbel Copula implementation.

The Gumbel copula is an Archimedean copula that captures upper tail dependence, 
commonly used for modeling extreme value dependence structures. It is suitable 
for positively dependent data where strong upper tail correlation is expected.

Attributes:
    name (str): Human-readable name of the copula.
    type (str): Identifier of the copula type.
    bounds_param (list of tuple): Parameter bounds for theta ∈ (1.01, 30.0).
    parameters (np.ndarray): Copula parameter [theta].
    default_optim_method (str): Optimization method used during fitting.
"""

import numpy as np
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


class GumbelCopula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """Gumbel Copula model."""

    def __init__(self):
        """Initialize the Gumbel copula with default parameter and bounds."""
        super().__init__()
        self.name = "Gumbel Copula"
        self.type = "gumbel"
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
            float or np.ndarray: CDF value(s).
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        log_u = -np.log(u)
        log_v = -np.log(v)
        sum_pow = (log_u ** theta + log_v ** theta) ** (1 / theta)
        return np.exp(-sum_pow)

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
        log_u = -np.log(u)
        log_v = -np.log(v)
        A = log_u ** theta + log_v ** theta
        C = A ** (1 / theta)
        cdf = np.exp(-C)

        part1 = cdf * (log_u * log_v) ** (theta - 1)
        part2 = (theta - 1) * A ** (-2 + 2 / theta)
        part3 = (theta + 1 - theta * (log_u ** theta / A)) * (theta + 1 - theta * (log_v ** theta / A))
        denom = u * v * A ** (2 - 2 / theta)

        return part1 * (part2 + part3) / denom

    def sample(self, n, param=None):
        """Generate random samples from the Gumbel copula.

        Args:
            n (int): Number of samples to generate.
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            np.ndarray: Samples of shape (n, 2).
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        from scipy.stats import expon
        e = expon.rvs(size=(n, 2))
        w = expon.rvs(size=n)
        t = (e[:, 0] / w) ** (1 / theta)
        s = (e[:, 1] / w) ** (1 / theta)
        u = np.exp(-t)
        v = np.exp(-s)
        return np.column_stack((u, v))

    def kendall_tau(self, param=None):
        """Compute Kendall's tau for the Gumbel copula.

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
        """Lower tail dependence coefficient (0 for Gumbel copula).

        Args:
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float: LTDC value.
        """
        return 0

    def UTDC(self, param=None):
        """Upper tail dependence coefficient (same as LTDC for Gumbel copula).

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
        """Integrated Absolute Deviation (disabled for Gumbel copula).

        Args:
            data (array-like): Input data (unused).

        Returns:
            float: NaN.
        """
        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        """Anderson–Darling test statistic (disabled for Gumbel copula).

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
        log_u = -np.log(u)
        log_v = -np.log(v)
        A = log_u ** theta + log_v ** theta
        C = A ** (1 / theta)
        return np.exp(-C) * log_u ** (theta - 1) * (log_v ** theta + log_u ** theta) ** (1 / theta - 1) / u

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
