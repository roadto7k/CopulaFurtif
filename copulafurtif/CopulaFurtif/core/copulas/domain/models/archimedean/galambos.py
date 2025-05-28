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
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


class GalambosCopula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """Galambos Copula model."""

    def __init__(self):
        """Initialize the Galambos copula with default parameters and bounds."""
        super().__init__()
        self.name = "Galambos Copula"
        self.type = "galambos"
        self.bounds_param = [(0.01, 10.0)]  # [theta]
        self.param_names = ["theta"]
        self.parameters = [1.5]
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
        x = -np.log(u)
        y = -np.log(v)
        S = x ** (-theta) + y ** (-theta)
        return np.exp(-S ** (-1 / theta))

    def get_pdf(self, u, v, param=None):
        """Compute the copula PDF c(u, v).

        Args:
            u (float or np.ndarray): First input in (0, 1).
            v (float or np.ndarray): Second input in (0, 1).
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float or np.ndarray: Value(s) of the PDF.
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        x = -np.log(u)
        y = -np.log(v)
        S = x ** (-theta) + y ** (-theta)
        C = S ** (-1 / theta)
        part1 = (x * y) ** (-theta - 1)
        part2 = (theta + 1) * S ** (-2 - 1 / theta)
        pdf = np.exp(-C) * part1 * part2 / (u * v)
        return pdf

    def sample(self, n, param=None):
        """Generate random samples from the Galambos copula (placeholder implementation).

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
        """Compute Kendall's tau for the Galambos copula.

        Args:
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float: Kendall's tau.
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        return theta / (theta + 2)

    def LTDC(self, param=None):
        """Lower tail dependence coefficient (always 0 for Galambos copula).

        Args:
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float: 0.0
        """
        return 2 - 2 ** (1 / self.parameters[0])

    def UTDC(self, param=None):
        """Upper tail dependence coefficient for the Galambos copula.

        Args:
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float: Same as LTDC.
        """
        return self.LTDC(param)

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
        """Compute ∂C(u,v)/∂u.

        Args:
            u (float or np.ndarray): U value(s).
            v (float or np.ndarray): V value(s).
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float or np.ndarray: Partial derivative values.
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        x = -np.log(u)
        y = -np.log(v)
        S = x ** (-theta) + y ** (-theta)
        A = S ** (-1 / theta - 1)
        B = x ** (-theta - 1)
        return np.exp(-S ** (-1 / theta)) * A * B / u

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
