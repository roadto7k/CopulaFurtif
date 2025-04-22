"""
Clayton Copula implementation.

The Clayton copula is a popular Archimedean copula used to model asymmetric 
dependence, especially in the lower tail. It is parameterized by a single positive 
parameter theta > 0, which controls the strength of dependence.

Attributes:
    name (str): Human-readable name of the copula.
    type (str): Identifier for the copula family.
    bounds_param (list of tuple): Bounds for the copula parameter [theta].
    parameters (np.ndarray): Copula parameter [theta].
    default_optim_method (str): Default optimization method for fitting.
"""

import numpy as np
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


class ClaytonCopula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """Clayton copula model."""

    def __init__(self):
        """Initialize the Clayton copula with default parameters and bounds."""
        super().__init__()
        self.name = "Clayton Copula"
        self.type = "clayton"
        self.bounds_param = [(0.01, 30.0)]
        self._parameters = np.array([2.0])
        self.default_optim_method = "SLSQP"

    @property
    def parameters(self):
        """Get the copula parameter.

        Returns:
            np.ndarray: Current copula parameter [theta].
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
            float or np.ndarray: Value(s) of the CDF.
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        return np.maximum((u ** -theta + v ** -theta - 1) ** (-1 / theta), 0.0)

    def get_pdf(self, u, v, param=None):
        """Compute the copula PDF c(u, v).

        Args:
            u (float or np.ndarray): First input in (0,1).
            v (float or np.ndarray): Second input in (0,1).
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float or np.ndarray: Value(s) of the PDF.
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        num = (theta + 1) * (u * v) ** (-theta - 1)
        denom = (u ** -theta + v ** -theta - 1) ** (2 + 1 / theta)
        return num / denom

    def sample(self, n, param=None):
        """Generate random samples from the Clayton copula.

        Args:
            n (int): Number of samples to generate.
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            np.ndarray: Samples of shape (n, 2).
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        u = np.random.rand(n)
        w = np.random.gamma(1 / theta, 1, n)
        v = (1 - np.log(np.random.rand(n)) / w) ** (-1 / theta)
        return np.column_stack((u, np.clip(v, 1e-12, 1 - 1e-12)))

    def kendall_tau(self, param=None):
        """Compute Kendall's tau for the Clayton copula.

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
        """Lower tail dependence coefficient for Clayton copula.

        Args:
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float: LTDC value.
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        return 2 ** (-1 / theta)

    def UTDC(self, param=None):
        """Upper tail dependence coefficient (always 0 for Clayton copula).

        Args:
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float: 0.0
        """
        return 0.0

    def IAD(self, data):
        """Integrated Absolute Deviation (disabled for Clayton copula).

        Args:
            data (array-like): Input data (unused).

        Returns:
            float: NaN
        """
        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        """Anderson–Darling test statistic (disabled for Clayton copula).

        Args:
            data (array-like): Input data (unused).

        Returns:
            float: NaN
        """
        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """Compute ∂C(u, v)/∂u.

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
        top = (u ** -theta + v ** -theta - 1) ** (-1 / theta - 1)
        return top * u ** (-theta - 1)

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """Compute ∂C(u, v)/∂v via symmetry.

        Args:
            u (float or np.ndarray): First input in (0,1).
            v (float or np.ndarray): Second input in (0,1).
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
