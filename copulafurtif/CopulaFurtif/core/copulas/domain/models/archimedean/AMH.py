"""
Ali-Mikhail-Haq (AMH) Copula implementation.

The AMH copula is an Archimedean copula with a single parameter that allows for a wide 
range of dependence structures, though it has limited tail dependence. This class supports 
CDF/PDF evaluation, parameter fitting, sampling (placeholder), and basic diagnostics.

Attributes:
    name (str): Human-readable name of the copula.
    type (str): Copula identifier.
    bounds_param (list of tuple): Bounds for the parameter theta ∈ (-0.999, 1.0).
    parameters (np.ndarray): Copula parameter [theta].
    default_optim_method (str): Optimization method used during fitting.
"""

import numpy as np
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


class AMHCopula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """AMH Copula model."""

    def __init__(self):
        """Initialize the AMH copula with default parameter and bounds."""
        super().__init__()
        self.name = "AMH Copula"
        self.type = "amh"
        self.bounds_param = [(-0.999, 1.0)]
        self._parameters = np.array([0.3])
        self.default_optim_method = "SLSQP"

    @property
    def parameters(self):
        """Get the current copula parameter.

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
        if not (self.bounds_param[0][0] <= param[0] <= self.bounds_param[0][1]):
            raise ValueError("Parameter out of bounds")
        self._parameters = param

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
        num = u * v
        denom = 1 - theta * (1 - u) * (1 - v)
        return num / denom

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
        numerator = 1 + theta * (1 - 2 * u) * (1 - 2 * v)
        denominator = (1 - theta * (1 - u) * (1 - v)) ** 2
        return numerator / denominator

    def sample(self, n, param=None):
        """Generate n samples from the AMH copula (placeholder implementation).

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
        v = np.random.rand(n)
        return np.column_stack((u, v))  # NOTE: Placeholder, not real AMH sampling

    def kendall_tau(self, param=None):
        """Compute Kendall's tau for the AMH copula.

        Args:
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float: Kendall's tau.
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        return 1 - 2 * (theta / (3 * (1 + theta)))

    def LTDC(self, param=None):
        """Lower tail dependence coefficient (always 0 for AMH copula).

        Args:
            param (np.ndarray, optional): Copula parameter.

        Returns:
            float: 0.0
        """
        return 0.0

    def UTDC(self, param=None):
        """Upper tail dependence coefficient (always 0 for AMH copula).

        Args:
            param (np.ndarray, optional): Copula parameter.

        Returns:
            float: 0.0
        """
        return 0.0

    def IAD(self, data):
        """Integrated Absolute Deviation (disabled for AMH copula).

        Args:
            data (array-like): Input data (unused).

        Returns:
            float: NaN.
        """
        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        """Anderson–Darling statistic (disabled for AMH copula).

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
            float or np.ndarray: Conditional CDF value.
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        num = v * (1 - theta * (1 - v))
        denom = (1 - theta * (1 - u) * (1 - v)) ** 2
        return num / denom

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """Compute ∂C(u,v)/∂v (via symmetry).

        Args:
            u (float or np.ndarray): U value(s).
            v (float or np.ndarray): V value(s).
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float or np.ndarray: Conditional CDF value.
        """
        return self.partial_derivative_C_wrt_u(v, u, param)

    def conditional_cdf_u_given_v(self, u, v, param=None):
        """Compute the conditional CDF P(U ≤ u | V = v).

        Args:
            u (float or np.ndarray): U value(s).
            v (float or np.ndarray): V value(s).
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float or np.ndarray: Conditional CDF value.
        """
        return self.partial_derivative_C_wrt_v(u, v, param)

    def conditional_cdf_v_given_u(self, u, v, param=None):
        """Compute the conditional CDF P(V ≤ v | U = u).

        Args:
            u (float or np.ndarray): U value(s).
            v (float or np.ndarray): V value(s).
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float or np.ndarray: Conditional CDF value.
        """
        return self.partial_derivative_C_wrt_u(u, v, param)
