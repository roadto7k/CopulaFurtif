import numpy as np
from scipy.special import beta
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


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
        self.bounds_param = [(1e-6, None), (1.0, None)]  # [theta, delta]
        self._parameters = np.array([0.5, 1.5])
        self.default_optim_method = "Powell"

    @property
    def parameters(self):
        """
        Get the copula parameters.

        Returns:
            np.ndarray: Current copula parameters [theta, delta].
        """
        return self._parameters

    @parameters.setter
    def parameters(self, param):
        """
        Set and validate the copula parameters.

        Args:
            param (array-like): New parameters [theta, delta].

        Raises:
            ValueError: If parameters are out of bounds.
        """
        param = np.asarray(param)
        for i, (lower, upper) in enumerate(self.bounds_param):
            if lower is not None and param[i] <= lower:
                raise ValueError(f"Parameter {i} must be > {lower}, got {param[i]}")
            if upper is not None and param[i] >= upper:
                raise ValueError(f"Parameter {i} must be < {upper}, got {param[i]}")
        self._parameters = param

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
            param = self.parameters
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
            param = self.parameters
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
        Compute Kendall's tau.

        Args:
            param (np.ndarray, optional): Copula parameters [theta, delta].

        Returns:
            float: Kendall's tau.
        """
        if param is None:
            param = self.parameters
        theta, delta = param
        return 1.0 - (2.0 / delta) * (1.0 - 1.0 / theta) * beta(1.0 - 1.0 / theta, 2.0 / delta + 1.0)

    def sample(self, n, param=None):
        """
        Generate samples using the Marshall–Olkin method.

        Args:
            n (int): Number of samples.
            param (np.ndarray, optional): Copula parameters [theta, delta].

        Returns:
            np.ndarray: Samples of shape (n, 2).
        """
        if param is None:
            param = self.parameters
        theta, delta = param
        V = np.random.gamma(1.0 / delta, 1.0, size=n)
        E1 = np.random.exponential(size=n)
        E2 = np.random.exponential(size=n)
        U = (1 + (E1 / V) ** (1.0 / delta)) ** (-1.0 / theta)
        W = (1 + (E2 / V) ** (1.0 / delta)) ** (-1.0 / theta)
        return np.column_stack((U, W))

    def LTDC(self, param=None):
        """
        Compute lower tail dependence coefficient.

        Args:
            param (np.ndarray, optional): Copula parameters [theta, delta].

        Returns:
            float: Lower tail dependence.
        """
        if param is None:
            param = self.parameters
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
            param = self.parameters
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
            param = self.parameters
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

    def conditional_cdf_u_given_v(self, u, v, param=None):
        """
        Compute conditional CDF P(U ≤ u | V = v).

        Args:
            u (float or np.ndarray): U values.
            v (float or np.ndarray): V values.
            param (np.ndarray, optional): Copula parameters [theta, delta].

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
            param (np.ndarray, optional): Copula parameters [theta, delta].

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
