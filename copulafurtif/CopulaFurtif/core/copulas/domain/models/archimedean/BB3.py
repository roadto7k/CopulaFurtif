import numpy as np
from scipy.optimize import brentq
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


class BB3Copula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """
    BB3 Copula (Two-parameter Archimedean copula).

    Attributes:
        name (str): Human-readable name of the copula.
        type (str): Identifier for the copula family.
        bounds_param (list of tuple): Parameter bounds [d > 0, q >= 1].
        parameters (np.ndarray): Current copula parameters.
        default_optim_method (str): Optimization method.
    """

    def __init__(self):
        """Initialize BB3 copula with default parameters d=1.0, q=1.0."""
        super().__init__()
        self.name = "BB3 Copula"
        self.type = "bb3"
        self.bounds_param = [(1e-6, None), (1.0, None)]  # [d, q]
        self._parameters = np.array([1.0, 1.0])
        self.default_optim_method = "Powell"

    @property
    def parameters(self):
        """
        Get the copula parameters.

        Returns:
            np.ndarray: Parameters [d, q].
        """
        return self._parameters

    @parameters.setter
    def parameters(self, param):
        """
        Set and validate copula parameters.

        Args:
            param (array-like): Parameters [d, q].

        Raises:
            ValueError: If parameters are out of bounds.
        """
        param = np.asarray(param)
        for idx, (lower, upper) in enumerate(self.bounds_param):
            if lower is not None and param[idx] <= lower:
                raise ValueError(f"Parameter {['d','q'][idx]} must be > {lower}, got {param[idx]}")
        self._parameters = param

    def _h(self, s, param=None):
        if param is None:
            param = self.parameters
        d, q = param
        return (np.log1p(s) / d) ** (1.0 / q)

    def _h_prime(self, s, param=None):
        if param is None:
            param = self.parameters
        d, q = param
        g = np.log1p(s) / d
        return (1.0 / (q * d * (1.0 + s))) * g ** (1.0 / q - 1.0)

    def _h_double(self, s, param=None):
        if param is None:
            param = self.parameters
        d, q = param
        g = np.log1p(s) / d
        A = 1.0 / (q * d * (1.0 + s) ** 2)
        term1 = -g ** (1.0 / q - 1.0)
        term2 = (1.0 / q - 1.0) * g ** (1.0 / q - 2.0) / d
        return A * (term1 + term2)

    def get_cdf(self, u, v, param=None):
        if param is None:
            param = self.parameters
        d, q = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        s_u = np.expm1(d * (-np.log(u)) ** q)
        s_v = np.expm1(d * (-np.log(v)) ** q)
        s = s_u + s_v
        return np.exp(-self._h(s, param))

    def get_pdf(self, u, v, param=None):
        if param is None:
            param = self.parameters
        d, q = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        s_u = np.expm1(d * (-np.log(u)) ** q)
        s_v = np.expm1(d * (-np.log(v)) ** q)
        s = s_u + s_v
        h = self._h(s, param)
        h1 = self._h_prime(s, param)
        h2 = self._h_double(s, param)
        phi_dd = np.exp(-h) * (h1 ** 2 - h2)
        phi_inv_u_prime = -d * q * np.exp(d * (-np.log(u)) ** q) * (((-np.log(u)) ** (q - 1)) / u)
        phi_inv_v_prime = -d * q * np.exp(d * (-np.log(v)) ** q) * (((-np.log(v)) ** (q - 1)) / v)
        return phi_dd * phi_inv_u_prime * phi_inv_v_prime

    def kendall_tau(self, param=None, n=201):
        if param is None:
            param = self.parameters
        eps = 1e-6
        u = np.linspace(eps, 1 - eps, n)
        U, V = np.meshgrid(u, u)
        Z = self.get_cdf(U, V, param)
        integral = np.trapz(np.trapz(Z, u, axis=1), u)
        return 4.0 * integral - 1.0

    def sample(self, n, param=None):
        if param is None:
            param = self.parameters
        samples = np.empty((n, 2))
        for i in range(n):
            u = np.random.rand()
            p = np.random.rand()
            root = brentq(lambda v_val: self.conditional_cdf_v_given_u(u, v_val, param) - p, 1e-6, 1 - 1e-6)
            samples[i] = [u, root]
        return samples

    def LTDC(self, param=None):
        return 0.0

    def UTDC(self, param=None):
        if param is None:
            param = self.parameters
        q = param[1]
        return 2.0 - 2.0 ** (1.0 / q)

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        if param is None:
            param = self.parameters
        d, q = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        s_u = np.expm1(d * (-np.log(u)) ** q)
        s_v = np.expm1(d * (-np.log(v)) ** q)
        s = s_u + s_v
        h1 = self._h_prime(s, param)
        phi_p = -h1 * np.exp(-self._h(s, param))
        phi_inv_u_prime = -d * q * np.exp(d * (-np.log(u)) ** q) * (((-np.log(u)) ** (q - 1)) / u)
        return phi_p * phi_inv_u_prime

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        return self.partial_derivative_C_wrt_u(v, u, param)

    def conditional_cdf_u_given_v(self, u, v, param=None):
        return self.partial_derivative_C_wrt_v(u, v, param) / self.partial_derivative_C_wrt_v(1.0, v, param)

    def conditional_cdf_v_given_u(self, u, v, param=None):
        return self.partial_derivative_C_wrt_u(u, v, param) / self.partial_derivative_C_wrt_u(u, 1.0, param)

    def IAD(self, data):
        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan
