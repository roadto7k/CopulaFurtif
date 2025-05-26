import numpy as np
from scipy.optimize import brentq
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


class BB4Copula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """
    BB4 Copula (Two-parameter Archimedean copula).

    Attributes:
        name (str): Human-readable name of the copula.
        type (str): Identifier for the copula family.
        bounds_param (list of tuple): Bounds for copula parameters [mu, delta].
        parameters (np.ndarray): Current copula parameters.
        default_optim_method (str): Optimization method.
    """

    def __init__(self):
        """Initialize BB4 copula with default parameters mu=1.0, delta=1.0."""
        super().__init__()
        self.name = "BB4 Copula"
        self.type = "bb4"
        self.bounds_param = [(1e-6, None), (1e-6, None)]  # [mu, delta]
        self._parameters = np.array([1.0, 1.0])
        self.default_optim_method = "Powell"

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, param):
        param = np.asarray(param)
        for idx, (lower, upper) in enumerate(self.bounds_param):
            if lower is not None and param[idx] <= lower:
                raise ValueError(f"Parameter {['mu','delta'][idx]} must be > {lower}, got {param[idx]}")
        self._parameters = param

    def get_cdf(self, u, v, param=None):
        if param is None:
            param = self.parameters
        mu, delta = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        x = u ** (-mu)
        y = v ** (-mu)
        T = (x - 1) ** (-delta) + (y - 1) ** (-delta)
        A = T ** (-1.0 / delta)
        z = x + y - 1.0 - A
        return z ** (-1.0 / mu)

    def get_pdf(self, u, v, param=None):
        if param is None:
            param = self.parameters
        mu, delta = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        x = u ** (-mu)
        y = v ** (-mu)
        T = (x - 1) ** (-delta) + (y - 1) ** (-delta)
        A = T ** (-1.0 / delta)
        z = x + y - 1.0 - A
        dzdu = -mu * u ** (-mu - 1) * (1.0 - (x - 1) ** (-delta - 1) * T ** (-1.0 / delta - 1))
        dzdv = -mu * v ** (-mu - 1) * (1.0 - (y - 1) ** (-delta - 1) * T ** (-1.0 / delta - 1))
        d2zdudv = -mu**2 * (delta + 1) * u**(-mu - 1) * v**(-mu - 1) * (x - 1)**(-delta - 1) * (y - 1)**(-delta - 1) * T**(-1.0 / delta - 2)
        dCdz = -1.0 / mu * z ** (-1.0 / mu - 1)
        d2Cdz2 = (1.0 / mu) * (1.0 / mu + 1.0) * z ** (-1.0 / mu - 2)
        return d2Cdz2 * dzdu * dzdv + dCdz * d2zdudv

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
            root = brentq(lambda vv: self.conditional_cdf_v_given_u(u, vv, param) - p, 1e-6, 1 - 1e-6)
            samples[i] = [u, root]
        return samples

    def LTDC(self, param=None):
        return 0.0

    def UTDC(self, param=None):
        if param is None:
            param = self.parameters
        delta = param[1]
        return 2.0 - 2.0 ** (1.0 / delta)

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        if param is None:
            param = self.parameters
        mu, delta = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        x = u ** (-mu)
        y = v ** (-mu)
        T = (x - 1) ** (-delta) + (y - 1) ** (-delta)
        h1 = - (1.0 / mu) * ((x + y - 1.0 - T ** (-1.0 / delta)) ** (-1.0 / mu - 1))
        phi_inv_u_prime = -mu * u ** (-mu - 1) * (1 - (x - 1) ** (-delta - 1) * T ** (-1.0 / delta - 1))
        return h1 * phi_inv_u_prime

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
