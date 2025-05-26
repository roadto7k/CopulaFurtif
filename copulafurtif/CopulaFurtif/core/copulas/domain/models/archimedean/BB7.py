import numpy as np
from scipy.optimize import brentq
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


class BB7Copula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """
    BB7 Copula (Joe-Clayton) Archimedean copula.

    Attributes:
        name (str): Name of the copula.
        type (str): Identifier for the copula family.
        bounds_param (list of tuple): Parameter bounds for optimization.
        parameters (np.ndarray): Copula parameters [theta, delta].
        default_optim_method (str): Optimization method.
    """

    def __init__(self):
        """Initialize the BB7 copula with default parameters."""
        super().__init__()
        self.name = "BB7 Copula"
        self.type = "bb7"
        self.bounds_param = [(1e-6, None), (1e-6, None)]
        self._parameters = np.array([1.0, 1.0])
        self.default_optim_method = "Powell"

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, param):
        param = np.asarray(param)
        for i, (lower, _) in enumerate(self.bounds_param):
            if param[i] < lower:
                raise ValueError(f"Parameter {['theta', 'delta'][i]} must be >= {lower}, got {param[i]}")
        self._parameters = param

    def _phi(self, t, theta, delta):
        t = np.clip(t, 1e-12, 1 - 1e-12)
        phiJ = 1.0 - (1.0 - t)**theta
        return (phiJ**(-delta) - 1.0) / delta

    def _phi_inv(self, s, theta, delta):
        s = np.maximum(s, 0.0)
        phiC_inv = (1.0 + delta * s)**(-1.0 / delta)
        return 1.0 - (1.0 - phiC_inv)**(1.0 / theta)

    def get_cdf(self, u, v, param=None):
        if param is None:
            param = self.parameters
        theta, delta = param
        phi_u = self._phi(u, theta, delta)
        phi_v = self._phi(v, theta, delta)
        return self._phi_inv(phi_u + phi_v, theta, delta)

    def get_pdf(self, u, v, param=None):
        if param is None:
            param = self.parameters
        eps = 1e-6
        c = self.get_cdf
        return (
            c(u+eps, v+eps, param) - c(u+eps, v-eps, param)
            - c(u-eps, v+eps, param) + c(u-eps, v-eps, param)
        ) / (4.0 * eps**2)

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        if param is None:
            param = self.parameters
        eps = 1e-6
        c = self.get_cdf
        return (c(u+eps, v, param) - c(u-eps, v, param)) / (2.0 * eps)

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        return self.partial_derivative_C_wrt_u(v, u, param)

    def conditional_cdf_u_given_v(self, u, v, param=None):
        num = self.partial_derivative_C_wrt_v(u, v, param)
        den = self.partial_derivative_C_wrt_v(1.0, v, param)
        return num / den

    def conditional_cdf_v_given_u(self, u, v, param=None):
        num = self.partial_derivative_C_wrt_u(u, v, param)
        den = self.partial_derivative_C_wrt_u(u, 1.0, param)
        return num / den

    def sample(self, n, param=None):
        if param is None:
            param = self.parameters
        samples = np.empty((n, 2))
        eps = 1e-6
        for i in range(n):
            u = np.random.rand()
            p = np.random.rand()
            root = brentq(
                lambda vv: self.conditional_cdf_v_given_u(u, vv, param) - p,
                eps, 1.0 - eps
            )
            samples[i] = [u, root]
        return samples

    def LTDC(self, param=None):
        if param is None:
            param = self.parameters
        eps = 1e-6
        return self.get_cdf(eps, eps, param) / eps

    def UTDC(self, param=None):
        if param is None:
            param = self.parameters
        eps = 1e-6
        u = 1.0 - eps
        return 2.0 - (1.0 - 2*u + self.get_cdf(u, u, param)) / eps

    def kendall_tau(self, param=None):
        raise NotImplementedError("Kendall's tau not implemented for BB7.")

    def IAD(self, data):
        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan
