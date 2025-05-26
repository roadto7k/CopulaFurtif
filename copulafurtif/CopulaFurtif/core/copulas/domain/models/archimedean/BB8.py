import numpy as np
from scipy.optimize import root_scalar
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


class BB8Copula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """
    BB8 Copula (Durante et al.):
      C(u,v) = [1 - (1-A)*(1-B)]^(1/theta),
      where A = [1 - (1-u)^theta]^delta,
            B = [1 - (1-v)^theta]^delta.

    Attributes:
        name (str): Human-readable name of the copula.
        type (str): Identifier for the copula family.
        bounds_param (list of tuple): Bounds for copula parameters.
        parameters (np.ndarray): Current copula parameters.
        default_optim_method (str): Default method for optimization.
    """

    def __init__(self):
        super().__init__()
        self.name = "BB8 Copula (Durante)"
        self.type = "bb8"
        self.bounds_param = [(1.0, None), (0.0, 1.0)]  # [theta, delta]
        self._parameters = np.array([2.0, 0.7])
        self.default_optim_method = "Powell"

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, param):
        param = np.asarray(param)
        for i, (lower, upper) in enumerate(self.bounds_param):
            if param[i] < lower:
                raise ValueError(f"Parameter {['theta','delta'][i]} must be >= {lower}, got {param[i]}")
            if upper is not None and param[i] > upper:
                raise ValueError(f"Parameter {['theta','delta'][i]} must be <= {upper}, got {param[i]}")
        self._parameters = param

    def get_cdf(self, u, v, param=None):
        if param is None:
            param = self.parameters
        theta, delta = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        A = (1.0 - (1.0 - u)**theta)**delta
        B = (1.0 - (1.0 - v)**theta)**delta
        inner = 1.0 - (1.0 - A)*(1.0 - B)
        return inner**(1.0/theta)

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
        return (c(u+eps, v, param) - c(u, v, param)) / eps

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        if param is None:
            param = self.parameters
        eps = 1e-6
        c = self.get_cdf
        return (c(u, v+eps, param) - c(u, v, param)) / eps

    def conditional_cdf_v_given_u(self, u, v, param=None):
        return self.partial_derivative_C_wrt_u(u, v, param)

    def conditional_cdf_u_given_v(self, u, v, param=None):
        return self.partial_derivative_C_wrt_v(u, v, param)

    def sample(self, n, param=None):
        if param is None:
            param = self.parameters
        samples = np.empty((n, 2))
        eps = 1e-6
        for i in range(n):
            u = np.random.rand()
            p = np.random.rand()
            root = root_scalar(
                lambda vv: self.partial_derivative_C_wrt_u(u, vv, param) - p,
                bracket=[eps, 1 - eps], method='bisect', xtol=1e-6
            )
            samples[i] = [u, root.root]
        return samples

    def LTDC(self, param=None):
        if param is None:
            param = self.parameters
        u = 1e-6
        return self.get_cdf(u, u, param) / u

    def UTDC(self, param=None):
        if param is None:
            param = self.parameters
        u = 1.0 - 1e-6
        return (1 - 2*u + self.get_cdf(u, u, param)) / (1 - u)

    def kendall_tau(self, param=None):
        raise NotImplementedError("Kendall's tau not implemented for BB8.")

    def IAD(self, data):
        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan
