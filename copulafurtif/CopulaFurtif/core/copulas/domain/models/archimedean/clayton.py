import numpy as np
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


class ClaytonCopula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    def __init__(self):
        super().__init__()
        self.name = "Clayton Copula"
        self.type = "clayton"
        self.bounds_param = [(0.01, 30.0)]
        self._parameters = np.array([2.0])
        self.default_optim_method = "SLSQP"

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, param):
        param = np.asarray(param)
        if not (self.bounds_param[0][0] < param[0] < self.bounds_param[0][1]):
            raise ValueError("Parameter out of bounds")
        self._parameters = param

    def get_cdf(self, u, v, param=None):
        if param is None:
            param = self.parameters
        theta = param[0]
        return np.maximum((u ** -theta + v ** -theta - 1) ** (-1 / theta), 0.0)

    def get_pdf(self, u, v, param=None):
        if param is None:
            param = self.parameters
        theta = param[0]
        num = (theta + 1) * (u * v) ** (-theta - 1)
        denom = (u ** -theta + v ** -theta - 1) ** (2 + 1 / theta)
        return num / denom

    def sample(self, n, param=None):
        if param is None:
            param = self.parameters
        theta = param[0]
        u = np.random.rand(n)
        w = np.random.gamma(1 / theta, 1, n)
        v = (1 - np.log(np.random.rand(n)) / w) ** (-1 / theta)
        return np.column_stack((u, np.clip(v, 1e-12, 1 - 1e-12)))

    def kendall_tau(self, param=None):
        if param is None:
            param = self.parameters
        theta = param[0]
        return theta / (theta + 2)

    def LTDC(self, param=None):
        if param is None:
            param = self.parameters
        theta = param[0]
        return 2 ** (-1 / theta)

    def UTDC(self, param=None):
        return 0.0

    def IAD(self, data):
        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        if param is None:
            param = self.parameters
        theta = param[0]
        top = (u ** -theta + v ** -theta - 1) ** (-1 / theta - 1)
        return top * u ** (-theta - 1)

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        return self.partial_derivative_C_wrt_u(v, u, param)

    def conditional_cdf_u_given_v(self, u, v, param=None):
        return self.partial_derivative_C_wrt_v(u, v, param)

    def conditional_cdf_v_given_u(self, u, v, param=None):
        return self.partial_derivative_C_wrt_u(u, v, param)