
import numpy as np
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


class AMHCopula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    def __init__(self):
        super().__init__()
        self.name = "AMH Copula"
        self.type = "amh"
        self.bounds_param = [(-0.999, 1.0)]
        self._parameters = np.array([0.3])
        self.default_optim_method = "SLSQP"

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, param):
        param = np.asarray(param)
        if not (self.bounds_param[0][0] <= param[0] <= self.bounds_param[0][1]):
            raise ValueError("Parameter out of bounds")
        self._parameters = param

    def get_cdf(self, u, v, param=None):
        if param is None:
            param = self.parameters
        theta = param[0]
        num = u * v
        denom = 1 - theta * (1 - u) * (1 - v)
        return num / denom

    def get_pdf(self, u, v, param=None):
        if param is None:
            param = self.parameters
        theta = param[0]
        numerator = 1 + theta * (1 - 2 * u) * (1 - 2 * v)
        denominator = (1 - theta * (1 - u) * (1 - v)) ** 2
        return numerator / denominator

    def sample(self, n, param=None):
        if param is None:
            param = self.parameters
        theta = param[0]
        u = np.random.rand(n)
        v = np.random.rand(n)
        return np.column_stack((u, v))  # Placeholder

    def kendall_tau(self, param=None):
        if param is None:
            param = self.parameters
        theta = param[0]
        return 1 - 2 * (theta / (3 * (1 + theta)))

    def LTDC(self, param=None):
        return 0.0

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
        num = v * (1 - theta * (1 - v))
        denom = (1 - theta * (1 - u) * (1 - v)) ** 2
        return num / denom

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        return self.partial_derivative_C_wrt_u(v, u, param)

    def conditional_cdf_u_given_v(self, u, v, param=None):
        return self.partial_derivative_C_wrt_v(u, v, param)

    def conditional_cdf_v_given_u(self, u, v, param=None):
        return self.partial_derivative_C_wrt_u(u, v, param)
