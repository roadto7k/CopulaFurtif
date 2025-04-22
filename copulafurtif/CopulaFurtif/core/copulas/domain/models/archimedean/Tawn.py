import numpy as np
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


class TawnCopula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    def __init__(self):
        super().__init__()
        self.name = "Tawn Copula"
        self.type = "tawn"
        self.bounds_param = [(1.01, 5.0), (0.0, 1.0)]  # [theta, delta]
        self._parameters = np.array([2.0, 0.5])
        self.default_optim_method = "SLSQP"

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, param):
        param = np.asarray(param)
        for i, (low, high) in enumerate(self.bounds_param):
            if not (low <= param[i] <= high):
                raise ValueError(f"Parameter {i} out of bounds")
        self._parameters = param

    def get_cdf(self, u, v, param=None):
        if param is None:
            param = self.parameters
        theta, delta = param
        x = -np.log(u)
        y = -np.log(v)
        s = x + y
        w = (1 - delta) * (x / s) ** theta + delta * (y / s) ** theta
        return np.exp(-s * w ** (1 / theta))

    def get_pdf(self, u, v, param=None):
        if param is None:
            param = self.parameters
        # Full analytical PDF is complex; returning placeholder
        return np.ones_like(u)

    def sample(self, n, param=None):
        if param is None:
            param = self.parameters
        u = np.random.rand(n)
        v = np.random.rand(n)
        return np.column_stack((u, v))

    def kendall_tau(self, param=None):
        if param is None:
            param = self.parameters
        theta, delta = param
        return (theta * (1 - delta + delta)) / (theta + 2)

    def LTDC(self, param=None):
        return 0.0

    def UTDC(self, param=None):
        if param is None:
            param = self.parameters
        theta, delta = param
        return 2 - 2 ** (1 / theta)

    def IAD(self, data):
        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        if param is None:
            param = self.parameters
        theta, delta = param
        # Placeholder
        return np.ones_like(u)

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        return self.partial_derivative_C_wrt_u(v, u, param)

    def conditional_cdf_u_given_v(self, u, v, param=None):
        return self.partial_derivative_C_wrt_v(u, v, param)

    def conditional_cdf_v_given_u(self, u, v, param=None):
        return self.partial_derivative_C_wrt_u(u, v, param)
