import numpy as np
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


class GalambosCopula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    def __init__(self):
        super().__init__()
        self.name = "Galambos Copula"
        self.type = "galambos"
        self.bounds_param = [(0.01, 10.0)]
        self._parameters = np.array([1.5])
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
        x = -np.log(u)
        y = -np.log(v)
        S = x ** (-theta) + y ** (-theta)
        return np.exp(-S ** (-1 / theta))

    def get_pdf(self, u, v, param=None):
        if param is None:
            param = self.parameters
        theta = param[0]
        x = -np.log(u)
        y = -np.log(v)
        S = x ** (-theta) + y ** (-theta)
        C = S ** (-1 / theta)
        part1 = (x * y) ** (-theta - 1)
        part2 = (theta + 1) * S ** (-2 - 1 / theta)
        pdf = np.exp(-C) * part1 * part2 / (u * v)
        return pdf

    def sample(self, n, param=None):
        if param is None:
            param = self.parameters
        u = np.random.rand(n)
        v = np.random.rand(n)
        return np.column_stack((u, v))  # Placeholder only

    def kendall_tau(self, param=None):
        if param is None:
            param = self.parameters
        theta = param[0]
        return theta / (theta + 2)

    def LTDC(self, param=None):
        if param is None:
            param = self.parameters
        theta = param[0]
        return 2 - 2 ** (1 / theta)

    def UTDC(self, param=None):
        return self.LTDC(param)

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
        x = -np.log(u)
        y = -np.log(v)
        S = x ** (-theta) + y ** (-theta)
        A = S ** (-1 / theta - 1)
        B = x ** (-theta - 1)
        return np.exp(-S ** (-1 / theta)) * A * B / u

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        return self.partial_derivative_C_wrt_u(v, u, param)

    def conditional_cdf_u_given_v(self, u, v, param=None):
        return self.partial_derivative_C_wrt_v(u, v, param)

    def conditional_cdf_v_given_u(self, u, v, param=None):
        return self.partial_derivative_C_wrt_u(u, v, param)