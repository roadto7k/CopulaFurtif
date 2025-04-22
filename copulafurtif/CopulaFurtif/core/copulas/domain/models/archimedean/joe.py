# domain/models/archimedean/joe_copula.py

import numpy as np
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


class JoeCopula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    def __init__(self):
        super().__init__()
        self.name = "Joe Copula"
        self.type = "joe"
        self.bounds_param = [(1.01, 30.0)]
        self._parameters = np.array([2.0])
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
        term1 = (1 - (1 - u) ** theta)
        term2 = (1 - (1 - v) ** theta)
        return 1 - ((1 - term1 * term2) ** (1 / theta))

    def get_pdf(self, u, v, param=None):
        if param is None:
            param = self.parameters
        theta = param[0]
        a = (1 - u) ** theta
        b = (1 - v) ** theta
        ab = a * b
        one_minus_ab = 1 - (1 - ab) ** (1 / theta)
        term = (1 - ab) ** (1 / theta - 2) * (a * (1 - v) ** (theta - 1) + b * (1 - u) ** (theta - 1))
        pdf = (1 - ab) ** (1 / theta - 1) * theta * (1 - u) ** (theta - 1) * (1 - v) ** (theta - 1) * term / (u * v)
        return pdf

    def sample(self, n, param=None):
        if param is None:
            param = self.parameters
        u = np.random.rand(n)
        v = np.random.rand(n)
        return np.column_stack((u, v))  # Placeholder

    def kendall_tau(self, param=None):
        if param is None:
            param = self.parameters
        theta = param[0]
        return 1 - 1 / theta

    def LTDC(self, param=None):
        return 0.0

    def UTDC(self, param=None):
        if param is None:
            param = self.parameters
        theta = param[0]
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
        theta = param[0]
        A = (1 - u) ** theta
        B = (1 - v) ** theta
        top = theta * A * (1 - v) ** (theta - 1) * (1 - A * B) ** (1 / theta - 2)
        return top / u

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        return self.partial_derivative_C_wrt_u(v, u, param)

    def conditional_cdf_u_given_v(self, u, v, param=None):
        return self.partial_derivative_C_wrt_v(u, v, param)

    def conditional_cdf_v_given_u(self, u, v, param=None):
        return self.partial_derivative_C_wrt_u(u, v, param)