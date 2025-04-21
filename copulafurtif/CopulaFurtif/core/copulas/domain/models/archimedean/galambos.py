import numpy as np
from domain.models.interfaces import CopulaModel
from domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


class GalambosCopula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    def __init__(self):
        super().__init__()
        self.name = "Galambos Copula"
        self.type = "galambos"
        self.bounds_param = [(0.01, 10.0)]
        self._parameters = np.array([1.2])
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
        log_u = -np.log(u)
        log_v = -np.log(v)
        sum_pow = log_u ** (-theta) + log_v ** (-theta)
        return np.exp(-sum_pow ** (-1 / theta))

    def get_pdf(self, u, v, param=None):
        if param is None:
            param = self.parameters
        theta = param[0]
        log_u = -np.log(u)
        log_v = -np.log(v)
        A = log_u ** (-theta) + log_v ** (-theta)
        B = A ** (-1 / theta - 2)
        part = (theta + 1) * (log_u * log_v) ** (-theta - 1)
        return np.exp(-A ** (-1 / theta)) * part * B / (u * v)

    def sample(self, n, param=None):
        if param is None:
            param = self.parameters
        u = np.random.rand(n)
        v = np.random.rand(n)
        return np.column_stack((u, v))  # Approximate sample

    def kendall_tau(self, param=None):
        if param is None:
            param = self.parameters
        theta = param[0]
        return theta / (theta + 2)