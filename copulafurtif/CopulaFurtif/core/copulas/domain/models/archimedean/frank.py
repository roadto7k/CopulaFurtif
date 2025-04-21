import numpy as np
from scipy.special import spence
from domain.models.interfaces import CopulaModel
from domain.models.mixins import ModelSelectionMixin, SupportsTailDependence

class FrankCopula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    def __init__(self):
        super().__init__()
        self.name = "Frank Copula"
        self.type = "frank"
        self.bounds_param = [(-35.0, 35.0)]
        self._parameters = np.array([5.0])
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
        if theta == 0:
            return u * v
        num = (np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1)
        denom = np.exp(-theta) - 1
        return -1 / theta * np.log(1 + num / denom)

    def get_pdf(self, u, v, param=None):
        if param is None:
            param = self.parameters
        theta = param[0]
        if theta == 0:
            return np.ones_like(u)
        e_theta_u = np.exp(-theta * u)
        e_theta_v = np.exp(-theta * v)
        num = theta * (e_theta_u - 1) * (e_theta_v - 1) * np.exp(-theta * (u + v))
        denom = (np.exp(-theta) - 1 + (e_theta_u - 1) * (e_theta_v - 1))**2
        return num / denom

    def sample(self, n, param=None):
        if param is None:
            param = self.parameters
        theta = param[0]
        u = np.random.rand(n)
        w = -np.log(1 - u * (1 - np.exp(-theta))) / theta
        v = np.random.rand(n)
        inner = lambda t: -np.log(1 + np.exp(-theta * w) * (np.exp(-theta * t) - 1)) / theta
        return np.column_stack((w, [inner(t) for t in v]))

    def kendall_tau(self, param=None):
        if param is None:
            param = self.parameters
        theta = param[0]
        if theta == 0:
            return 0.0
        return 1 + 4 / theta * (1 - spence(1 - np.exp(-theta)) / np.exp(-theta))