import numpy as np
from domain.models.interfaces import CopulaModel
from domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


class FGMCopula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    def __init__(self):
        super().__init__()
        self.name = "FGM Copula"
        self.type = "fgm"
        self.bounds_param = [(-1.0, 1.0)]
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
        return u * v * (1 + theta * (1 - u) * (1 - v))

    def get_pdf(self, u, v, param=None):
        if param is None:
            param = self.parameters
        theta = param[0]
        return 1 + theta * (1 - 2 * u) * (1 - 2 * v)

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
        return (2 * theta) / 9