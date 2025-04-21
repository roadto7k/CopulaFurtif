import numpy as np
from domain.models.interfaces import CopulaModel
from domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


class TawnCopula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    def __init__(self):
        super().__init__()
        self.name = "Tawn Type III Copula"
        self.type = "tawn3"
        self.bounds_param = [(0.01, 5.0), (0.0, 1.0)]  # [theta, delta]
        self._parameters = np.array([1.5, 0.5])
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
        log_u = -np.log(u)
        log_v = -np.log(v)
        S = log_u + log_v
        A = (1 - delta) * (log_u / S) ** theta + delta * (log_v / S) ** theta
        return np.exp(-S * A ** (1 / theta))

    def get_pdf(self, u, v, param=None):
        if param is None:
            param = self.parameters
        theta, delta = param
        # NOTE: PDF for Tawn copula is analytically complex; simplified return for now
        return np.ones_like(u)  # Placeholder for testing

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
