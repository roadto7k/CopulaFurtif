"""
Clayton Copula implementation following the project coding standard:

Norms:
 1. Use private `_parameters` with public `@property parameters` and validation in setter.
 2. All methods accept `param: np.ndarray = None` defaulting to `self.parameters`.
 3. Docstrings include **Parameters** and **Returns** with types.
 4. Parameter bounds in `bounds_param`; setter enforces them.
 5. Uniform boundary clipping with `eps=1e-12`.
"""
import numpy as np
from domain.models.interfaces import CopulaModel
from domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


class ClaytonCopula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    def __init__(self):
        super().__init__()
        self.name = "Clayton Copula"
        self.type = "clayton"
        self.bounds_param = [(0.01, 20.0)]
        self._parameters = np.array([1.5])
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