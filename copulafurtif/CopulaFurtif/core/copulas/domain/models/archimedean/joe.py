from __future__ import annotations

import numpy as np
from domain.models.interfaces import CopulaModel
from domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


class JoeCopula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    def __init__(self):
        super().__init__()
        self.name = "Joe Copula"
        self.type = "joe"
        self.bounds_param = [(1.01, 30.0)]
        self._parameters = np.array([2.5])
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
        return 1 - (1 - (1 - u) ** theta) * (1 - (1 - v) ** theta) ** (1 / theta)

    def get_pdf(self, u, v, param=None):
        if param is None:
            param = self.parameters
        theta = param[0]
        one_minus_u = 1 - u
        one_minus_v = 1 - v
        A = (1 - one_minus_u ** theta)
        B = (1 - one_minus_v ** theta)
        C = A * B
        D = (1 - C) ** (1 / theta - 2)
        num = (theta - 1) * one_minus_u ** (theta - 2) * one_minus_v ** (theta - 2)
        num *= theta * D * (C + (theta - 1) * C * np.log(1 - C))
        denom = (A * B) ** 2
        return num / denom

    def sample(self, n, param=None):
        if param is None:
            param = self.parameters
        theta = param[0]
        u = np.random.rand(n)
        v = np.random.rand(n)
        u_trans = 1 - (1 - u) ** (1 / theta)
        v_trans = 1 - (1 - v) ** (1 / theta)
        return np.column_stack((u_trans, v_trans))

    def kendall_tau(self, param=None):
        if param is None:
            param = self.parameters
        theta = param[0]
        return 1 - 2 / (theta + 2)