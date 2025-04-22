import numpy as np
from scipy.special import spence
from scipy.stats import uniform
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


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
        if np.isclose(theta, 0.0):
            return u * v
        num = (np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1)
        denom = np.exp(-theta) - 1
        return -1 / theta * np.log(1 + num / denom)

    def get_pdf(self, u, v, param=None):
        if param is None:
            param = self.parameters
        theta = param[0]
        if np.isclose(theta, 0.0):
            return np.ones_like(u)
        e_theta_u = np.exp(-theta * u)
        e_theta_v = np.exp(-theta * v)
        num = theta * e_theta_u * e_theta_v * (1 - np.exp(-theta))
        denom = (1 - np.exp(-theta) + (e_theta_u - 1) * (e_theta_v - 1)) ** 2
        return num / denom

    def sample(self, n, param=None):
        if param is None:
            theta = self.parameters[0]
        else:
            theta = float(param[0])
        # independent case
        if abs(theta) < 1e-8:
            u = uniform.rvs(size=n)
            v = uniform.rvs(size=n)
            return np.column_stack((u, v))

        u = uniform.rvs(size=n)
        w = uniform.rvs(size=n)
        exp_neg_t = np.exp(-theta)
        exp_neg_t_u = np.exp(-theta * u)
        numerator = np.log(1 - w * (1 - exp_neg_t))
        denominator = exp_neg_t_u - 1
        denominator = np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
        v = -1.0 / theta * np.log(1 + numerator / denominator)
        return np.column_stack((u, v))

    def kendall_tau(self, param=None):
        if param is None:
            param = self.parameters
        theta = param[0]
        if np.isclose(theta, 0.0):
            return 0.0
        return 1 + 4 / theta * (1 - spence(1 - np.exp(-theta)) / np.exp(-theta))

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
        num = theta * np.exp(-theta * u) * (np.exp(-theta * v) - 1)
        denom = (np.exp(-theta) - 1 + (np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1))
        return num / denom

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        return self.partial_derivative_C_wrt_u(v, u, param)

    def conditional_cdf_u_given_v(self, u, v, param=None):
        return self.partial_derivative_C_wrt_v(u, v, param)

    def conditional_cdf_v_given_u(self, u, v, param=None):
        return self.partial_derivative_C_wrt_u(u, v, param)