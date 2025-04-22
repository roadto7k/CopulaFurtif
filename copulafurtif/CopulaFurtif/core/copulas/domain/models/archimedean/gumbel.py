import numpy as np
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


class GumbelCopula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    def __init__(self):
        super().__init__()
        self.name = "Gumbel Copula"
        self.type = "gumbel"
        self.bounds_param = [(1.01, 30.0)]
        self._parameters = np.array([2.0])
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
        log_u = -np.log(u)
        log_v = -np.log(v)
        sum_pow = (log_u ** theta + log_v ** theta) ** (1 / theta)
        return np.exp(-sum_pow)

    def get_pdf(self, u, v, param=None):
        if param is None:
            param = self.parameters
        theta = param[0]
        log_u = -np.log(u)
        log_v = -np.log(v)
        A = log_u ** theta + log_v ** theta
        C = A ** (1 / theta)
        cdf = np.exp(-C)

        part1 = cdf * (log_u * log_v) ** (theta - 1)
        part2 = (theta - 1) * A ** (-2 + 2 / theta)
        part3 = (theta + 1 - theta * (log_u ** theta / A)) * (theta + 1 - theta * (log_v ** theta / A))
        denom = u * v * A ** (2 - 2 / theta)

        return part1 * (part2 + part3) / denom

    def sample(self, n, param=None):
        if param is None:
            param = self.parameters
        theta = param[0]
        from scipy.stats import expon
        e = expon.rvs(size=(n, 2))
        w = expon.rvs(size=n)
        t = (e[:, 0] / w) ** (1 / theta)
        s = (e[:, 1] / w) ** (1 / theta)
        u = np.exp(-t)
        v = np.exp(-s)
        return np.column_stack((u, v))

    def kendall_tau(self, param=None):
        if param is None:
            param = self.parameters
        theta = param[0]
        return 1 - 1 / theta

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
        log_u = -np.log(u)
        log_v = -np.log(v)
        A = log_u ** theta + log_v ** theta
        C = A ** (1 / theta)
        return np.exp(-C) * log_u ** (theta - 1) * (log_v ** theta + log_u ** theta) ** (1 / theta - 1) / u

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        return self.partial_derivative_C_wrt_u(v, u, param)

    def conditional_cdf_u_given_v(self, u, v, param=None):
        return self.partial_derivative_C_wrt_v(u, v, param)

    def conditional_cdf_v_given_u(self, u, v, param=None):
        return self.partial_derivative_C_wrt_u(u, v, param)
