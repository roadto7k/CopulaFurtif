import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


class TawnT1Copula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """
    Tawn Type-1 (asymmetric logistic) extreme-value copula.

    Attributes:
        name (str): Human-readable name.
        type (str): Internal identifier.
        bounds_param (list): Bounds for [theta, beta].
        parameters (np.ndarray): Current parameters.
        default_optim_method (str): Optimization method.
    """

    def __init__(self):
        super().__init__()
        self.name = "Tawn Type-1 Copula"
        self.type = "tawn1"
        self.bounds_param = [(1.0, None), (0.0, 1.0)]  # [theta, delta]
        self.param_names = ["theta", "delta"]
        self.parameters = [2.0, 0.5]
        self.default_optim_method = "Powell"

    def _A(self, t, param=None):
        if param is None:
            param = self.parameters
        theta, beta = param
        return (1 - beta) * t + ((1 - t)**theta + (beta * t)**theta)**(1.0 / theta)

    def _A_prime(self, t, param=None):
        if param is None:
            param = self.parameters
        theta, beta = param
        h = (1 - t)**theta + (beta * t)**theta
        hp = -theta * (1 - t)**(theta - 1) + theta * beta * (beta * t)**(theta - 1)
        return (1 - beta) + (1.0 / theta) * h**(1.0 / theta - 1) * hp

    def _A_double(self, t, param=None):
        if param is None:
            param = self.parameters
        theta, beta = param
        h = (1 - t)**theta + (beta * t)**theta
        hp = -theta * (1 - t)**(theta - 1) + theta * beta * (beta * t)**(theta - 1)
        hpp = theta * (theta - 1) * ((1 - t)**(theta - 2) + beta**theta * t**(theta - 2))
        term1 = (1.0 / theta) * (1.0 / theta - 1) * h**(1.0 / theta - 2) * hp**2
        term2 = (1.0 / theta) * h**(1.0 / theta - 1) * hpp
        return term1 + term2

    def get_cdf(self, u, v, param=None):
        if param is None:
            param = self.parameters
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        x, y = -np.log(u), -np.log(v)
        s = x + y
        t = x / s
        return np.exp(-s * self._A(t, param))

    def get_pdf(self, u, v, param=None):
        if param is None:
            param = self.parameters
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        x, y = -np.log(u), -np.log(v)
        s = x + y
        t = x / s
        A = self._A(t, param)
        Ap = self._A_prime(t, param)
        App = self._A_double(t, param)
        Lx = A + (y / s) * Ap
        Ly = A - (x / s) * Ap
        Lxy = - (x * y / s**3) * App
        C_val = np.exp(-s * A)
        return C_val * (Lx * Ly - Lxy) / (u * v)

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        if param is None:
            param = self.parameters
        x, y = -np.log(u), -np.log(v)
        s = x + y
        t = x / s
        C_val = self.get_cdf(u, v, param)
        A = self._A(t, param)
        Ap = self._A_prime(t, param)
        return C_val * (A / u + (y / (u * s)) * Ap)

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        if param is None:
            param = self.parameters
        x, y = -np.log(u), -np.log(v)
        s = x + y
        t = x / s
        C_val = self.get_cdf(u, v, param)
        A = self._A(t, param)
        Ap = self._A_prime(t, param)
        return C_val * (A / v - (x / (v * s)) * Ap)

    def conditional_cdf_v_given_u(self, u, v, param=None):
        return self.partial_derivative_C_wrt_u(u, v, param)

    def conditional_cdf_u_given_v(self, u, v, param=None):
        return self.partial_derivative_C_wrt_v(u, v, param)

    def sample(self, n, param=None):
        if param is None:
            param = self.parameters
        u = np.random.rand(n)
        v = np.empty(n)
        eps = 1e-6
        for i in range(n):
            p = np.random.rand()
            sol = root_scalar(
                lambda vv: self.conditional_cdf_v_given_u(u[i], vv, param) - p,
                bracket=[eps, 1 - eps], method="bisect", xtol=1e-6
            )
            v[i] = sol.root
        return np.column_stack((u, v))

    def kendall_tau(self, param=None):
        if param is None:
            param = self.parameters
        integral, _ = quad(lambda t: self._A(t, param), 0.0, 1.0)
        return 1.0 - 4.0 * integral

    def LTDC(self, param=None):
        return 0.0

    def UTDC(self, param=None):
        if param is None:
            param = self.parameters
        theta = param[0]
        return 2.0 - 2.0**(1.0 / theta)

    def IAD(self, data):
        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan