import numpy as np
from scipy.optimize import root_scalar
from scipy.integrate import quad
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


class BB5Copula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """
    BB5 Copula (Joe's two-parameter extreme-value copula).

    Attributes:
        name (str): Human-readable name of the copula.
        type (str): Identifier for the copula family.
        bounds_param (list of tuple): Bounds for copula parameters [theta, delta].
        parameters (np.ndarray): Current copula parameters.
        default_optim_method (str): Optimization method.
    """

    def __init__(self):
        """Initialize BB5 copula with default parameters theta=1.0, delta=1.0."""
        super().__init__()
        self.name = "BB5 Copula"
        self.type = "bb5"
        self.bounds_param = [(1.0, None), (1e-6, None)]  # [theta, delta]
        self.param_names = ["theta", "delta"]
        self.parameters = [1.0, 1.0]
        self.default_optim_method = "Powell"

    def get_cdf(self, u, v, param=None):
        if param is None:
            param = self.parameters
        theta, delta = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        x = -np.log(u)
        y = -np.log(v)
        xt = x ** theta
        yt = y ** theta
        S = x ** (-delta * theta) + y ** (-delta * theta)
        xyp = S ** (-1.0 / delta)
        w = xt + yt - xyp
        g = w ** (1.0 / theta)
        return np.exp(-g)

    def get_pdf(self, u, v, param=None):
        if param is None:
            param = self.parameters
        theta, delta = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        x = -np.log(u)
        y = -np.log(v)
        xt = x ** theta
        yt = y ** theta
        xdt = x ** (-delta * theta)
        ydt = y ** (-delta * theta)
        S = xdt + ydt
        xyp = S ** (-1.0 / delta)
        w = xt + yt - xyp
        g = w ** (1.0 / theta)
        C = np.exp(-g)
        dx = -1.0 / u
        dxt_du = theta * x ** (theta - 1) * dx
        dxdt_du = -delta * theta * x ** (-delta * theta - 1) * dx
        dw_du = dxt_du - (-(1.0 / delta) * S ** (-1.0 / delta - 1) * dxdt_du)
        dyt_dv = theta * y ** (theta - 1) * (-1.0 / v)
        dydt_dv = -delta * theta * y ** (-delta * theta - 1) * (-1.0 / v)
        dw_dv = dyt_dv - (-(1.0 / delta) * S ** (-1.0 / delta - 1) * dydt_dv)
        w_uv = -((1.0 / delta) * (1.0 / delta + 1.0) * S ** (-1.0 / delta - 2) * dxdt_du * dydt_dv)
        dg_dw = (1.0 / theta) * w ** (1.0 / theta - 1)
        d2g_dw2 = (1.0 / theta) * (1.0 / theta - 1.0) * w ** (1.0 / theta - 2)
        term1 = -C * dg_dw * w_uv
        term2 = -C * d2g_dw2 * dw_du * dw_dv
        term3 = C * (dg_dw ** 2) * dw_du * dw_dv
        return term1 + term2 + term3

    def kendall_tau(self, param=None):
        if param is None:
            param = self.parameters
        theta, delta = param
        def A(t):
            return (t ** theta + (1 - t) ** theta - (t ** (-delta * theta) + (1 - t) ** (-delta * theta)) ** (-1.0 / delta)) ** (1.0 / theta)
        integral, _ = quad(A, 0.0, 1.0)
        return 1.0 - 4.0 * integral

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        if param is None:
            param = self.parameters
        theta, delta = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        x = -np.log(u)
        y = -np.log(v)
        S = x ** (-delta * theta) + y ** (-delta * theta)
        w = x ** theta + y ** theta - S ** (-1.0 / delta)
        C = np.exp(-w ** (1.0 / theta))
        dg_dw = (1.0 / theta) * w ** (1.0 / theta - 1)
        dxt_du = theta * x ** (theta - 1) * (-1.0 / u)
        dxdt_du = -delta * theta * x ** (-delta * theta - 1) * (-1.0 / u)
        dw_du = dxt_du - (-(1.0 / delta) * S ** (-1.0 / delta - 1) * dxdt_du)
        return -C * dg_dw * dw_du

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        return self.partial_derivative_C_wrt_u(v, u, param)

    def conditional_cdf_u_given_v(self, u, v, param=None):
        return self.partial_derivative_C_wrt_v(u, v, param) / self.partial_derivative_C_wrt_v(1.0, v, param)

    def conditional_cdf_v_given_u(self, u, v, param=None):
        return self.partial_derivative_C_wrt_u(u, v, param) / self.partial_derivative_C_wrt_u(u, 1.0, param)

    def sample(self, n, param=None):
        if param is None:
            param = self.parameters
        u = np.random.rand(n)
        v = np.empty(n)
        for i in range(n):
            p = np.random.rand()
            sol = root_scalar(
                lambda vv: self.conditional_cdf_v_given_u(u[i], vv, param) - p,
                bracket=[1e-6, 1 - 1e-6], method='bisect', xtol=1e-6
            )
            v[i] = sol.root
        return np.column_stack((u, v))

    def LTDC(self, param=None):
        return 0.0

    def UTDC(self, param=None):
        if param is None:
            param = self.parameters
        delta = param[1]
        return 2.0 - 2.0 ** (1.0 / delta)

    def IAD(self, data):
        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan