import numpy as np
from scipy.optimize import root_scalar
from scipy.integrate import quad
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel, CopulaParameters
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
        self.default_optim_method = "Powell"
        self.init_parameters(CopulaParameters([1.0, 1.0], [(1.0, None), (1e-6, None)], ["theta", "delta"]))
        
    def get_cdf(self, u, v, param=None):
        """
        Evaluate the copula cumulative distribution function at (u, v).

        Args:
            u (float or array-like): First uniform margin in (0,1).
            v (float or array-like): Second uniform margin in (0,1).
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float or np.ndarray: CDF value C(u, v).
        """

        if param is None:
            param = self.get_parameters()
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
        """
        Evaluate the copula probability density function at (u, v).

        Args:
            u (float or array-like): First uniform margin in (0,1).
            v (float or array-like): Second uniform margin in (0,1).
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float or np.ndarray: PDF value c(u, v).
        """

        if param is None:
            param = self.get_parameters()
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
        """
        Compute Kendall’s tau implied by the copula via numerical integration.

        Args:
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float: Theoretical Kendall’s tau (1 − 4 ∫₀¹ A(t) dt).
        """

        if param is None:
            param = self.get_parameters()
        theta, delta = param
        def A(t):
            return (t ** theta + (1 - t) ** theta - (t ** (-delta * theta) + (1 - t) ** (-delta * theta)) ** (-1.0 / delta)) ** (1.0 / theta)
        integral, _ = quad(A, 0.0, 1.0)
        return 1.0 - 4.0 * integral

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """
        Compute the partial derivative ∂C(u,v)/∂u of the copula CDF.

        Args:
            u (float or array-like): First margin in (0,1).
            v (float or array-like): Second margin in (0,1).
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float or np.ndarray: Value of ∂C/∂u at (u, v).
        """

        if param is None:
            param = self.get_parameters()
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
        """
        Compute the partial derivative ∂C(u,v)/∂v of the copula CDF.

        Args:
            u (float or array-like): First margin in (0,1).
            v (float or array-like): Second margin in (0,1).
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float or np.ndarray: Value of ∂C/∂v at (u, v).
        """

        return self.partial_derivative_C_wrt_u(v, u, param)

    def conditional_cdf_u_given_v(self, u, v, param=None):
        """
        Compute the conditional CDF P(U ≤ u | V = v).

        Args:
            u (float or array-like): Value of U in (0,1).
            v (float or array-like): Conditioning value of V in (0,1).
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float or np.ndarray: Conditional CDF of U given V.
        """

        return self.partial_derivative_C_wrt_v(u, v, param) / self.partial_derivative_C_wrt_v(1.0, v, param)

    def conditional_cdf_v_given_u(self, u, v, param=None):
        """
        Compute the conditional CDF P(V ≤ v | U = u).

        Args:
            u (float or array-like): Conditioning value of U in (0,1).
            v (float or array-like): Value of V in (0,1).
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float or np.ndarray: Conditional CDF of V given U.
        """

        return self.partial_derivative_C_wrt_u(u, v, param) / self.partial_derivative_C_wrt_u(u, 1.0, param)

    def sample(self, n, param=None):
        """
        Generate random samples from the copula using conditional inversion.

        Args:
            n (int): Number of samples to generate.
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            numpy.ndarray: Array of shape (n, 2) with uniform samples on [0,1]^2.
        """

        if param is None:
            param = self.get_parameters()
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
        """
        Compute the lower tail dependence coefficient (LTDC) of the copula.

        Args:
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float: LTDC value (0.0 for this copula).
        """

        return 0.0

    def UTDC(self, param=None):
        """
        Compute the upper tail dependence coefficient (UTDC) of the copula.

        Args:
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float: UTDC value (2 − 2^(1/δ)).
        """

        if param is None:
            param = self.get_parameters()

        theta = param[0]
        delta = param[1]
        return 2.0 - (2.0 - 2.0 ** (-1.0 / delta)) ** (1.0 / theta)

    def IAD(self, data):
        """
        Return NaN for the Integrated Anderson-Darling (IAD) statistic.

        Args:
            data (Sequence[array-like, array-like]): Ignored pseudo-observations.

        Returns:
            float: Always returns numpy.nan.
        """

        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        """
        Return NaN for the Anderson-Darling (AD) statistic.

        Args:
            data (Sequence[array-like, array-like]): Ignored pseudo-observations.

        Returns:
            float: Always returns numpy.nan.
        """

        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan