import numpy as np
from scipy.optimize import root_scalar
from scipy.integrate import trapezoid
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel, CopulaParameters
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


class BB6Copula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """
    BB6 Copula (Archimedean with generator φ(t) = [-ln(1-(1-t)ˆθ)]ˆ(1/δ)).

    Attributes:
        name (str): Human-readable name of the copula.
        type (str): Identifier for the copula family.
        bounds_param (list of tuple): Bounds for copula parameters [theta, delta].
        parameters (np.ndarray): Current copula parameters.
        default_optim_method (str): Optimization method.
    """

    def __init__(self):
        """Initialize BB6 copula with default parameters theta=2.0, delta=2.0."""
        super().__init__()
        self.name = "BB6 Copula"
        self.type = "bb6"
        self.default_optim_method = "Powell"
        self.init_parameters(CopulaParameters([2.0, 2.0], [(1.0, None), (1.0, None)], ["theta", "delta"]))

    def _phi(self, t, theta, delta):
        """
        Compute the generator function φ(t) = (−log(1 − (1 − t)^θ))^(1/δ).

        Args:
            t (float or array-like): Input variable in [0,1].
            theta (float): Copula parameter θ.
            delta (float): Copula parameter δ.

        Returns:
            float or numpy.ndarray: Value of φ(t).
        """

        return (-np.log(1.0 - (1.0 - t)**theta))**(1.0 / delta)

    def _phi_prime(self, t, theta, delta):
        """
        Compute the derivative φ′(t) of the generator function.

        Args:
            t (float or array-like): Input variable in [0,1].
            theta (float): Copula parameter θ.
            delta (float): Copula parameter δ.

        Returns:
            float or numpy.ndarray: Value of φ′(t).
        """

        g = 1.0 - (1.0 - t)**theta
        gp = theta * (1.0 - t)**(theta - 1)
        L = -np.log(g)
        Lp = -gp / g
        return (1.0 / delta) * L**(1.0 / delta - 1.0) * Lp

    def get_cdf(self, u, v, param=None):
        """
        Evaluate the copula cumulative distribution function C(u, v).

        Args:
            u (float or array-like): First uniform margin in (0,1).
            v (float or array-like): Second uniform margin in (0,1).
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float or numpy.ndarray: CDF value C(u, v).
        """

        if param is None:
            param = self.get_parameters()
        theta, delta = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        ubar, vbar = 1.0 - u, 1.0 - v
        x = -np.log(1.0 - ubar**theta)
        y = -np.log(1.0 - vbar**theta)
        sm = x**delta + y**delta
        tem = sm**(1.0 / delta)
        w = np.exp(-tem)
        return 1.0 - (1.0 - w)**(1.0 / theta)

    def get_pdf(self, u, v, param=None):
        """
        Compute the partial derivative ∂C(u,v)/∂u of the copula CDF.

        Args:
            u (float or array-like): First margin in (0,1).
            v (float or array-like): Second margin in (0,1).
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float or numpy.ndarray: Value of ∂C/∂u at (u, v).
        """

        if param is None:
            param = self.get_parameters()
        theta, delta = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        ubar, vbar = 1.0 - u, 1.0 - v
        zu = 1.0 - ubar**theta
        zv = 1.0 - vbar**theta
        x = -np.log(zu)
        y = -np.log(zv)
        xd, yd = x**delta, y**delta
        sm = xd + yd
        tem = sm**(1.0 / delta)
        w = np.exp(-tem)
        prefac = (1.0 - w)**(1.0/theta - 2.0) * w * (tem / sm**2) * (xd / x) * (yd / y)
        bracket = (theta - w) * tem + theta * (delta - 1.0) * (1.0 - w)
        jac = (1.0 - zu) * (1.0 - zv) / (zu * zv * ubar * vbar)
        return prefac * bracket * jac

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """
        Compute the partial derivative ∂C(u,v)/∂u of the copula CDF.

        Args:
            u (float or array-like): First margin in (0,1).
            v (float or array-like): Second margin in (0,1).
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float or numpy.ndarray: Value of ∂C/∂u at (u, v).
        """


        if param is None:
            param = self.get_parameters()
        theta, delta = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        ubar, vbar = 1.0 - u, 1.0 - v
        x = -np.log(1.0 - ubar**theta)
        y = -np.log(1.0 - vbar**theta)
        xd, yd = x**delta, y**delta
        sm = xd + yd
        tem = sm**(1.0 / delta)
        w = np.exp(-tem)
        C = 1.0 - (1.0 - w)**(1.0 / theta)
        dC_dw = (1.0 / theta) * (1.0 - w)**(1.0 / theta - 1.0)
        dC_dtem = dC_dw * (-w)
        dtem_dsm = (1.0 / delta) * sm**(1.0 / delta - 1.0)
        dsm_dx = delta * x**(delta - 1.0)
        dC_dx = dC_dtem * dtem_dsm * dsm_dx
        dx_du = -theta * ubar**(theta - 1.0) / (1.0 - ubar**theta)
        return dC_dx * dx_du

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """
        Compute the partial derivative ∂C(u,v)/∂v of the copula CDF.

        Args:
            u (float or array-like): First margin in (0,1).
            v (float or array-like): Second margin in (0,1).
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float or numpy.ndarray: Value of ∂C/∂v at (u, v).
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
            float or numpy.ndarray: Conditional CDF of U given V.
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
            float or numpy.ndarray: Conditional CDF of V given U.
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
        eps = 1e-6
        u = np.random.rand(n)
        v = np.empty(n)
        for i in range(n):
            p = np.random.rand()
            root = root_scalar(
                lambda vi: self.conditional_cdf_v_given_u(u[i], vi, param) - p,
                bracket=[eps, 1 - eps], method='bisect', xtol=1e-6
            )
            v[i] = root.root
        return np.column_stack((u, v))

    def kendall_tau(self, param=None, n=1001):
        """
        Compute the lower tail dependence coefficient (LTDC) of the copula.

        Args:
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float: LTDC value (0.0 for this copula).
        """

        if param is None:
            param = self.get_parameters()
        theta, delta = param
        t = np.linspace(0.0, 1.0, n)[1:-1]
        phi_vals = self._phi(t, theta, delta)
        phi_p_vals = self._phi_prime(t, theta, delta)
        integrand = phi_vals / phi_p_vals
        integral = trapezoid(integrand, t)
        return 1.0 + 4.0 * integral

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
        delta = param[1]
        return 2.0 - 2.0 ** (1.0 / delta)

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