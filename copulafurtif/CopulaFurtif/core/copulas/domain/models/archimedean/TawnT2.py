import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar

from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


class TawnT2Copula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """
    Tawn Type-2 (asymmetric mixed) extreme-value copula.

    Parameters
    ----------
    theta : float
        Dependence strength parameter (>= 1).
    beta : float
        Asymmetry parameter in [0, 1].
    """

    def __init__(self):
        super().__init__()
        self.name = "Tawn Type-2 Copula"
        self.type = "tawn2"
        self.bounds_param = [(1.0, None), (0.0, 1.0)]  # [theta, beta]
        self.param_names = ["theta", "beta"]
        self.parameters = [2.0, 0.5]
        self.default_optim_method = "Powell"

    def _A(self, t: float, param: np.ndarray) -> float:
        """
        Compute the dependence function A(t) = (1−β)(1−t) + (t^θ + (β(1−t))^θ)^(1/θ).

        Args:
            t (float): Proportion x/(x+y) in (0,1).
            param (np.ndarray, optional): Copula parameters [θ, β]. Defaults to self.parameters.

        Returns:
            float: Value of A(t).
        """

        theta, beta = param
        return (1 - beta) * (1 - t) + (t**theta + (beta * (1 - t))**theta)**(1.0 / theta)

    def _A_prime(self, t: float, param: np.ndarray) -> float:
        """
        Compute the first derivative A′(t) of the dependence function.

        Args:
            t (float): Proportion x/(x+y) in (0,1).
            param (np.ndarray, optional): Copula parameters [θ, β]. Defaults to self.parameters.

        Returns:
            float: Value of A′(t).
        """

        theta, beta = param
        h = t**theta + (beta * (1 - t))**theta
        hp = theta * t**(theta - 1) - theta * beta * (beta * (1 - t))**(theta - 1)
        return -(1 - beta) + (1.0 / theta) * h**(1.0 / theta - 1) * hp

    def _A_double(self, t: float, param: np.ndarray) -> float:
        """
        Compute the second derivative A″(t) of the dependence function.

        Args:
            t (float): Proportion x/(x+y) in (0,1).
            param (np.ndarray, optional): Copula parameters [θ, β]. Defaults to self.parameters.

        Returns:
            float: Value of A″(t).
        """

        theta, beta = param
        h = t**theta + (beta * (1 - t))**theta
        hp = theta * t**(theta - 1) - theta * beta * (beta * (1 - t))**(theta - 1)
        hpp = theta * (theta - 1) * (t**(theta - 2) + beta**theta * (1 - t)**(theta - 2))
        term1 = (1.0 / theta) * (1.0 / theta - 1) * h**(1.0 / theta - 2) * hp**2
        term2 = (1.0 / theta) * h**(1.0 / theta - 1) * hpp
        return term1 + term2

    def get_cdf(self, u, v, param: np.ndarray = None):
        """
        Evaluate the copula cumulative distribution function at (u, v).

        Args:
            u (float or array-like): First uniform margin in (0,1).
            v (float or array-like): Second uniform margin in (0,1).
            param (np.ndarray, optional): Copula parameters [θ, β]. Defaults to self.parameters.

        Returns:
            float or np.ndarray: CDF value C(u, v).
        """

        if param is None:
            param = self.parameters
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        x, y = -np.log(u), -np.log(v)
        s = x + y
        t = x / s
        return np.exp(-s * self._A(t, param))

    def get_pdf(self, u, v, param: np.ndarray = None):
        """
        Evaluate the copula probability density function at (u, v).

        Args:
            u (float or array-like): First uniform margin in (0,1).
            v (float or array-like): Second uniform margin in (0,1).
            param (np.ndarray, optional): Copula parameters [θ, β]. Defaults to self.parameters.

        Returns:
            float or np.ndarray: PDF value c(u, v).
        """

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

    def partial_derivative_C_wrt_u(self, u, v, param: np.ndarray = None):
        """
        Compute the partial derivative ∂C/∂u of the copula CDF.

        Args:
            u (float or array-like): First margin in (0,1).
            v (float or array-like): Second margin in (0,1).
            param (np.ndarray, optional): Copula parameters [θ, β]. Defaults to self.parameters.

        Returns:
            float or np.ndarray: Value of ∂C/∂u.
        """

        if param is None:
            param = self.parameters
        C_val = self.get_cdf(u, v, param)
        x, y = -np.log(u), -np.log(v)
        s = x + y
        t = x / s
        A = self._A(t, param)
        Ap = self._A_prime(t, param)
        return C_val * (A / u + (y / (u * s)) * Ap)

    def partial_derivative_C_wrt_v(self, u, v, param: np.ndarray = None):
        """
        Compute the partial derivative ∂C/∂v of the copula CDF.

        Args:
            u (float or array-like): First margin in (0,1).
            v (float or array-like): Second margin in (0,1).
            param (np.ndarray, optional): Copula parameters [θ, β]. Defaults to self.parameters.

        Returns:
            float or np.ndarray: Value of ∂C/∂v.
        """

        if param is None:
            param = self.parameters
        C_val = self.get_cdf(u, v, param)
        x, y = -np.log(u), -np.log(v)
        s = x + y
        t = x / s
        A = self._A(t, param)
        Ap = self._A_prime(t, param)
        return C_val * (A / v - (x / (v * s)) * Ap)

    def conditional_cdf_v_given_u(self, u, v, param: np.ndarray = None):
        """
        Compute the conditional CDF P(V ≤ v | U = u).

        Args:
            u (float or array-like): Conditioning value of U in (0,1).
            v (float or array-like): Value of V in (0,1).
            param (np.ndarray, optional): Copula parameters [θ, β]. Defaults to self.parameters.

        Returns:
            float or np.ndarray: Conditional CDF of V given U.
        """

        return self.partial_derivative_C_wrt_u(u, v, param)

    def conditional_cdf_u_given_v(self, u, v, param: np.ndarray = None):
        """
        Compute the conditional CDF P(U ≤ u | V = v).

        Args:
            u (float or array-like): Value of U in (0,1).
            v (float or array-like): Conditioning value of V in (0,1).
            param (np.ndarray, optional): Copula parameters [θ, β]. Defaults to self.parameters.

        Returns:
            float or np.ndarray: Conditional CDF of U given V.
        """

        return self.partial_derivative_C_wrt_v(u, v, param)

    def sample(self, n: int, param: np.ndarray = None) -> np.ndarray:
        """
        Generate random samples from the copula using conditional inversion.

        Args:
            n (int): Number of samples to generate.
            param (np.ndarray, optional): Copula parameters [θ, β]. Defaults to self.parameters.

        Returns:
            np.ndarray: Array of shape (n, 2) with uniform samples on [0,1]^2.
        """

        if param is None:
            param = self.parameters
        eps = 1e-6
        u = np.random.rand(n)
        v = np.empty(n)
        for i in range(n):
            p = np.random.rand()
            sol = root_scalar(
                lambda vv: self.conditional_cdf_v_given_u(u[i], vv, param) - p,
                bracket=[eps, 1 - eps], method="bisect", xtol=1e-6
            )
            v[i] = sol.root
        return np.column_stack((u, v))

    def kendall_tau(self, param: np.ndarray = None) -> float:
        """
        Compute theoretical Kendall’s tau implied by the copula.

        Args:
            param (np.ndarray, optional): Copula parameters [θ, β]. Defaults to self.parameters.

        Returns:
            float: Kendall’s tau = 1 − 4 ∫₀¹ A(t) dt.
        """

        if param is None:
            param = self.parameters
        integral, _ = quad(lambda t: self._A(t, param), 0.0, 1.0)
        return 1.0 - 4.0 * integral

    def LTDC(self, param: np.ndarray = None) -> float:
        """
        Compute the lower tail dependence coefficient (LTDC).

        Args:
            param (np.ndarray, optional): Copula parameters [θ, β]. Defaults to self.parameters.

        Returns:
            float: LTDC value (0.0 for this copula).
        """

        return 0.0

    def UTDC(self, param: np.ndarray = None) -> float:
        """
        Compute the upper tail dependence coefficient (UTDC).

        Args:
            param (np.ndarray, optional): Copula parameters [θ, β]. Defaults to self.parameters.

        Returns:
            float: UTDC value = 2 − 2^(1/θ).
        """

        if param is None:
            param = self.parameters
        theta = param[0]
        return 2.0 - 2.0**(1.0 / theta)
