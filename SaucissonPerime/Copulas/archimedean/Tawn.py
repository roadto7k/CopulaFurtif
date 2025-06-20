import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar

from SaucissonPerime.Copulas.base import BaseCopula


class TawnCopula(BaseCopula):
    """
    Tawn 3-parameter (asymmetric mixed) extreme-value copula.

    Parameters
    ----------
    theta : float
        Dependence strength parameter (>= 1).
    psi1 : float
        Asymmetry parameter on the t margin (psi1), in [0, 1].
    psi2 : float
        Asymmetry parameter on the (1 - t) margin (psi2), in [0, 1].

    The Pickands dependence function is:
        A(t) = (1 - psi2)*(1 - t)
             + (1 - psi1)*t
             + [ (psi1*(1 - t))**theta + (psi2*t)**theta ]**(1/theta)

    The copula CDF is defined by:
        C(u, v) = exp(-ℓ(-ln u, -ln v)),
    where ℓ(x, y) = (x + y) * A( y / (x + y) ).
    """

    def __init__(self):
        super().__init__()
        self.type = "tawn3"
        self.name = "Tawn 3-Param Copula"
        self.bounds_param = [(1.0, None), (0.0, 1.0), (0.0, 1.0)]
        self._parameters = np.array([2.0, 0.5, 0.5])  # [theta, psi1, psi2]
        self.default_optim_method = "Powell"

    @property
    def parameters(self) -> np.ndarray:
        return self._parameters

    @parameters.setter
    def parameters(self, param: np.ndarray):
        self._parameters = np.asarray(param)

    def _A(self, t: float, param: np.ndarray = None) -> float:
        """
        Pickands dependence function A(t).
        """
        if param is None:
            param = self.parameters
        theta, psi1, psi2 = param
        term_linear = (1 - psi2) * (1 - t) + (1 - psi1) * t
        term_power = ((psi1 * (1 - t))**theta + (psi2 * t)**theta)**(1.0 / theta)
        return term_linear + term_power

    def _A_prime(self, t: float, param: np.ndarray = None) -> float:
        """
        First derivative A'(t).
        """
        if param is None:
            param = self.parameters
        theta, psi1, psi2 = param
        h = (psi1 * (1 - t))**theta + (psi2 * t)**theta
        hp = -theta * psi1 * (psi1 * (1 - t))**(theta - 1) + theta * psi2 * (psi2 * t)**(theta - 1)
        linear_deriv = -(1 - psi2) + (1 - psi1)
        return linear_deriv + (1.0 / theta) * h**(1.0 / theta - 1) * hp

    def _A_double(self, t: float, param: np.ndarray = None) -> float:
        """
        Second derivative A''(t).
        """
        if param is None:
            param = self.parameters
        theta, psi1, psi2 = param
        h = (psi1 * (1 - t))**theta + (psi2 * t)**theta
        hp = -theta * psi1 * (psi1 * (1 - t))**(theta - 1) + theta * psi2 * (psi2 * t)**(theta - 1)
        hpp = theta * (theta - 1) * (
            psi1**2 * (psi1 * (1 - t))**(theta - 2)
            + psi2**2 * (psi2 * t)**(theta - 2)
        )
        term1 = (1.0 / theta) * (1.0 / theta - 1) * h**(1.0 / theta - 2) * hp**2
        term2 = (1.0 / theta) * h**(1.0 / theta - 1) * hpp
        return term1 + term2

    def get_cdf(self, u: np.ndarray, v: np.ndarray, param: np.ndarray = None) -> np.ndarray:
        """
        Copula CDF: C(u,v) = exp(-ℓ(-ln u, -ln v)).
        """
        if param is None:
            param = self.parameters
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        x, y = -np.log(u), -np.log(v)
        s = x + y
        t = y / s
        return np.exp(-s * self._A(t, param))

    def get_pdf(self, u: np.ndarray, v: np.ndarray, param: np.ndarray = None) -> np.ndarray:
        """
        Copula PDF via derivatives of the stable-tail dependence function.
        """
        if param is None:
            param = self.parameters
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        x, y = -np.log(u), -np.log(v)
        s = x + y
        t = y / s

        A_val = self._A(t, param)
        Ap = self._A_prime(t, param)
        App = self._A_double(t, param)

        Lx = A_val + (y / s) * Ap
        Ly = A_val - (x / s) * Ap
        Lxy = - (x * y / s**3) * App
        C_val = np.exp(-s * A_val)
        return C_val * (Lx * Ly - Lxy) / (u * v)

    def partial_derivative_C_wrt_u(self, u: np.ndarray, v: np.ndarray, param: np.ndarray = None) -> np.ndarray:
        """
        Partial derivative ∂C/∂u for conditional sampling.
        """
        if param is None:
            param = self.parameters
        x, y = -np.log(u), -np.log(v)
        s = x + y
        t = y / s

        C_val = self.get_cdf(u, v, param)
        A_val = self._A(t, param)
        Ap = self._A_prime(t, param)

        return C_val * (A_val / u - (y / (u * s)) * Ap)

    def partial_derivative_C_wrt_v(self, u: np.ndarray, v: np.ndarray, param: np.ndarray = None) -> np.ndarray:
        """
        Partial derivative ∂C/∂v for conditional sampling.
        """
        if param is None:
            param = self.parameters
        x, y = -np.log(u), -np.log(v)
        s = x + y
        t = y / s

        C_val = self.get_cdf(u, v, param)
        A_val = self._A(t, param)
        Ap = self._A_prime(t, param)

        return C_val * (A_val / v + (x / (v * s)) * Ap)

    def conditional_cdf_v_given_u(self, u: np.ndarray, v: np.ndarray, param: np.ndarray = None) -> np.ndarray:
        """
        Conditional CDF P(V <= v | U = u).
        """
        if param is None:
            param = self.parameters
        return self.partial_derivative_C_wrt_u(u, v, param)

    def conditional_cdf_u_given_v(self, u: np.ndarray, v: np.ndarray, param: np.ndarray = None) -> np.ndarray:
        """
        Conditional CDF P(U <= u | V = v).
        """
        if param is None:
            param = self.parameters
        return self.partial_derivative_C_wrt_v(u, v, param)

    def sample(self, n: int, param: np.ndarray = None) -> np.ndarray:
        """
        Generate n samples (u,v) via conditional inversion.
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
        Compute Kendall's tau: τ = 1 - 4 ∫₀¹ A(t) dt.
        """
        if param is None:
            param = self.parameters
        integral, _ = quad(lambda t: self._A(t, param), 0.0, 1.0)
        return 1.0 - 4.0 * integral

    def LTDC(self, param: np.ndarray = None) -> float:
        """
        Lower-tail dependence coefficient λ_L = 0.
        """
        # param not needed
        return 0.0

    def UTDC(self, param: np.ndarray = None) -> float:
        """
        Upper-tail dependence coefficient λ_U = 2 - 2^(1/θ).
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        return 2.0 - 2.0**(1.0 / theta)
