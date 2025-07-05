from typing import Union

import numpy as np
from numpy.random import default_rng
from scipy import integrate

from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel, CopulaParameters
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


class BB2Copula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """
    BB2 Copula (Survival version of BB1 Copula).

    This copula is the survival (180-degree rotated) version of BB1, used to
    model upper tail dependence with two parameters.

    Attributes:
        name (str): Human-readable name of the copula.
        type (str): Identifier for the copula family.
        bounds_param (list of tuple): Bounds for copula parameters [theta, delta].
        parameters (np.ndarray): Current copula parameters.
        default_optim_method (str): Optimization method for parameter fitting.
    """

    def __init__(self):
        """Initialize BB2 copula with default parameters."""
        super().__init__()
        self.name = "BB2 Copula"
        self.type = "bb2"
        # self.bounds_param = [(0.05, 30.0), (1.0, 10.0)]  # [theta, delta]
        # self.param_names = ["theta", "delta"]
        # self.parameters = [2, 1.5]
        self.default_optim_method = "Powell"
        self.init_parameters(CopulaParameters([2, 1.5],[(0.05, 30.0), (1.0, 10.0)], ["theta", "delta"] ))

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _phi(s: Union[np.ndarray, float], theta: float, delta: float):
        """Generator φ(s; theta, delta) (Joe 2014, Eq. 4.57)."""
        return (1.0 + (1.0 / delta) * np.log1p(s)) ** (-1.0 / theta)

    @staticmethod
    def _phi_inv(t: Union[np.ndarray, float], theta: float, delta: float):
        """Inverse generator φ⁻¹(t; theta, delta)."""
        return np.expm1(delta * (t ** (-theta) - 1.0))

    @staticmethod
    def _phi_prime(s: Union[np.ndarray, float], theta: float, delta: float):
        """First derivative φ′(s)."""
        A = 1.0 + (1.0 / delta) * np.log1p(s)
        return -(1.0 / (theta * delta)) * A ** (-1.0 / theta - 1.0) / (1.0 + s)

    @staticmethod
    def _phi_double_prime(s: Union[np.ndarray, float], theta: float, delta: float):
        """Second derivative φ″(s) (sign‑corrected)."""
        A = 1.0 + (1.0 / delta) * np.log1p(s)
        term1 = 1.0 / (theta * delta * (1.0 + s) ** 2) * A ** (-1.0 / theta - 1.0)
        p = 1.0 / theta + 1.0
        term2 = p / (theta * delta ** 2 * (1.0 + s) ** 2) * A ** (-1.0 / theta - 2.0)
        return term1 + term2

    def _invert_conditional_v(
            self,
            u: np.ndarray,
            p: np.ndarray,
            theta: float,
            delta: float,
            *,
            eps: float = 1e-12,
            max_iter: int = 40,
    ) -> np.ndarray:
        """Bisection solver for V | U=u defined by ∂C/∂u(u, v)=p."""
        lo = np.full_like(u, eps)
        hi = np.full_like(u, 1.0 - eps)
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            val = self.partial_derivative_C_wrt_u(u, mid, [theta, delta])
            lo = np.where(val < p, mid, lo)
            hi = np.where(val >= p, mid, hi)
        return 0.5 * (lo + hi)

    def get_cdf(self, u, v, param=None):
        """
        Compute the BB2 copula CDF.

        Args:
            u (float or np.ndarray): First input in (0,1).
            v (float or np.ndarray): Second input in (0,1).
            param (np.ndarray, optional): Parameters [theta, delta].

        Returns:
            float or np.ndarray: Copula CDF values.
        """
        if param is None:
            param = self.get_parameters()
        theta, delta = param
        eps = 1e-12
        u = np.clip(u, eps, 1.0 - eps)
        v = np.clip(v, eps, 1.0 - eps)

        return self._phi(
            self._phi_inv(u, theta, delta) + self._phi_inv(v, theta, delta),
            theta,
            delta,
        )

    def get_pdf(self, u, v, param=None):
        """
        Compute the BB2 copula PDF.

        Args:
            u (float or np.ndarray): First input in (0,1).
            v (float or np.ndarray): Second input in (0,1).
            param (np.ndarray, optional): Parameters [theta, delta].

        Returns:
            float or np.ndarray: Copula PDF values.
        """
        if param is None:
            param = self.get_parameters()
        theta, delta = param
        eps = 1e-12
        u = np.clip(u, eps, 1.0 - eps)
        v = np.clip(v, eps, 1.0 - eps)

        x = self._phi_inv(u, theta, delta)
        y = self._phi_inv(v, theta, delta)
        s = x + y

        dx_du = -theta * delta * u ** (-theta - 1.0) * (x + 1.0)
        dy_dv = -theta * delta * v ** (-theta - 1.0) * (y + 1.0)

        pdf = self._phi_double_prime(s, theta, delta) * dx_du * dy_dv
        return np.nan_to_num(pdf, nan=0.0, neginf=0.0, posinf=np.inf)  # guard tiny negative round‑offs

    def kendall_tau(self, param=None):
        """
        Compute Kendall's tau.

        Args:
            param (np.ndarray, optional): Parameters [theta, delta].

        Returns:
            float: Kendall's tau.
        """
        if param is None:
            param = self.get_parameters()
        theta, delta = param

        def integrand(t):
            t_safe = np.clip(t, 1e-10, 1.0 - 1e-10)
            return self._phi(2.0 * self._phi_inv(t_safe, theta, delta), theta, delta) - t_safe

        try:
            res, _ = integrate.quad(integrand, 0.0, 1.0, epsabs=1e-7, epsrel=1e-7, limit=200)
            return 1.0 + 4.0 * res
        except (OverflowError, ValueError):
            rng = default_rng(12345)
            u = rng.random(40000)
            v = rng.random(40000)
            x = self._phi_inv(u, theta, delta)
            y = self._phi_inv(v, theta, delta)
            w = self._phi(x + y, theta, delta)
            rank = np.argsort(u)
            u_rank = np.empty_like(rank)
            u_rank[rank] = np.arange(len(u))
            rank = np.argsort(w)
            v_rank = np.empty_like(rank)
            v_rank[rank] = np.arange(len(w))
            tau = 1 - 4 * np.sum(np.abs(u_rank - v_rank)) / (len(u) * (len(u) - 1))
            return tau

    def sample(self,
               n: int,
               param=None,
               rng=None,
               eps: float = 1e-12,
               max_iter: int = 40):
        """
        Generate random samples from the BB2 copula.

        Args:
            n (int): Number of samples.
            param (np.ndarray, optional): Parameters [theta, delta].

        Returns:
            np.ndarray: Array of shape (n, 2).
        """
        if rng is None:
            rng = default_rng()
        if param is None:
            theta, delta = map(float, self.get_parameters())
        else:
            theta, delta = map(float, param)

        u = rng.random(n)
        p = rng.random(n)
        v = self._invert_conditional_v(u, p, theta, delta, eps=eps, max_iter=max_iter)

        np.clip(u, eps, 1.0 - eps, out=u)
        np.clip(v, eps, 1.0 - eps, out=v)
        return np.column_stack((u, v))

    def LTDC(self, param=None):
        """
        Compute lower tail dependence coefficient.

        Args:
            param (np.ndarray, optional): Parameters [theta, delta].

        Returns:
            float: Lower tail dependence.
        """
        return 1.0

    def UTDC(self, param=None):
        """
        Compute upper tail dependence coefficient.

        Args:
            param (np.ndarray, optional): Parameters [theta, delta].

        Returns:
            float: Upper tail dependence.
        """
        return 0.0

    def partial_derivative_C_wrt_u(self, u, v, param=None, *, wrt='u'):
        """
        Compute partial derivative ∂C/∂u.

        Args:
            u (float or np.ndarray): U values.
            v (float or np.ndarray): V values.
            param (np.ndarray, optional): Parameters [theta, delta].

        Returns:
            float or np.ndarray: Partial derivative values.
        """
        if param is None:
            theta, delta = self.get_parameters()
        else:
            theta, delta = param

            # 1) vector-friendly clipping
        eps = 1.0e-12
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        uc = np.clip(u, eps, 1.0 - eps)
        vc = np.clip(v, eps, 1.0 - eps)

        # 2) raw analytic ratio  φ′(s)/φ′(x or y)
        if wrt == "u":
            x = self._phi_inv(uc, theta, delta)
            s = x + self._phi_inv(vc, theta, delta)
            raw = self._phi_prime(s, theta, delta) / self._phi_prime(x, theta, delta)
            near_zero = u <= eps * 1.01
            limit_val = vc if theta < 1.0 - 1e-12 else 0.0
        elif wrt == "v":
            y = self._phi_inv(vc, theta, delta)
            s = self._phi_inv(uc, theta, delta) + y
            raw = self._phi_prime(s, theta, delta) / self._phi_prime(y, theta, delta)
            near_zero = v <= eps * 1.01
            limit_val = uc if theta < 1.0 - 1e-12 else 0.0
        else:
            raise ValueError("`wrt` must be 'u' or 'v'.")

        # 3) numerical clean-up + correct boundary limit
        deriv = np.nan_to_num(raw, nan=0.0, neginf=0.0,
                              posinf=np.finfo(float).max)
        if np.isscalar(deriv):
            if near_zero:
                deriv = float(limit_val)
        else:
            deriv = np.where(near_zero, limit_val, deriv)

        return deriv


    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """
        Compute partial derivative ∂C/∂v.

        Args:
            u (float or np.ndarray): U values.
            v (float or np.ndarray): V values.
            param (np.ndarray, optional): Parameters [theta, delta].

        Returns:
            float or np.ndarray: Partial derivative values.
        """
        return self.partial_derivative_C_wrt_u(v, u, param)

    def conditional_cdf_u_given_v(self, u, v, param=None):
        """
        Compute conditional CDF P(U ≤ u | V = v).

        Args:
            u (float or np.ndarray): U values.
            v (float or np.ndarray): V values.
            param (np.ndarray, optional): Parameters [theta, delta].

        Returns:
            float or np.ndarray: Conditional CDF values.
        """
        return self.partial_derivative_C_wrt_v(u, v, param)

    def conditional_cdf_v_given_u(self, u, v, param=None):
        """
        Compute conditional CDF P(V ≤ v | U = u).

        Args:
            u (float or np.ndarray): U values.
            v (float or np.ndarray): V values.
            param (np.ndarray, optional): Parameters [theta, delta].

        Returns:
            float or np.ndarray: Conditional CDF values.
        """
        return self.partial_derivative_C_wrt_u(u, v, param)

    def IAD(self, data):
        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan
