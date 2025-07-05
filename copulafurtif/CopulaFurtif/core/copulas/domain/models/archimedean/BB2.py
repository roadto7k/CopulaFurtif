import math
from typing import Union

import numpy as np
from numpy.random import default_rng
from scipy import integrate

from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel, CopulaParameters
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence
from CopulaFurtif.core.copulas.domain.models.archimedean.BB1 import BB1Copula

_LOG_MAX = 700.0          # safe upper bound for exp() on 64-bit floats


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
        self.init_parameters(CopulaParameters([2, 1.5],[(0.05, 10.0), (1.0, 5.0)], ["theta", "delta"] ))

    # ------------------------------------------------------------------ utils
    # ----- Archimedean generator and safe inverse -----
    @staticmethod
    def _phi(s: Union[np.ndarray, float], theta: float, delta: float) -> Union[np.ndarray, float]:
        return (1.0 + (1.0 / delta) * np.log1p(s)) ** (-1.0 / theta)

    @staticmethod
    def _log1p_sum(a, b):
        """
        Compute log( exp(a) + exp(b) - 1 ) safely.

        a = log(1+x), b = log(1+y)  with  x,y >= 0.
        Handles large or very different magnitudes without loss of precision.
        """
        m = np.maximum(a, b)
        inside = np.exp(a - m) + np.exp(b - m) - np.exp(-m)
        return m + np.log(inside)

    @staticmethod
    def _phi_inv(
            t: Union[np.ndarray, float], theta: float, delta: float,
            *, return_log: bool = False, _LM: float = _LOG_MAX
    ):
        z = delta * (np.power(t, -theta) - 1.0)
        big = z > _LM
        x = np.empty_like(z, dtype=float)
        log1p = np.empty_like(z, dtype=float)
        x[~big] = np.expm1(z[~big])
        log1p[~big] = np.log1p(x[~big])
        x[big] = np.inf
        log1p[big] = z[big]
        return (x, log1p) if return_log else x

    # ----- log-derivative of φ -----
    @staticmethod
    def _log_phi_prime(log1ps: np.ndarray, theta: float, delta: float) -> np.ndarray:
        A_log = np.log1p(log1ps / delta)
        return -(1.0 / theta + 1.0) * A_log - log1ps - math.log(theta * delta)

    # ----- second derivative for PDF -----
    @staticmethod
    def _phi_double_prime(s: Union[np.ndarray, float], theta: float, delta: float):
        A = 1.0 + (1.0 / delta) * np.log1p(s)
        t1 = 1.0 / (theta * delta * (1 + s) ** 2) * A ** (-1.0 / theta - 1.0)
        p = 1.0 / theta + 1.0
        t2 = p / (theta * delta ** 2 * (1 + s) ** 2) * A ** (-1.0 / theta - 2.0)
        return t1 + t2

    def _invert_conditional_v(
            self,
            u: np.ndarray,
            p: np.ndarray,
            theta: float,
            delta: float,
            *,
            eps: float = 1e-12,
            max_iter: int = 20,
            tol: float = 1e-12,
    ) -> np.ndarray:
        """
        Solve V | U=u  by Newton–Raphson on  f(y)=∂C/∂u(u,v)−p = 0,
        with the Joe & Hu (1996) closed-form *seed*:

            y₀ = (1 + x) · ( p^(−θ/(θ+1)) − 1 ),
            where x = φ⁻¹(u).

        Parameters
        ----------
        u, p : float or ndarray
            Same shape.  u in (0,1), p in (0,1).
        theta, delta : float
            BB2 parameters.
        eps : float
            Numerical lower/upper bound for probabilities.
        max_iter : int
            Newton iterations.
        tol : float
            |Δy| stop criterion (absolute).

        Returns
        -------
        v : ndarray
            Same shape as u – the solution in (0,1).
        """
        # 1) clip inputs
        u = np.asarray(u, float)
        p = np.asarray(p, float)
        uc = np.clip(u, eps, 1.0 - eps)
        pc = np.clip(p, eps, 1.0 - eps)

        # 2) compute x = φ⁻¹(u) and log1p_x
        x, log1p_x = self._phi_inv(uc, theta, delta, return_log=True)

        # 3) Joe & Hu seed: y0 = (1 + x) * (p^(−θ/(θ+1)) - 1)
        safe_max = 1e100
        expo = -theta / (theta + 1.0)
        y0 = (1.0 + x) * (pc ** expo - 1.0)
        y = np.minimum(np.maximum(y0, 0.0), safe_max)

        # 4) Newton–Raphson iterations
        bad = np.zeros_like(y, dtype=bool)
        for _ in range(max_iter):
            # compute f = φ′(s) - p·φ′(x) and f' = φ''(s)
            log1p_y = np.log1p(y)
            log1p_s = self._log1p_sum(log1p_x, log1p_y)
            log_phi_s = self._log_phi_prime(log1p_s, theta, delta)
            log_phi_x = self._log_phi_prime(log1p_x, theta, delta)
            phi_s = np.exp(log_phi_s)
            phi_x = np.exp(log_phi_x)
            s = x + y
            phi2_s = self._phi_double_prime(s, theta, delta)

            delta_y = (phi_s - pc * phi_x) / phi2_s
            # detect divergence
            bad_iter = ~np.isfinite(delta_y)
            if bad_iter.any():
                bad |= bad_iter
                break

            y_new = np.maximum(y - delta_y, 0.0)
            if np.all(np.abs(delta_y) < tol):
                y = y_new
                break
            y = y_new
        else:
            # non-convergence but no NaN: mark none as bad
            bad |= False

        # 5) bissection fallback pour les cas divergents
        if bad.any():
            # vector bisection on the subset
            lo = np.full_like(y[bad], eps)
            hi = np.full_like(y[bad], 1.0 - eps)
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                val = self.partial_derivative_C_wrt_u(uc[bad], self._phi(mid, theta, delta),
                                                      [theta, delta])
                lo = np.where(val < pc[bad], mid, lo)
                hi = np.where(val >= pc[bad], mid, hi)
            y[bad] = 0.5 * (lo + hi)

        # 6) convert back to v and clip
        v = self._phi(y, theta, delta)
        return np.clip(v, eps, 1.0 - eps)

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
        theta, delta = param or self.get_parameters()
        eps = 1e-12
        uc = np.clip(u, eps, 1 - eps)
        vc = np.clip(v, eps, 1 - eps)
        x = self._phi_inv(uc, theta, delta)
        y = self._phi_inv(vc, theta, delta)
        return self._phi(x + y, theta, delta)

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
        theta, delta = param or self.get_parameters()
        eps = 1e-12
        uc = np.clip(u, eps, 1 - eps)
        vc = np.clip(v, eps, 1 - eps)
        x = self._phi_inv(uc, theta, delta)
        y = self._phi_inv(vc, theta, delta)
        s = x + y
        # dx/du and dy/dv from inverse generator
        dx_du = -theta * delta * uc ** (-theta - 1) * (x + 1.0)
        dy_dv = -theta * delta * vc ** (-theta - 1) * (y + 1.0)
        pdf = self._phi_double_prime(s, theta, delta) * dx_du * dy_dv
        return np.nan_to_num(pdf, nan=0.0, neginf=0.0, posinf=np.inf)  # guard tiny negative round‑offs

    def kendall_tau(self, param=None) -> float:
        """
        Kendall’s tau for BB2 is the same as for its survival‐inverse BB1.
        """
        theta, delta = param or self.get_parameters()
        base = BB1Copula()
        base.set_parameters([theta, delta])
        return base.kendall_tau()

    def sample(
            self,
            n: int,
            param=None,
            rng=None,
    ) -> np.ndarray:
        """
        Exact BB2 sample via 180° rotation of BB1.
        """
        theta, delta = param or self.get_parameters()
        if rng is None:
            rng = default_rng()

        from CopulaFurtif.core.copulas.domain.models.archimedean.BB1 import BB1Copula
        base = BB1Copula()
        base.set_parameters([theta, delta])
        uv1 = base.sample(n, [theta, delta], rng)  # exact BB1

        U2 = 1.0 - uv1[:, 0]
        V2 = 1.0 - uv1[:, 1]
        return np.column_stack((U2, V2))

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

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """Compute the partial derivative ∂C_BB2/∂u at (u, v).

        This uses the 180°-rotated relation
        ∂C_BB2/∂u(u,v) = 1 − ∂C_BB1/∂u(1−u, 1−v).

        Args:
            u (float or np.ndarray): U-coordinates in [0, 1].
            v (float or np.ndarray): V-coordinates in [0, 1].
            param (Sequence[float], optional):
                [theta, delta] copula parameters. If None, uses
                the model’s current parameters.

        Returns:
            float or np.ndarray: The value(s) of ∂C/∂u at the given points.
        """

        theta, delta = param or self.get_parameters()
        eps = 1e-12

        # 1) to array, clip once
        u_arr = np.asarray(u, float)
        v_arr = np.asarray(v, float)
        uc = np.clip(u_arr, eps, 1 - eps)
        vc = np.clip(v_arr, eps, 1 - eps)

        # 2) inverse generator & log(1+x)
        x, log1p_x = self._phi_inv(uc, theta, delta, return_log=True)
        y, log1p_y = self._phi_inv(vc, theta, delta, return_log=True)
        log1p_s = self._log1p_sum(log1p_x, log1p_y)

        # 3) log-space ratio φ′(s)/φ′(x)
        log_phi_s = self._log_phi_prime(log1p_s, theta, delta)
        log_phi_x = self._log_phi_prime(log1p_x, theta, delta)
        deriv = np.exp(log_phi_s - log_phi_x)

        # 4) exact boundary limits
        deriv = np.where(u_arr <= eps, 1.0, deriv)
        deriv = np.where(u_arr >= 1 - eps, 0.0, deriv)
        overflow = np.logical_or(np.isinf(x), np.isinf(y))
        deriv = np.where(overflow, 0.0, deriv)
        deriv = np.nan_to_num(deriv, nan=0.0, posinf=0.0, neginf=0.0)

        return float(deriv) if deriv.shape == () else deriv



    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """Compute the partial derivative ∂C_BB2/∂v at (u, v).

        By exchangeability of the Archimedean base copula, we can
        reuse ∂C/∂u by swapping arguments:
        ∂C_BB2/∂v(u,v) = ∂C_BB2/∂u(v,u).

        Args:
            u (float or np.ndarray): U-coordinates in [0, 1].
            v (float or np.ndarray): V-coordinates in [0, 1].
            param (Sequence[float], optional):
                [theta, delta] copula parameters. If None, uses
                the model’s current parameters.

        Returns:
            float or np.ndarray: The value(s) of ∂C/∂v at the given points.
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
