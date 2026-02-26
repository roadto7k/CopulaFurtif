"""
Plackett Copula implementation.

The Plackett copula is a symmetric copula that allows for modeling both positive 
and negative dependence while not exhibiting tail dependence. It is particularly 
useful due to its simplicity and flexibility with a single parameter.

Attributes:
    name (str): Human-readable name of the copula.
    type (str): Identifier for the copula type.
    bounds_param (list of tuple): Bounds for the copula parameter [theta] ∈ (0.0, 100.0).
    parameters (np.ndarray): Copula parameter [theta].
    default_optim_method (str): Optimization method used during fitting.
"""
import math

import numpy as np
from scipy.optimize import brentq

from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel, CopulaParameters
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence
from numpy.random import default_rng
from scipy.stats import norm, kendalltau


class PlackettCopula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """Plackett Copula model."""

    def __init__(self):
        """Initialize the Plackett copula with default parameters and bounds."""
        super().__init__()
        self.name = "Plackett Copula"
        self.type = "plackett"
        self.default_optim_method = "SLSQP"
        self.init_parameters(CopulaParameters(np.array([2.0]),  [(0.0, 100.0)],["delta"]))

    def get_cdf(self, u, v, param=None):
        """Compute the copula CDF C(u, v).

        Args:
            u (float or np.ndarray): First input in (0, 1).
            v (float or np.ndarray): Second input in (0, 1).
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float or np.ndarray: CDF value(s).
        """
        theta = float(self.get_parameters()[0]) if param is None else float(param[0])
        if abs(theta - 1.0) < 1e-8:
            return np.asarray(u) * np.asarray(v)
        eps = 1e-12
        u = np.clip(u, eps, 1.0 - eps)
        v = np.clip(v, eps, 1.0 - eps)
        a = theta - 1
        b = 1 + a * (u + v)
        c = np.sqrt(b ** 2 - 4 * theta * a * u * v)
        return (2 * theta * u * v) / (b + c)

    def get_pdf(self, u, v, param=None):
        """Compute the copula PDF c(u, v).

        Args:
            u (float or np.ndarray): First input in (0, 1).
            v (float or np.ndarray): Second input in (0, 1).
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float or np.ndarray: PDF value(s).
        """
        theta = float(self.get_parameters()[0]) if param is None else float(param[0])
        if abs(theta - 1.0) < 1e-8:
            return np.ones_like(np.asarray(u), dtype=float)
        eps = 1e-12
        u = np.clip(u, eps, 1.0 - eps)
        v = np.clip(v, eps, 1.0 - eps)
        num = theta * (1 + (theta - 1) * (u + v - 2 * u * v))
        denom = ((1 + (theta - 1) * (u + v)) ** 2 - 4 * theta * (theta - 1) * u * v) ** 1.5
        return num / denom

    def sample(self, n: int, param=None, rng=None, eps: float = 1e-12) -> np.ndarray:
        if rng is None:
            rng = default_rng()
        theta = float(self.get_parameters()[0]) if param is None else float(param[0])

        # Indépendance
        if abs(theta - 1.0) < 1e-12:
            uv = rng.random((n, 2))
            np.clip(uv, eps, 1.0 - eps, out=uv)
            return uv

        u = rng.random(n)
        w = rng.random(n)  # target CDF values for V|U=u
        # Bisection on v in (0,1)
        v_lo = np.full(n, eps)
        v_hi = np.full(n, 1.0 - eps)

        for _ in range(40):  # ~1e-12 accuracy
            v_mid = 0.5 * (v_lo + v_hi)
            delta = theta - 1.0
            A = 1.0 + delta * (u + v_mid)
            sqrtD = np.sqrt(A * A - 4.0 * theta * delta * u * v_mid)
            F = 0.5 * (1.0 - (A - 2.0 * theta * v_mid) / sqrtD)  # ∂C/∂u(u,v)
            mask = F < w
            v_lo[mask] = v_mid[mask]
            v_hi[~mask] = v_mid[~mask]

        v = 0.5 * (v_lo + v_hi)
        uv = np.column_stack([u, v])
        np.clip(uv, eps, 1.0 - eps, out=uv)
        return uv

    def kendall_tau(self, param=None, m: int = 300):
        """
        Kendall's tau by numerical integration:
          tau = 4 * ∬ C(u,v) c(u,v) du dv - 1
        Deterministic grid (m x m), assez rapide & stable pour l'init/logging.
        """
        theta = float(self.get_parameters()[0]) if param is None else float(param[0])
        if abs(theta - 1.0) < 1e-12:
            return 0.0

        # grid in (0,1)^2
        u = (np.arange(m) + 0.5) / m
        U, V = np.meshgrid(u, u, indexing="ij")

        # CDF and PDF at [theta]
        C = self.get_cdf(U, V, [theta])
        num = theta * (1 + (theta - 1) * (U + V - 2 * U * V))
        denom = ((1 + (theta - 1) * (U + V)) ** 2 - 4 * theta * (theta - 1) * U * V) ** 1.5
        c = num / denom

        I = np.mean(C * c)
        return float(4.0 * I - 1.0)

    def LTDC(self, param=None):
        """Lower tail dependence coefficient (0 for Plackett copula).

        Args:
            param (np.ndarray, optional): Copula parameter.

        Returns:
            float: 0.0
        """
        return 0.0

    def UTDC(self, param=None):
        """Upper tail dependence coefficient (0 for Plackett copula).

        Args:
            param (np.ndarray, optional): Copula parameter.

        Returns:
            float: 0.0
        """
        return 0.0

    def IAD(self, data):
        """Integrated Absolute Deviation (disabled for Plackett copula).

        Args:
            data (array-like): Input data (unused).

        Returns:
            float: NaN.
        """
        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        """Anderson–Darling test statistic (disabled for Plackett copula).

        Args:
            data (array-like): Input data (unused).

        Returns:
            float: NaN.
        """
        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """Compute ∂C(u,v)/∂u.

        Args:
            u (float or np.ndarray): U values.
            v (float or np.ndarray): V values.
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float or np.ndarray: Partial derivative values.
        """
        theta = float(self.get_parameters()[0]) if param is None else float(param[0])

        if abs(theta - 1.0) < 1e-8:
            return np.asarray(v)

        eps = 1e-12
        u = np.clip(u, eps, 1.0 - eps)
        v = np.clip(v, eps, 1.0 - eps)

        delta = theta - 1.0
        A = 1.0 + delta * (u + v)
        sqrtD = np.sqrt(A ** 2 - 4.0 * theta * delta * u * v)

        # ½ [1 − (A − 2 θ v) / √D]
        return 0.5 * (1.0 - (A - 2.0 * theta * v) / sqrtD)

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """Compute ∂C(u,v)/∂v via symmetry.

        Args:
            u (float or np.ndarray): U values.
            v (float or np.ndarray): V values.
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float or np.ndarray: Partial derivative values.
        """
        return self.partial_derivative_C_wrt_u(v, u, param)

    def blomqvist_beta(self, param=None):
        """
        Compute Blomqvist's beta (theoretical) for the Plackett copula.

        Formula
        -------
        beta(theta) = (sqrt(theta) - 1) / (sqrt(theta) + 1)

        Inverse
        -------
        theta = ((1 + beta) / (1 - beta))**2

        Parameters
        ----------
        param : np.ndarray, optional
            Copula parameter [theta]. If None, uses current parameters.

        Returns
        -------
        float
            Theoretical Blomqvist's beta.
        """
        if param is None:
            param = self.get_parameters()
        theta = float(param[0])
        if theta <= 0:
            return 0.0
        return (np.sqrt(theta) - 1.0) / (np.sqrt(theta) + 1.0)

    def init_from_data(self, u, v):
        """
        Initialize theta from data.

        Priority: empirical Blomqvist beta (closed-form inversion for Plackett).
        Fallback: empirical Kendall tau + numerical inversion if needed.

        Plackett (theta=δ):
          - theta = 1 => independence
          - 0 < theta < 1 => negative dependence
          - theta > 1 => positive dependence

        Blomqvist beta for Plackett:
          beta = (sqrt(theta) - 1) / (sqrt(theta) + 1)
          theta = ((1 + beta) / (1 - beta))^2
        """
        import numpy as np
        from scipy.stats import kendalltau
        from scipy.optimize import brentq

        u = np.asarray(u, dtype=float).ravel()
        v = np.asarray(v, dtype=float).ravel()

        # Filter finite and keep inside (0,1) open interval
        mask = np.isfinite(u) & np.isfinite(v)
        u = u[mask]
        v = v[mask]
        if u.size < 20:
            return self.get_parameters()

        eps_uv = 1e-12
        u = np.clip(u, eps_uv, 1.0 - eps_uv)
        v = np.clip(v, eps_uv, 1.0 - eps_uv)

        low, high = self.get_bounds()[0]
        eps_th = 1e-6

        # ---------------------------------------------------------------------
        # 1) Preferred: Blomqvist beta (fast, robust, closed-form inversion)
        # beta_hat = 4*C_n(1/2,1/2) - 1
        # ---------------------------------------------------------------------
        c_hat = float(np.mean((u <= 0.5) & (v <= 0.5)))
        beta_emp = 4.0 * c_hat - 1.0

        if np.isfinite(beta_emp):
            # keep away from +/-1
            beta_emp = float(np.clip(beta_emp, -0.999999, 0.999999))

            # independence-ish => theta ~ 1
            if abs(beta_emp) < 1e-3:
                theta0 = float(np.clip(1.0, low + eps_th, high - eps_th))
                self.set_parameters([theta0])
                return self.get_parameters()

            # closed-form inversion
            theta0 = ((1.0 + beta_emp) / (1.0 - beta_emp)) ** 2
            theta0 = float(np.clip(theta0, low + eps_th, high - eps_th))

            # If inversion yields something sane, accept it
            try:
                self.set_parameters([theta0])
                return self.get_parameters()
            except Exception:
                # fall through to tau-based init
                pass

        # ---------------------------------------------------------------------
        # 2) Fallback: Kendall tau + numerical inversion (slower, but general)
        # ---------------------------------------------------------------------
        tau_emp, _ = kendalltau(u, v)
        if not np.isfinite(tau_emp):
            return self.get_parameters()

        # near independence
        if abs(tau_emp) < 1e-3:
            theta0 = float(np.clip(1.0, low + eps_th, high - eps_th))
            self.set_parameters([theta0])
            return self.get_parameters()

        def f(theta: float) -> float:
            return float(self.kendall_tau(param=[theta]) - tau_emp)

        try:
            if tau_emp > 0:
                a = max(1.0 + eps_th, low + eps_th)
                b = high - eps_th
            else:
                a = max(low + eps_th, eps_th)
                b = min(1.0 - eps_th, high - eps_th)

            if not (a < b):
                theta0 = float(np.clip(1.0, low + eps_th, high - eps_th))
                self.set_parameters([theta0])
                return self.get_parameters()

            fa = f(a)
            fb = f(b)

            if not (np.isfinite(fa) and np.isfinite(fb)) or fa * fb > 0:
                theta0 = float(np.clip(1.0 + (0.1 if tau_emp > 0 else -0.1),
                                       low + eps_th, high - eps_th))
                self.set_parameters([theta0])
                return self.get_parameters()

            theta0 = brentq(f, a, b, maxiter=200)
            theta0 = float(np.clip(theta0, low + eps_th, high - eps_th))
            self.set_parameters([theta0])
            return self.get_parameters()

        except Exception:
            theta0 = float(np.clip(1.0, low + eps_th, high - eps_th))
            self.set_parameters([theta0])
            return self.get_parameters()

