"""
Plackett Copula implementation.

The Plackett copula is a symmetric copula that allows for modeling both positive 
and negative dependence while not exhibiting tail dependence. It is particularly 
useful due to its simplicity and flexibility with a single parameter.

Attributes:
    name (str): Human-readable name of the copula.
    type (str): Identifier for the copula type.
    bounds_param (list of tuple): Bounds for the copula parameter [theta] ∈ (0.01, 100.0).
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
        self.init_parameters(CopulaParameters(np.array([2.0]),  [(0.01, 100.0)],["theta"]))

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
        Robust initialization of Plackett copula parameter theta from data.

        Strategy
        --------
        - Compute empirical Kendall's tau (tau_emp).
        - Compute empirical Blomqvist's beta (beta_emp).
        - If |tau_emp| > 0.05, invert tau -> theta = (1+tau)/(1-tau).
        - Else, invert beta -> theta = ((1+beta)/(1-beta))**2.
        - Clip theta to parameter bounds.

        Parameters
        ----------
        u, v : array-like
            Pseudo-observations in (0,1).

        Returns
        -------
        float
            Initial guess theta0 for MLE fitting.
        """

        u, v = np.asarray(u), np.asarray(v)
        # Empirical beta
        concord = np.mean(((u > 0.5) & (v > 0.5)) | ((u < 0.5) & (v < 0.5)))
        beta_emp = float(np.clip(2.0 * concord - 1.0, -0.99, 0.99))

        # beta -> theta (fermé)
        theta0 = ((1.0 + beta_emp) / max(1e-6, (1.0 - beta_emp))) ** 2

        # Optional refine with τ numeric (safe bracketing)
        tau_emp = float(np.clip(kendalltau(u, v)[0], -0.999, 0.999))
        low, high = self.get_bounds()[0]

        def tau_of(theta):
            return self.kendall_tau([theta], m=200)

        try:
            if tau_emp > 1e-6:
                a = max(1.0 + 1e-6, low)
                b = high
                fa = tau_of(a) - tau_emp
                fb = tau_of(b) - tau_emp
                if fa * fb < 0:
                    theta0 = brentq(lambda t: tau_of(t) - tau_emp, a, b, maxiter=60)
            elif tau_emp < -1e-6:
                a = low
                b = min(1.0 - 1e-6, high)
                fa = tau_of(a) - tau_emp
                fb = tau_of(b) - tau_emp
                if fa * fb < 0:
                    theta0 = brentq(lambda t: tau_of(t) - tau_emp, a, b, maxiter=60)
            # sinon tau ~ 0 : on garde l'init via beta (propre près de l'indépendance)
        except Exception:
            # si la bracketing rate (dataset atypique), garder l'init beta
            pass

        theta0 = float(np.clip(theta0, low, high))
        return np.array([theta0])

