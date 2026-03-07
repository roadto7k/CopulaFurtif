"""
Frank Copula implementation.

The Frank copula is an Archimedean copula that supports both positive and negative
dependence. It is symmetric and does not exhibit tail dependence. This implementation
includes methods for CDF, PDF, sampling, and conditional distributions.

Attributes:
    name (str): Human-readable name of the copula.
    type (str): Copula identifier.
    bounds_param (list of tuple): Bounds for the copula parameter [theta] ∈ (-35, 35).
    parameters (np.ndarray): Copula parameter [theta].
    default_optim_method (str): Optimization method used for parameter fitting.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import root_scalar
from scipy.special import spence
from scipy.stats import uniform, kendalltau
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel, CopulaParameters
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence
from scipy.integrate import quad
from numpy.random import default_rng


class FrankCopula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """Frank Copula model."""

    def __init__(self):
        """Initialize the Frank copula with default parameters and bounds."""
        super().__init__()
        self.name = "Frank Copula"
        self.type = "frank"
        self.default_optim_method = "SLSQP"
        self.init_parameters(CopulaParameters(np.array([5.0]),  [(-35.0, 35.0)], ["theta"]))

    def get_cdf(self, u, v, param=None):
        """Compute the copula CDF C(u, v).

        Args:
            u (float or np.ndarray): First input in (0,1).
            v (float or np.ndarray): Second input in (0,1).
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float or np.ndarray: CDF values at (u, v).
        """
        if param is None:
            param = self.get_parameters()
        theta = float(param[0])

        # independence limit
        if abs(theta) < 1e-8:
            return np.asarray(u) * np.asarray(v)

        # optional safety (copula is defined on (0,1))
        eps = 1e-12
        u = np.clip(u, eps, 1.0 - eps)
        v = np.clip(v, eps, 1.0 - eps)

        A = np.expm1(-theta * u)  # exp(-θu) - 1
        B = np.expm1(-theta * v)  # exp(-θv) - 1
        D = np.expm1(-theta)  # exp(-θ) - 1

        return -(1.0 / theta) * np.log1p((A * B) / D)

    def get_pdf(self, u, v, param=None):
        """Compute the copula PDF c(u, v).

        Args:
            u (float or np.ndarray): First input in (0,1).
            v (float or np.ndarray): Second input in (0,1).
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float or np.ndarray: PDF values at (u, v).
        """
        if param is None:
            param = self.get_parameters()
        theta = param[0]

        # independence case theta→0
        if abs(theta) < 1e-12:
            return np.ones_like(u)

        # clip to avoid under/overflow
        eps = 1e-12
        u = np.clip(u, eps, 1.0 - eps)
        v = np.clip(v, eps, 1.0 - eps)

        exp_m_theta = np.exp(-theta)
        exp_m_theta_u = np.exp(-theta * u)
        exp_m_theta_v = np.exp(-theta * v)
        one_minus_exp = 1.0 - exp_m_theta

        # numerator = theta * (1 - e^{-theta}) * e^{-theta*(u+v)}
        num = theta * one_minus_exp * exp_m_theta_u * exp_m_theta_v

        # A = (1 - e^{-theta*u}) * (1 - e^{-theta*v})
        A = (1.0 - exp_m_theta_u) * (1.0 - exp_m_theta_v)
        # denominator = [1 - e^{-theta} - A]^2
        denom = (one_minus_exp - A) ** 2

        return num / denom

    def sample(self, n: int, seed: int | None = None, param=None, rng=None):
        """
        Sample (u, v) from the Frank copula using conditional inversion.

        Returns:
            ndarray of shape (n, 2)
        """
        if param is None:
            param = self.get_parameters()
        theta = float(param[0])

        if rng is None:
            rng = np.random.default_rng(seed)

        # Independence limit
        if abs(theta) < 1e-8:
            return rng.random((n, 2))

        # Draw U and an auxiliary W ~ U(0,1) for inversion
        u = rng.random(n)
        w = rng.random(n)

        # Stable building blocks
        # a = exp(-θu), d = exp(-θ), but use expm1 for stability where it matters
        a = np.exp(-theta * u)  # could use exp here; u in (0,1), theta bounded
        d = np.exp(-theta)  # same
        D = np.expm1(-theta)  # d - 1, stable even if theta small

        # Inversion formula (stable form):
        # v = -(1/θ) * log( 1 + D*w / (a - w*(a - 1)) )
        denom = a - w * (a - 1.0)  # always positive for w in (0,1), a > 0
        inside = 1.0 + (D * w) / denom

        v = -(1.0 / theta) * np.log(inside)

        # Safety clip to (0,1) in case of tiny numeric drift
        eps = 1e-12
        u = np.clip(u, eps, 1.0 - eps)
        v = np.clip(v, eps, 1.0 - eps)

        return np.column_stack([u, v])

    @staticmethod
    def debye1(theta, *, epsabs=1e-12, epsrel=1e-12):
        """
        Compute the Debye function of order 1:
            D1(θ) = (1/θ) ∫₀^θ t / (eᵗ − 1) dt

        Uses a Maclaurin series for |θ|<1e-4, and adaptive quadrature otherwise.
        """
        # Maclaurin series for small |θ|
        if abs(theta) < 1e-4:
            t = theta
            t2 = t * t
            return 1 - t / 4 + t2 / 36 - t2 * t2 / 3600 + t2 * t2 * t2 / 211_680

        # Adaptive quadrature for everything else
        integrand = lambda t: t / np.expm1(t)  # t / (eᵗ − 1)
        val, _ = quad(integrand, 0.0, theta,
                      limit=200, epsabs=epsabs, epsrel=epsrel)
        return val / theta

    def kendall_tau(self, param=None):
        """Compute Kendall's tau for the Frank copula.

        Args:
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float: Kendall's tau value.
        """
        if param is None:
            param = self.get_parameters()
        theta = param[0]

        if abs(theta) < 1e-8:
            t = theta
            t2 = t * t
            return t / 9 - t2 * t / 900 + t2 * t2 / 52920  # θ/9 − θ³/900 + θ⁵/52920

        D1 = self.debye1(theta)
        return 1.0 + 4.0 * (D1 - 1.0) / theta

    def LTDC(self, param=None):
        """Lower tail dependence coefficient (always 0 for Frank copula).

        Args:
            param (np.ndarray, optional): Copula parameter.

        Returns:
            float: 0.0
        """
        return 0.0

    def UTDC(self, param=None):
        """Upper tail dependence coefficient (always 0 for Frank copula).

        Args:
            param (np.ndarray, optional): Copula parameter.

        Returns:
            float: 0.0
        """
        return 0.0

    def IAD(self, data):
        """Integrated Absolute Deviation (disabled for Frank copula).

        Args:
            data (array-like): Input data (unused).

        Returns:
            float: NaN
        """
        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        """Anderson–Darling test statistic (disabled for Frank copula).

        Args:
            data (array-like): Input data (unused).

        Returns:
            float: NaN
        """
        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """Compute ∂C(u,v)/∂u for the Frank copula."""

        if param is None:
            param = self.get_parameters()

        theta = float(param[0])

        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)

        # Frank -> independence as theta -> 0
        if np.abs(theta) < 1e-10:
            res = v
            return float(res) if np.ndim(res) == 0 else res

        a = np.expm1(-theta * u)  # exp(-theta*u) - 1
        b = np.expm1(-theta * v)  # exp(-theta*v) - 1
        c = np.expm1(-theta)  # exp(-theta) - 1
        eu = np.exp(-theta * u)

        num = eu * b
        denom = c + a * b

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            res = num / denom

        # Théoriquement, la dérivée conditionnelle doit rester dans [0,1]
        res = np.clip(res, 0.0, 1.0)

        return float(res) if np.ndim(res) == 0 else res

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

    def blomqvist_beta(self, param=None) -> float:
        """
        Blomqvist's beta for Frank (closed form):
            β(θ) = 4*C(1/2,1/2) - 1
                 = -(4/θ) * log( 2a/(1+a) ) - 1,  where a = exp(-θ/2)

        Limit: β(0) = 0 (independence).
        """
        if param is None:
            param = self.get_parameters()
        theta = float(param[0])

        # independence limit
        if abs(theta) < 1e-12:
            return 0.0

        a = np.exp(-0.5 * theta)  # exp(-θ/2)
        ratio = (2.0 * a) / (1.0 + a)  # 2a/(1+a) in (0,1) for theta>0
        beta = -(4.0 / theta) * np.log(ratio) - 1.0

        # clamp for tiny numerical noise
        if beta > 1.0:
            beta = 1.0
        elif beta < -1.0:
            beta = -1.0
        return float(beta)

    def init_from_data(self, u, v):
        """
        Robust initialization of Frank copula parameter θ from data.

        Strategy
        --------
        - Compute empirical Kendall's tau (τ̂).
        - Compute empirical Blomqvist's beta (β̂).
        - If |τ̂| < 0.05 (weak dependence, small sample) → use β̂.
        - Otherwise → use τ̂.
        - Invert the theoretical relation (τ(θ) or β(θ)) numerically
          using a root-finder (Brentq).
        - Clip θ within parameter bounds [−35, 35].

        Parameters
        ----------
        u : array-like
            Pseudo-observations in (0,1), first margin.
        v : array-like
            Pseudo-observations in (0,1), second margin.

        Returns
        -------
        float
            Initial guess for θ suitable as a starting point for MLE.
        """

        u, v = np.asarray(u), np.asarray(v)

        # --- 1) empirical Kendall tau ---
        tau_emp, _ = kendalltau(u, v)
        tau_emp = np.clip(tau_emp, -0.99, 0.99)

        # --- 2) empirical Blomqvist beta ---
        concord = np.mean(((u > 0.5) & (v > 0.5)) | ((u < 0.5) & (v < 0.5)))
        beta_emp = 2.0 * concord - 1.0
        beta_emp = np.clip(beta_emp, -0.99, 0.99)

        # --- 3) choose moment ---
        use_beta = abs(tau_emp) < 0.05
        target = beta_emp if use_beta else tau_emp

        # --- 4) equations for τ(θ) and β(θ) ---
        def tau_theta(theta):
            if abs(theta) < 1e-8:
                return 0.0
            D1 = self.debye1(theta)
            return 1.0 + 4.0 * (D1 - 1.0) / theta

        def beta_theta(theta):
            if abs(theta) < 1e-8:
                return 0.0
            num = 2.0 * np.exp(-theta / 2.0) - 2.0 * np.exp(-theta)
            den = 1.0 - np.exp(-theta)
            return (4.0 / theta) * np.log(num / den) - 1.0

        func = beta_theta if use_beta else tau_theta

        # --- 5) root-finding to invert moment ---
        theta0 = 0.0
        try:
            sol = root_scalar(lambda th: func(th) - target,
                              bracket=(-35.0, 35.0), method="brentq")
            if sol.converged:
                theta0 = sol.root
        except Exception:
            theta0 = 0.0  # fallback → independence

        # --- 6) clip to bounds ---
        low, high = self.get_bounds()[0]
        theta0 = float(np.clip(theta0, low, high))

        return np.array([theta0])
