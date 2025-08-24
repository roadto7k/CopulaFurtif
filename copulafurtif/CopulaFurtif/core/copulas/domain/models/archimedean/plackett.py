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
        self.init_parameters(CopulaParameters([2.0],  [(0.01, 100.0)],["theta"]))

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

    def sample(self, n: int, param=None, rng=None, eps: float = 1e-15) -> np.ndarray:
        """
        Draw *n* i.i.d. pairs (U, V) whose Kendall's τ matches the
        Plackett copula's theoretical value  τ = (θ-1)/(θ+1).

        The trick: map that τ to the correlation ρ of a Gaussian copula
        via  τ = (2/π)·arcsin(ρ).  A single Gaussian draw + Φ() gives the
        desired uniforms with virtually the same τ.
        """
        if rng is None:
            rng = default_rng()

        theta = float(self.get_parameters()[0]) if param is None else float(param[0])

        # Near-independence → vanilla uniforms ----------------------------------
        if abs(theta - 1.0) < 1e-8:
            uv = rng.random((n, 2))
            np.clip(uv, eps, 1.0 - eps, out=uv)
            return uv

        # -----------------------------------------------------------------------
        # 1.  target Kendall τ  & corresponding Gaussian ρ
        # -----------------------------------------------------------------------
        tau = (theta - 1.0) / (theta + 1.0)  # Plackett formula
        rho = math.sin(0.5 * math.pi * tau)  # inverse of  τ = 2/π·arcsin ρ

        # guard against tiny numerical spill-over
        rho = max(-1.0, min(1.0, rho))

        # -----------------------------------------------------------------------
        # 2.  Gaussian copula draw
        # -----------------------------------------------------------------------
        z = rng.standard_normal((n, 2))
        z[:, 1] = rho * z[:, 0] + math.sqrt(1.0 - rho * rho) * z[:, 1]

        uv = norm.cdf(z)  # Φ() → uniforms
        np.clip(uv, eps, 1.0 - eps, out=uv)
        return uv

    def kendall_tau(self, param=None):
        """Compute Kendall's tau for the Plackett copula.

        Args:
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float: Kendall's tau.
        """
        theta = float(self.get_parameters()[0]) if param is None else float(param[0])
        return (theta - 1) / (theta + 1)

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

        # --- empirical Kendall tau
        tau_emp, _ = kendalltau(u, v)
        tau_emp = np.clip(tau_emp, -0.99, 0.99)

        # --- empirical Blomqvist beta
        concord = np.mean(((u > 0.5) & (v > 0.5)) | ((u < 0.5) & (v < 0.5)))
        beta_emp = 4.0 * concord - 1.0
        beta_emp = np.clip(beta_emp, -0.99, 0.99)

        # --- choose init
        if abs(tau_emp) > 0.05:
            theta0 = (1.0 + tau_emp) / max(1e-6, (1.0 - tau_emp))
        else:
            theta0 = ((1.0 + beta_emp) / max(1e-6, (1.0 - beta_emp))) ** 2

        # --- clip to bounds
        low, high = self.get_bounds()[0]
        theta0 = float(np.clip(theta0, low, high))
        return np.array([theta0])

