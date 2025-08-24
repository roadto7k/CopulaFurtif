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
        self.init_parameters(CopulaParameters([5.0],  [(-35.0, 35.0)], ["theta"]))

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
        theta = param[0]
        if np.isclose(theta, 0.0):
            return u * v
        num = (np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1)
        denom = np.exp(-theta) - 1
        return -1 / theta * np.log(1 + num / denom)

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

    def sample(self, n, param=None, rng=None):
        """Generate samples from the Frank copula.

        Args:
            n (int): Number of samples to generate.
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            np.ndarray: Array of shape (n, 2) of pseudo-observations.
        """
        if param is None:
            theta = self.get_parameters()[0]
        else:
            theta = float(param[0])

        if rng is None:
            rng = default_rng()

            # --- independence limit ------------------------------------------
        if abs(theta) < 1e-8:
            return rng.random((n, 2))

            # --- core algorithm ----------------------------------------------
        u = rng.random(n)
        w = rng.random(n)

        D = np.exp(-theta) - 1.0  # common denominator   D = e^{-θ} − 1
        A = np.exp(-theta * u) - 1.0  # A = e^{-θu} − 1

        B = (w * D) / (A + 1.0 - w * A)  # inversion term  B = w D /(A+1−w A)
        v = -1.0 / theta * np.log1p(B)  # v = −(1/θ)·ln(1+B)

        return np.column_stack((u, v))

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
        """Compute ∂C(u,v)/∂u.

        Args:
            u (float or np.ndarray): First input in (0,1).
            v (float or np.ndarray): Second input in (0,1).
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float or np.ndarray: Partial derivative values.
        """
        if param is None:
            param = self.get_parameters()

        theta = param[0]

        e_theta_u = np.exp(-theta * u)
        e_theta_v = np.exp(-theta * v)

        num = e_theta_u * (e_theta_v - 1)
        denom = np.exp(-theta) - 1 + (e_theta_u - 1) * (e_theta_v - 1)
        return num / denom

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
        Compute Blomqvist's beta (theoretical) for the Frank copula.

        Notes
        -----
        Defined as:
            β(θ) = (4/θ) * log( (2*exp(-θ/2) - 2*exp(-θ)) / (1 - exp(-θ)) ) - 1

        Reference
        ---------
        - Nelsen (2006), "An Introduction to Copulas", Springer.
        - Genest (1987).

        Parameters
        ----------
        param : np.ndarray, optional
            Copula parameter [theta]. If None, uses the current parameters.

        Returns
        -------
        float
            Theoretical Blomqvist's beta.
        """
        if param is None:
            param = self.get_parameters()
        theta = float(param[0])

        if abs(theta) < 1e-8:
            return 0.0  # independence limit

        num = 2.0 * np.exp(-theta / 2.0) - 2.0 * np.exp(-theta)
        den = 1.0 - np.exp(-theta)
        return (4.0 / theta) * np.log(num / den) - 1.0

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
        beta_emp = 4.0 * concord - 1.0
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
