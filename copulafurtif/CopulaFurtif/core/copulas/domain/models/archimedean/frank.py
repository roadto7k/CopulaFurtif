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
from scipy.special import spence
from scipy.stats import uniform
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
        if np.isclose(theta, 0.0):
            return np.ones_like(u)
        e_theta_u = np.exp(-theta * u)
        e_theta_v = np.exp(-theta * v)
        num = theta * e_theta_u * e_theta_v * (1 - np.exp(-theta))
        denom = (1 - np.exp(-theta) + (e_theta_u - 1) * (e_theta_v - 1)) ** 2
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
