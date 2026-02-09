import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel, CopulaParameters
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


class BB4Copula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """
    BB4 Copula (Two-parameter Archimedean copula).

    Attributes:
        name (str): Human-readable name of the copula.
        type (str): Identifier for the copula family.
        bounds_param (list of tuple): Bounds for copula parameters [mu, delta].
        parameters (np.ndarray): Current copula parameters.
        default_optim_method (str): Optimization method.
    """

    def __init__(self):
        """Initialize BB4 copula with default parameters mu=1.0, delta=1.0."""
        super().__init__()
        self.name = "BB4 Copula"
        self.type = "bb4"
        self.default_optim_method = "Powell"
        self.init_parameters(CopulaParameters(np.array([1.0, 1.0]), [(1e-6, np.inf), (1e-6, np.inf)], ["theta", "delta"]))

    @staticmethod
    def _logsumexp(a, b):
        m = np.maximum(a, b)
        return m + np.log1p(np.exp(-np.abs(a - b)))


    def get_cdf(self, u, v, param=None):
        """
        Evaluate the copula cumulative distribution function at (u, v).

        Args:
            u (float or array-like): First uniform margin in (0,1).
            v (float or array-like): Second uniform margin in (0,1).
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float or np.ndarray: CDF value C(u, v).
        """

        if param is None:
            param = self.get_parameters()
        theta, delta = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        x = u ** (-theta)
        y = v ** (-theta)
        T = (x - 1) ** (-delta) + (y - 1) ** (-delta)
        A = T ** (-1.0 / delta)
        z = x + y - 1.0 - A
        return np.round(z ** (-1.0 / theta), 14)

    def get_pdf(self, u, v, param=None):
        """
        Evaluate the copula probability density function at (u, v).

        Args:
            u (float or array-like): First uniform margin in (0,1).
            v (float or array-like): Second uniform margin in (0,1).
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float or np.ndarray: PDF value c(u, v).
        """

        if param is None:
            param = self.get_parameters()
        theta, delta = param

        # avoid 0/1
        eps = 1e-14
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        a = u ** (-theta) - 1.0  # (u^{-θ}-1)
        b = v ** (-theta) - 1.0
        S = a ** delta + b ** delta
        Z = 1.0 + S ** (1.0 / delta)

        return (theta * delta *
                Z ** (-1.0 / theta - 2.0) *
                S ** (1.0 / delta - 1.0) *
                u ** (-theta - 1.0) * v ** (-theta - 1.0) *
                a ** (delta - 1.0) * b ** (delta - 1.0))

    # def kendall_tau(self, param=None, m=256):
    #     """
    #      Kendall τ with 2‑D Gauss‑Legendre quadrature (default m=64 nodes).
    #
    #     Args:
    #         param (Sequence[float], optional): Copula parameters (mu, delta). Defaults to self.get_parameters().
    #         n (int, optional): Number of grid points per margin. Defaults to 201.
    #
    #     Returns:
    #         float: Theoretical Kendall’s tau
    #     """
    #
    #     if param is None:
    #         param = self.get_parameters()
    #
    #         # 1‑D Gauss‑Legendre on [0,1]
    #     k, w = leggauss(m)
    #     u = 0.5 * (k + 1.0)
    #     w = 0.5 * w
    #
    #     U, V = np.meshgrid(u, u, indexing="ij")
    #     Cuv = self.get_cdf(U, V, param)
    #
    #     integral = np.sum(w[:, None] * w[None, :] * Cuv)
    #     return 4.0 * integral - 1.0
    #
    # def sample(self, n, param=None):
    #     """
    #     Generate random samples from the copula using conditional inversion.
    #
    #     Args:
    #         n (int): Number of samples to generate.
    #         param (Sequence[float], optional): Copula parameters (mu, delta). Defaults to self.get_parameters().
    #
    #     Returns:
    #         np.ndarray: Array of shape (n, 2) with uniform samples on [0,1]².
    #     """
    #
    #     if param is None:
    #         param = self.get_parameters()
    #
    #     rng = np.random.default_rng()
    #     out = np.empty((n, 2))
    #     for i in range(n):
    #         u = rng.random()
    #         #   C(u,1)=u  ⇒   w ∈ [0, u]
    #         w = rng.random() * u
    #         root = brentq(lambda vv: self.get_cdf(u, vv, param) - w,
    #                       1e-12, 1 - 1e-12)
    #         out[i] = (u, root)
    #     return out

    def kendall_tau(self, param=None):
        """
        Placeholder for Kendall's tau.
        Currently unimplemented; returns NaN.
        """
        # If you ever do need the true formula, it lives in textbooks:
        # τ = 1 - 2 * Beta(1 + 1/θ, 1 + 1/δ)
        # from scipy.special import beta
        # θ, δ = self.get_parameters() if param is None else param
        # return 1 - 2 * beta(1 + 1/θ, 1 + 1/δ)
        return np.nan

    def sample(self, n, random_state=None):
        """
        Placeholder sampler.
        Not implemented for BB4Copula.
        """
        raise NotImplementedError("Sampling not implemented for BB4Copula")

    def LTDC(self, param=None):
        """
        Compute the lower tail dependence coefficient (LTDC) of the copula.

        Args:
            param (Sequence[float], optional): Copula parameters (mu, delta). Defaults to self.get_parameters().

        Returns:
            float: LTDC value (0.0 for this copula).
        """

        if param is None:
            param = self.get_parameters()
        theta, delta = param
        return (2.0 - 2.0 ** (-1.0 / delta)) ** (-1.0 / theta)

    def UTDC(self, param=None):
        """
        Compute the upper tail dependence coefficient (UTDC) of the copula.

        Args:
            param (Sequence[float], optional): Copula parameters (mu, delta). Defaults to self.get_parameters().

        Returns:
            float: UTDC value (2^(1/δ)).
        """

        if param is None:
            param = self.get_parameters()
        delta = param[1]
        return 2.0 ** (-1.0 / delta)

    @staticmethod
    def _log1pexp(t):
        return t if t > 36 else math.log1p(math.exp(t))

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """
        Compute the partial derivative ∂C(u,v)/∂u of the copula CDF.

        Args:
            u (float or array-like): First margin in (0,1).
            v (float or array-like): Second margin in (0,1).
            param (Sequence[float], optional): Copula parameters (mu, delta). Defaults to self.get_parameters().

        Returns:
            float or np.ndarray: Value of ∂C/∂u at (u, v).
        """

        if param is None:
            param = self.get_parameters()
        theta, delta = param

        eps = 1e-14
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        # core variables
        x = u ** (-theta)
        y = v ** (-theta)
        a = x - 1.0
        b = y - 1.0

        log_a = np.log(a)
        log_b = np.log(b)

        # S = a^{-δ} + b^{-δ}  in log‑space
        log_S = self._logsumexp(-delta * log_a, -delta * log_b)
        S = np.exp(log_S)

        # T = S^{-1/δ}
        log_T = -log_S / delta
        T = np.exp(log_T)

        Z = x + y - 1.0 - T

        # derivatives
        dxdu = -theta * u ** (-theta - 1.0)
        # log|dT/du|
        log_dT = (-delta - 1) * log_a + (-1 / delta - 1) * log_S + np.log(np.abs(dxdu))
        dTdu = np.sign(dxdu) * np.exp(log_dT)

        dZdu = dxdu - dTdu
        return (-1.0 / theta) * Z ** (-1.0 / theta - 1.0) * dZdu

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """
        Compute the partial derivative ∂C(u,v)/∂v of the copula CDF.

        Args:
            u (float or array-like): First margin in (0,1).
            v (float or array-like): Second margin in (0,1).
            param (Sequence[float], optional): Copula parameters (mu, delta). Defaults to self.get_parameters().

        Returns:
            float or np.ndarray: Value of ∂C/∂v at (u, v).
        """

        return self.partial_derivative_C_wrt_u(v, u, param)

    def IAD(self, data):
        """
        Return NaN for the Integrated Anderson-Darling (IAD) statistic.

        Args:
            data (Sequence[array-like, array-like]): Ignored pseudo-observations.

        Returns:
            float: Always returns np.nan.
        """

        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        """
        Return NaN for the Anderson-Darling (AD) statistic.

        Args:
            data (Sequence[array-like, array-like]): Ignored pseudo-observations.

        Returns:
            float: Always returns np.nan.
        """

        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan
