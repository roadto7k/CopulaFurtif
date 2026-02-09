import numpy as np
from numpy import log as _np_log, exp as _np_exp

from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel, CopulaParameters
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence

_FLOAT_MIN = 1e-308  # min positive normal float64
_FLOAT_MAX_LOG = 709.0  # np.exp(709) ~ 8e307 < max float64
_FLOAT_MIN_LOG = -745.0  # np.exp(-745) ~ 5e-324 > 0


def _safe_log(x):
    """Natural log, inputs clipped away from 0."""
    return _np_log(np.clip(x, _FLOAT_MIN, None))


def _safe_exp(log_x):
    """Exp of log_x with clipping to keep inside float64 range."""
    return _np_exp(np.clip(log_x, _FLOAT_MIN_LOG, _FLOAT_MAX_LOG))


def _safe_pow(base, exponent):
    """Stable power via exp(exponent * log(base)). base must be > 0."""
    return _safe_exp(exponent * _safe_log(base))


class BB9Copula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """
    BB9 Copula (Crowder) Archimedean copula.

    Attributes:
        name (str): Name of the copula.
        type (str): Identifier for the copula family.
        bounds_param (list of tuple): Parameter bounds for optimization.
        parameters (np.ndarray): Copula parameters [theta, delta].
        default_optim_method (str): Optimization method.
    """

    def __init__(self):
        """Initialize the BB9 copula with default parameters."""
        super().__init__()
        self.name = "BB9 Copula"
        self.type = "bb9"
        self.default_optim_method = "Powell"
        self.init_parameters(CopulaParameters(np.array([2.0, 2.0]), [(1.0, np.inf), (1e-6, np.inf)], ["theta", "delta"]))


    def get_cdf(self, u, v, param=None):
        """Compute the BB9 copula cumulative distribution function.

        The BB9 (Crowder) copula is defined by
            x = 1/δ − log(u),
            y = 1/δ − log(v),
            W = x**θ + y**θ − (1/δ)**θ,
            A = W**(1/θ),
            C(u,v) = exp(−A + 1/δ).

        Numeric stability is ensured via safe‐log, safe‐exp, and safe‐pow.

        Args:
            u (float or array-like): First margin in (0, 1).
            v (float or array-like): Second margin in (0, 1).
            param (Sequence[float], optional): Copula parameters
                `[theta, delta]`. If `None`, uses current parameters.

        Returns:
            float or np.ndarray:
            The value of the BB9 copula CDF at points (u, v), guaranteed in [0, 1].
        """

        if param is None:
            param = self.get_parameters()
        theta, delta = param
        delta_inv = 1.0 / delta

        u = np.clip(u, 1e-12, 1 - 1e-12)
        v = np.clip(v, 1e-12, 1 - 1e-12)

        # x = δ⁻¹ - ln u,  y = δ⁻¹ - ln v
        x = delta_inv - _safe_log(u)
        y = delta_inv - _safe_log(v)

        # W = x^θ + y^θ - δ⁻θ   (>0)
        W = x ** theta + y ** theta - delta_inv ** theta
        A = _safe_pow(W, 1.0 / theta)  # W^{1/θ}

        return _safe_exp(-A + delta_inv)

    def get_pdf(self, u, v, param=None):
        """Compute the BB9 copula probability density function.

        Uses the analytic formula
            c(u,v) = C(u,v)
                     · W^(1/θ−2)
                     · (A + θ − 1)
                     · x^(θ−1) y^(θ−1)
                     · (u v)^(−1),
        where
            δ⁻¹ = 1/delta,
            x = δ⁻¹ − log(u),
            y = δ⁻¹ − log(v),
            W = x^θ + y^θ − (δ⁻¹)^θ,
            A = W^(1/θ),
            C(u,v) = exp(−A + δ⁻¹).

        All operations are performed in log‐space (via safe‐log, safe‐exp, safe‐pow)
        to maintain numerical stability for extreme parameter values.

        Args:
            u (float or array-like): First uniform margin in (0, 1).
            v (float or array-like): Second uniform margin in (0, 1).
            param (Sequence[float], optional): Copula parameters `[theta, delta]`.
                If None, uses current model parameters.

        Returns:
            float or np.ndarray: The joint density c(u,v), guaranteed non‐negative.
        """
        if param is None:
            param = self.get_parameters()
        theta, delta = param
        delta_inv = 1.0 / delta

        # Clip inputs to avoid log(0)
        u = np.clip(u, 1e-12, 1 - 1e-12)
        v = np.clip(v, 1e-12, 1 - 1e-12)

        # Core transforms
        x = delta_inv - _safe_log(u)
        y = delta_inv - _safe_log(v)
        W = x ** theta + y ** theta - delta_inv ** theta
        A = _safe_pow(W, 1.0 / theta)

        # Base copula C(u,v)
        C = _safe_exp(-A + delta_inv)

        # Assemble log‐pdf for stability
        log_pdf = (
                _safe_log(C)
                + (1.0 / theta - 2.0) * _safe_log(W)
                + _safe_log(A + theta - 1.0)
                + (theta - 1.0) * (_safe_log(x) + _safe_log(y))
                - _safe_log(u)
                - _safe_log(v)
        )
        return _safe_exp(log_pdf)

    def kendall_tau(self, param=None):
        """
        Placeholder for Kendall's tau.
        Currently unimplemented; returns NaN.
        """
        return np.nan

    def sample(self, n, random_state=None):
        """
        Placeholder sampler.
        Not implemented for BB9Copula.
        """
        raise NotImplementedError("Sampling not implemented for BB9Copula")

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """Compute the partial derivative ∂C(u,v)/∂u of the BB9 copula.

        Uses the closed‐form expression:
            ∂C/∂u = C(u,v) · W^(1/θ−1) · x^(θ−1) / u,
        where
            δ⁻¹ = 1/delta,
            x = δ⁻¹ − log(u),
            y = δ⁻¹ − log(v),
            W = x^θ + y^θ − (δ⁻¹)^θ,
            C(u,v) = exp(−W^(1/θ) + δ⁻¹).

        All intermediate operations are done via safe‐log, safe‐exp, and safe‐pow
        to maintain numerical stability.

        Args:
            u (float or array-like): First uniform margin in (0, 1).
            v (float or array-like): Second uniform margin in (0, 1).
            param (Sequence[float], optional): Copula parameters `[theta, delta]`.
                If `None`, uses current model parameters.

        Returns:
            float or np.ndarray:
            The value of ∂C/∂u at (u, v).
        """

        if param is None:
            param = self.get_parameters()
        theta, delta = param
        delta_inv = 1.0 / delta

        # clip to avoid log(0) or log(>1)
        u = np.clip(u, 1e-12, 1 - 1e-12)
        v = np.clip(v, 1e-12, 1 - 1e-12)

        # compute transforms
        x = delta_inv - _safe_log(u)
        y = delta_inv - _safe_log(v)
        W = x ** theta + y ** theta - delta_inv ** theta

        # base copula value
        C = _safe_exp(-_safe_pow(W, 1.0 / theta) + delta_inv)

        # analytic derivative
        return C * _safe_pow(W, 1.0 / theta - 1.0) * x ** (theta - 1.0) / u

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """
        Approximate the partial derivative ∂C(u,v)/∂v for the BB9 copula.

        Args:
            u (float or array-like): First uniform margin in (0,1).
            v (float or array-like): Second uniform margin in (0,1).
            param (Sequence[float], optional): Copula parameters (θ, δ). Defaults to self.get_parameters().

        Returns:
            float or numpy.ndarray: Approximate ∂C/∂v at (u, v).
        """

        return self.partial_derivative_C_wrt_u(v, u, param)


    def LTDC(self, param=None):
        """
        Compute the lower tail dependence coefficient (LTDC) for the BB9 copula.

        Args:
            param (Sequence[float], optional): Copula parameters (θ, δ). Defaults to self.get_parameters().

        Returns:
            float: LTDC value.
        """

        return 0.0

    def UTDC(self, param=None):
        """
        Compute the upper tail dependence coefficient (UTDC) for the BB9 copula.

        Args:
            param (Sequence[float], optional): Copula parameters (θ, δ). Defaults to self.get_parameters().

        Returns:
            float: UTDC value = 2 − lim₍u→1₎ [1 − 2u + C(u,u)]/(1−u).
        """

        return 0.0


    def IAD(self, data):
        """
        Return NaN for the Integrated Anderson-Darling (IAD) statistic for BB9.

        Args:
            data (Sequence[array-like, array-like]): Ignored pseudo-observations.

        Returns:
            float: Always returns numpy.nan.
        """

        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        """
        Return NaN for the Anderson-Darling (AD) statistic for BB9.

        Args:
            data (Sequence[array-like, array-like]): Ignored pseudo-observations.

        Returns:
            float: Always returns numpy.nan.
        """

        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan
