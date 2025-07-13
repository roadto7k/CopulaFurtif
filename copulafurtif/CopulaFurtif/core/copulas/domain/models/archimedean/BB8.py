import numpy as np
from numpy import log as _np_log, exp as _np_exp

from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel, CopulaParameters
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence

_FLOAT_MIN = 1e-308
_FLOAT_MAX_LOG = 709.0
_FLOAT_MIN_LOG = -745.0


def _safe_log(x):
    """Natural log with floor to avoid -inf."""
    return _np_log(np.clip(x, _FLOAT_MIN, None))


def _safe_exp(log_x):
    """Exp with clipping of exponent to float64 range."""
    return _np_exp(np.clip(log_x, _FLOAT_MIN_LOG, _FLOAT_MAX_LOG))


def _safe_pow(base, exponent):
    """Stable positive power via exp(exponent * log(base))."""
    return _safe_exp(exponent * _safe_log(base))


class BB8Copula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """
    BB8 Copula (Durante et al.):
      C(u,v) = [1 - (1-A)*(1-B)]^(1/theta),
      where A = [1 - (1-u)^theta]^delta,
            B = [1 - (1-v)^theta]^delta.

    Attributes:
        name (str): Human-readable name of the copula.
        type (str): Identifier for the copula family.
        bounds_param (list of tuple): Bounds for copula parameters.
        parameters (np.ndarray): Current copula parameters.
        default_optim_method (str): Default method for optimization.
    """

    def __init__(self):
        super().__init__()
        self.name = "BB8 Copula (Durante)"
        self.type = "bb8"
        self.default_optim_method = "Powell"
        self.init_parameters(CopulaParameters([2.0, 0.7], [(1.0, np.inf), (1e-6, 1.0)], ["theta", "delta"]))

    @staticmethod
    def _transform(u, theta):
        """Return X = 1-(1-u)^θ and log(1-u) (for reuse)."""
        log_one_minus_u = _safe_log(1.0 - u)
        pow_term = _safe_exp(theta * log_one_minus_u)  # (1-u)^θ
        X = 1.0 - pow_term
        return X, pow_term, log_one_minus_u

    def get_cdf(self, u, v, param=None):
        """
        Evaluate the BB8 copula cumulative distribution function at (u, v).

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

        # X = 1 - (1‑u)^θ   and   Y = 1 - (1‑v)^θ      (use log‑space for stability)
        log_one_minus_u = _safe_log(1.0 - u)
        log_one_minus_v = _safe_log(1.0 - v)
        X = 1.0 - _safe_exp(theta * log_one_minus_u)
        Y = 1.0 - _safe_exp(theta * log_one_minus_v)

        # A, B
        A = _safe_pow(X, delta)
        B = _safe_pow(Y, delta)

        # inner term ∈[0,1]
        inner = 1.0 - (1.0 - A) * (1.0 - B)
        inner = np.clip(inner, 0.0, 1.0)

        # final CDF
        return _safe_pow(inner, 1.0 / theta)


    def get_pdf(self, u, v, param=None):
        """
        Approximate the BB8 copula probability density function at (u, v) .

        Args:
            u (float or array-like): First uniform margin in (0,1).
            v (float or array-like): Second uniform margin in (0,1).
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float or np.ndarray: Approximate PDF c(u, v).
        """

        if param is None:
            param = self.get_parameters()
        theta, delta = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        X, _, log_one_minus_u = self._transform(u, theta)
        Y, _, log_one_minus_v = self._transform(v, theta)
        eta = 1.0 - (1.0 - delta) ** theta
        T = 1.0 - (X * Y) / eta
        T = np.clip(T, _FLOAT_MIN, 1.0)

        dX_du = theta * _safe_exp((theta - 1.0) * log_one_minus_u)
        dY_dv = theta * _safe_exp((theta - 1.0) * log_one_minus_v)

        # Components
        pref = (1.0 / (delta * eta)) * _safe_pow(T, 1.0 / theta - 2.0)
        first_term = dX_du * dY_dv * _safe_pow(T, 1.0)  # actually multiplier 1 but keep log‑safe
        second_term = -(1.0 / eta) * X * dY_dv * dX_du * (1.0 / theta - 1.0)

        pdf = pref * ((1.0 / theta) * first_term + second_term)
        return np.maximum(pdf, 0.0)

    def kendall_tau(self, param=None):
        """
        Placeholder for Kendall's tau.
        Currently unimplemented; returns NaN.
        """
        return np.nan

    def sample(self, n, random_state=None):
        """
        Placeholder sampler.
        Not implemented for BB8Copula.
        """
        raise NotImplementedError("Sampling not implemented for BB8Copula")

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """
        Approximate the partial derivative ∂C(u,v)/∂u for the BB8 copula.

        Args:
            u (float or array-like): First margin in (0,1).
            v (float or array-like): Second margin in (0,1).
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float or np.ndarray.
        """

        if param is None:
            param = self.get_parameters()
        theta, delta = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        X, _, log_one_minus_u = self._transform(u, theta)
        Y, _, _ = self._transform(v, theta)
        eta = 1.0 - (1.0 - delta) ** theta
        T = 1.0 - (X * Y) / eta
        T = np.clip(T, _FLOAT_MIN, 1.0)

        dX_du = theta * _safe_exp((theta - 1.0) * log_one_minus_u)  # θ(1-u)^{θ-1}
        coef = (1.0 / (delta * eta))
        return coef * Y * _safe_pow(T, 1.0 / theta - 1.0) * dX_du

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """
        Approximate the partial derivative ∂C(u,v)/∂v for the BB8 copula.

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
        Compute the lower tail dependence coefficient (LTDC) for the BB8 copula.

        Args:
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float: LTDC value = lim_{u→0} C(u,u)/u.
        """

        return 0.0

    def UTDC(self, param=None):
        """
        Compute the upper tail dependence coefficient (UTDC) for the BB8 copula.

        Args:
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float: UTDC value = lim_{u→1} [1 − 2u + C(u,u)]/(1−u).
        """
        return 0.0

    def IAD(self, data):
        """
        Return NaN for the Integrated Anderson-Darling (IAD) statistic for BB7.

        Args:
            data (Sequence[array-like, array-like]): Ignored pseudo-observations.

        Returns:
            float: Always returns numpy.nan.
        """

        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        """
        Return NaN for the Anderson-Darling (AD) statistic for BB7.

        Args:
            data (Sequence[array-like, array-like]): Ignored pseudo-observations.

        Returns:
            float: Always returns numpy.nan.
        """

        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan
