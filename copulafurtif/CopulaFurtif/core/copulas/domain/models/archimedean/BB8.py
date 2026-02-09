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
        self.init_parameters(CopulaParameters(np.array([2.0, 0.7]), [(1.0, np.inf), (1e-6, 1.0)], ["theta", "delta"]))

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

        log1mu = _safe_log(1.0 - u)
        log1mv = _safe_log(1.0 - v)
        log_X = _safe_log(1.0 - _safe_exp(theta * log1mu))
        log_Y = _safe_log(1.0 - _safe_exp(theta * log1mv))
        X = _safe_exp(log_X)
        Y = _safe_exp(log_Y)
        A = _safe_pow(X, delta)
        B = _safe_pow(Y, delta)
        Z = np.clip(A + B - A * B, _FLOAT_MIN, 1.0)

        # d²C/(du dv)  = (1/θ)(1/θ-1) Z^{1/θ-2} (1-B)(1-A) δ² X^{δ-1}Y^{δ-1} θ² (1-u)^{θ-1}(1-v)^{θ-1}
        log_pdf = (
                _safe_log(delta) * 2
                + 2.0 * _safe_log(theta)
                + _safe_log(1.0 - A)
                + _safe_log(1.0 - B)
                + (1.0 / theta - 2.0) * _safe_log(Z)
                + (delta - 1.0) * (log_X + log_Y)
                + _safe_log(_safe_exp((theta - 1.0) * log1mu))
                + _safe_log(_safe_exp((theta - 1.0) * log1mv))
                + _safe_log(1.0 / theta)  # prefactor
                + _safe_log(abs(1.0 / theta - 1.0))  # second derivative factor
        )
        pdf_val = _safe_exp(log_pdf)
        return np.maximum(pdf_val, 0.0)

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

        # log‑helpers
        log1mu = _safe_log(1.0 - u)
        log1mv = _safe_log(1.0 - v)

        # X, Y, A, B
        log_X = _safe_log(1.0 - _safe_exp(theta * log1mu))  # log X
        log_Y = _safe_log(1.0 - _safe_exp(theta * log1mv))
        log_A = delta * log_X
        log_B = delta * log_Y
        A = _safe_exp(log_A)
        B = _safe_exp(log_B)

        # Z = 1-(1-A)(1-B) = A+B-AB
        Z = A + B - A * B
        Z = np.clip(Z, _FLOAT_MIN, 1.0)  # pour éviter 0
        log_Z = _safe_log(Z)

        # dX/du
        dX_du = theta * _safe_exp((theta - 1.0) * log1mu)
        # dA/du = δ X^{δ-1} dX/du  -> log(dA/du)
        log_dA_du = _safe_log(delta) + (delta - 1.0) * log_X + _safe_log(dX_du)

        # (1-B)
        log_1mB = _safe_log(1.0 - B)

        # log(C_u) = log(δ) - log(θ) + log(1-B) + (1/θ-1)log Z + (δ-1)log X + log dX/du - log X
        log_Cu = (
                _safe_log(delta)
                - _safe_log(theta)
                + log_1mB
                + (1.0 / theta - 1.0) * log_Z
                + (delta - 1.0) * log_X
                + _safe_log(dX_du)
        )
        return _safe_exp(log_Cu)

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
