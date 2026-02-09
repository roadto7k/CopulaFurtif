import numpy as np
from numpy import log as _np_log, exp as _np_exp

from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel, CopulaParameters
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence

_FLOAT_MIN = 1e-308  # strictly positive, smallest normal double
_FLOAT_MAX_LOG = 709.0  # np.exp(709) ~= 8.2e307, still in float64 range
_FLOAT_MIN_LOG = -745.0  # np.exp(-745) ~= 5e-324, smallest sub‑normal


def _safe_log(x):
    """log clipped away from 0.  Works on arrays or scalars."""
    return _np_log(np.clip(x, _FLOAT_MIN, None))


def _safe_exp(log_x):
    """exp with log‑input clipping so the result is finite."""
    return _np_exp(np.clip(log_x, _FLOAT_MIN_LOG, _FLOAT_MAX_LOG))


def _safe_pow(base, exponent):
    """Compute base**exponent robustly for base>0 via exp(exponent*log(base))."""
    return _safe_exp(exponent * _safe_log(base))


class BB5Copula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """
    BB5 Copula (Joe's two-parameter extreme-value copula).

    Attributes:
        name (str): Human-readable name of the copula.
        type (str): Identifier for the copula family.
        bounds_param (list of tuple): Bounds for copula parameters [theta, delta].
        parameters (np.ndarray): Current copula parameters.
        default_optim_method (str): Optimization method.
    """

    def __init__(self):
        """Initialize BB5 copula with default parameters theta=1.0, delta=1.0."""
        super().__init__()
        self.name = "BB5 Copula"
        self.type = "bb5"
        self.default_optim_method = "Powell"
        self.init_parameters(CopulaParameters(np.array([2.0, 2.0]), [(1.0, np.inf), (1e-6, np.inf)], ["theta", "delta"]))
        
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

        # Clip inputs away from 0 and 1 to avoid log(0)
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        x = -_safe_log(u)
        y = -_safe_log(v)

        # Pre‑compute powers in log‑space where possible
        log_xt = theta * _safe_log(x)
        log_yt = theta * _safe_log(y)

        log_xdt = -delta * theta * _safe_log(x)  # log(x^(-delta*theta))
        log_ydt = -delta * theta * _safe_log(y)

        # log‑sum‑exp for S = x^(-δθ)+y^(-δθ)
        log_Sdt = np.logaddexp(log_xdt, log_ydt)
        # (x^(-δθ)+y^(-δθ))^{-1/δ}
        log_xyp = -(1.0 / delta) * log_Sdt

        # Go back to real space where safe
        xt = _safe_exp(log_xt)
        yt = _safe_exp(log_yt)
        xyp = _safe_exp(log_xyp)

        w = xt + yt - xyp
        w = np.clip(w, 1e-300, np.inf)  # guarantee strictly positive

        g = _safe_pow(w, 1.0 / theta)
        log_C = -g
        C = _safe_exp(log_C)
        return C

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

        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        x = -_safe_log(u)
        y = -_safe_log(v)

        # Same log‑space machinery as in CDF
        log_xt = theta * _safe_log(x)
        log_yt = theta * _safe_log(y)
        log_xdt = -delta * theta * _safe_log(x)
        log_ydt = -delta * theta * _safe_log(y)
        log_Sdt = np.logaddexp(log_xdt, log_ydt)
        log_xyp = -(1.0 / delta) * log_Sdt

        xt = _safe_exp(log_xt)
        yt = _safe_exp(log_yt)
        xdt = _safe_exp(log_xdt)
        ydt = _safe_exp(log_ydt)
        xyp = _safe_exp(log_xyp)
        S = xdt + ydt  # in real space; finite by construction

        w = xt + yt - xyp
        w = np.clip(w, 1e-300, np.inf)

        g = _safe_pow(w, 1.0 / theta)
        C = _safe_exp(-g)

        # Derivative helpers (all finite by design)
        dx_du = -1.0 / u
        dy_dv = -1.0 / v

        dxt_du = theta * _safe_pow(x, theta - 1) * dx_du
        dyt_dv = theta * _safe_pow(y, theta - 1) * dy_dv

        dxdt_du = -delta * theta * _safe_pow(x, -delta * theta - 1) * dx_du
        dydt_dv = -delta * theta * _safe_pow(y, -delta * theta - 1) * dy_dv

        S_pow = _safe_pow(S, -1.0 / delta - 1)
        dw_du = dxt_du - (-(1.0 / delta) * S_pow * dxdt_du)
        dw_dv = dyt_dv - (-(1.0 / delta) * S_pow * dydt_dv)

        w_uv = -(
                (1.0 / delta) * (1.0 / delta + 1.0) * _safe_pow(S, -1.0 / delta - 2) * dxdt_du * dydt_dv
        )

        dg_dw = (1.0 / theta) * _safe_pow(w, 1.0 / theta - 1)
        d2g_dw2 = (1.0 / theta) * (1.0 / theta - 1.0) * _safe_pow(w, 1.0 / theta - 2)

        term1 = -C * dg_dw * w_uv
        term2 = -C * d2g_dw2 * dw_du * dw_dv
        term3 = C * (dg_dw ** 2) * dw_du * dw_dv

        pdf = term1 + term2 + term3
        return np.clip(pdf, 0.0, None)

    def kendall_tau(self, param=None):
        """
        Placeholder for Kendall's tau.
        Currently unimplemented; returns NaN.
        """
        return np.nan

    def sample(self, n, random_state=None):
        """
        Placeholder sampler.
        Not implemented for BB4Copula.
        """
        raise NotImplementedError("Sampling not implemented for BB5Copula")

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """
        Compute the partial derivative ∂C(u,v)/∂u of the copula CDF.

        Args:
            u (float or array-like): First margin in (0,1).
            v (float or array-like): Second margin in (0,1).
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float or np.ndarray: Value of ∂C/∂u at (u, v).
        """

        if param is None:
            param = self.get_parameters()
        theta, delta = param

        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        x = -_safe_log(u)
        y = -_safe_log(v)

        log_xdt = -delta * theta * _safe_log(x)
        log_ydt = -delta * theta * _safe_log(y)
        log_Sdt = np.logaddexp(log_xdt, log_ydt)
        S_pow = _safe_pow(_safe_exp(log_Sdt), -1.0 / delta - 1)

        w = _safe_pow(x, theta) + _safe_pow(y, theta) - _safe_pow(_safe_exp(log_Sdt), -1.0 / delta)
        w = np.clip(w, 1e-300, np.inf)

        C = _safe_exp(-_safe_pow(w, 1.0 / theta))
        dg_dw = (1.0 / theta) * _safe_pow(w, 1.0 / theta - 1)

        dxt_du = theta * _safe_pow(x, theta - 1) * (-1.0 / u)
        dxdt_du = -delta * theta * _safe_pow(x, -delta * theta - 1) * (-1.0 / u)
        dw_du = dxt_du - (-(1.0 / delta) * S_pow * dxdt_du)
        return -C * dg_dw * dw_du

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """
        Compute the partial derivative ∂C(u,v)/∂v of the copula CDF.

        Args:
            u (float or array-like): First margin in (0,1).
            v (float or array-like): Second margin in (0,1).
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float or np.ndarray: Value of ∂C/∂v at (u, v).
        """

        return self.partial_derivative_C_wrt_u(v, u, param)

    def LTDC(self, param=None):
        """
        Compute the lower tail dependence coefficient (LTDC) of the copula.

        Args:
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float: LTDC value (0.0 for this copula).
        """

        return 0.0

    def UTDC(self, param=None):
        """
        Compute the upper tail dependence coefficient (UTDC) of the copula.

        Args:
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float: UTDC value (2 − 2^(1/δ)).
        """

        if param is None:
            param = self.get_parameters()

        theta = param[0]
        delta = param[1]
        return 2.0 - (2.0 - 2.0 ** (-1.0 / delta)) ** (1.0 / theta)

    def IAD(self, data):
        """
        Return NaN for the Integrated Anderson-Darling (IAD) statistic.

        Args:
            data (Sequence[array-like, array-like]): Ignored pseudo-observations.

        Returns:
            float: Always returns numpy.nan.
        """

        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        """
        Return NaN for the Anderson-Darling (AD) statistic.

        Args:
            data (Sequence[array-like, array-like]): Ignored pseudo-observations.

        Returns:
            float: Always returns numpy.nan.
        """

        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan