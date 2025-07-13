import numpy as np
from numpy import log as _np_log, exp as _np_exp

from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel, CopulaParameters
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence

_FLOAT_MIN = 1e-308
_FLOAT_MAX_LOG = 709.0  # exp(709) ~ 8e307 < float64 max
_FLOAT_MIN_LOG = -745.0  # exp(-745) ~ 5e-324 > 0


def _safe_log(x):
    """Log clipped away from 0 to avoid -inf."""
    return _np_log(np.clip(x, _FLOAT_MIN, None))


def _safe_exp(log_x):
    """Exp with clipping of the exponent so the result stays finite."""
    return _np_exp(np.clip(log_x, _FLOAT_MIN_LOG, _FLOAT_MAX_LOG))


def _safe_pow(base, exponent):
    """Stable power via exp(exponent * log(base)). base must be >0."""
    return _safe_exp(exponent * _safe_log(base))


class BB6Copula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """
    BB6 Copula (Archimedean with generator φ(t) = [-ln(1-(1-t)ˆθ)]ˆ(1/δ)).

    Attributes:
        name (str): Human-readable name of the copula.
        type (str): Identifier for the copula family.
        bounds_param (list of tuple): Bounds for copula parameters [theta, delta].
        parameters (np.ndarray): Current copula parameters.
        default_optim_method (str): Optimization method.
    """

    def __init__(self):
        """Initialize BB6 copula with default parameters theta=2.0, delta=2.0."""
        super().__init__()
        self.name = "BB6 Copula"
        self.type = "bb6"
        self.default_optim_method = "Powell"
        self.init_parameters(CopulaParameters([2.0, 2.0], [(1.0, np.inf), (1.0, np.inf)], ["theta", "delta"]))

    @staticmethod
    def _transform(u, theta):
        """Return x = -log(1 - (1-u)^θ) with stable numerics."""
        ubar = 1.0 - u
        log_ubar_theta = theta * _safe_log(ubar)
        one_minus_pow = 1.0 - _safe_exp(log_ubar_theta)
        return -_safe_log(one_minus_pow), ubar, log_ubar_theta, one_minus_pow



    def _phi(self, t, theta, delta):
        """
        Compute the generator function φ(t) = (−log(1 − (1 − t)^θ))^(1/δ).

        Args:
            t (float or array-like): Input variable in [0,1].
            theta (float): Copula parameter θ.
            delta (float): Copula parameter δ.

        Returns:
            float or numpy.ndarray: Value of φ(t).
        """

        return (-np.log(1.0 - (1.0 - t)**theta))**(1.0 / delta)

    def _phi_prime(self, t, theta, delta):
        """
        Compute the derivative φ′(t) of the generator function.

        Args:
            t (float or array-like): Input variable in [0,1].
            theta (float): Copula parameter θ.
            delta (float): Copula parameter δ.

        Returns:
            float or numpy.ndarray: Value of φ′(t).
        """

        g = 1.0 - (1.0 - t)**theta
        gp = theta * (1.0 - t)**(theta - 1)
        L = -np.log(g)
        Lp = -gp / g
        return (1.0 / delta) * L**(1.0 / delta - 1.0) * Lp

    def get_cdf(self, u, v, param=None):
        """
        Evaluate the copula cumulative distribution function C(u, v).

        Args:
            u (float or array-like): First uniform margin in (0,1).
            v (float or array-like): Second uniform margin in (0,1).
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float or numpy.ndarray: CDF value C(u, v).
        """

        if param is None:
            param = self.get_parameters()
        theta, delta = param

        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        # Transform margins
        x, ubar, _, _ = self._transform(u, theta)
        y, vbar, _, _ = self._transform(v, theta)

        # Work mostly in log space
        log_xd = delta * _safe_log(x)
        log_yd = delta * _safe_log(y)
        log_sum = np.logaddexp(log_xd, log_yd)  # log(x^δ + y^δ)
        log_tem = (1.0 / delta) * log_sum  # log((x^δ+y^δ)^{1/δ})

        # w = exp(-tem)
        w = _safe_exp(-_safe_exp(log_tem))
        # C(u,v) = 1 - (1 - w)^{1/θ}
        C = 1.0 - _safe_pow(1.0 - w, 1.0 / theta)
        return C

    def get_pdf(self, u, v, param=None):
        """
        Compute the partial derivative ∂C(u,v)/∂u of the copula CDF.

        Args:
            u (float or array-like): First margin in (0,1).
            v (float or array-like): Second margin in (0,1).
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float or numpy.ndarray: Value of ∂C/∂u at (u, v).
        """

        if param is None:
            param = self.get_parameters()
        theta, delta = param

        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        # Transform margins + keep auxiliary terms
        x, ubar, log_ubar_theta, one_minus_u_pow = self._transform(u, theta)
        y, vbar, log_vbar_theta, one_minus_v_pow = self._transform(v, theta)

        # Pre‑compute powers/logs
        log_xd = delta * _safe_log(x)
        log_yd = delta * _safe_log(y)
        log_sum = np.logaddexp(log_xd, log_yd)
        log_tem = (1.0 / delta) * log_sum
        tem = _safe_exp(log_tem)

        w = _safe_exp(-tem)
        one_minus_w = 1.0 - w

        # Jacobian pieces (all positive)
        dx_du = -theta * _safe_exp((theta - 1) * _safe_log(ubar)) / one_minus_u_pow  # from paper
        dy_dv = -theta * _safe_exp((theta - 1) * _safe_log(vbar)) / one_minus_v_pow

        # Partial derivatives of G in log‑form
        # g1 = ∂G/∂x = (1/θ)(1-w)^{1/θ-1} * (-w) * ∂tem/∂x
        pow_factor = _safe_pow(one_minus_w, 1.0 / theta - 1.0)
        dtem_dx = _safe_exp(log_xd - log_sum)  # = x^{δ-1} / (x^{δ}+y^{δ})^{1 - 1/δ}
        g1 = (1.0 / theta) * pow_factor * (-w) * dtem_dx

        dtem_dy = _safe_exp(log_yd - log_sum)
        g2 = (1.0 / theta) * pow_factor * (-w) * dtem_dy

        # log_pdf = log(g1) + log(g2) + log|dx/du| + log|dy/dv|
        log_pdf = (
                _safe_log(np.abs(g1)) + _safe_log(np.abs(g2)) + _safe_log(np.abs(dx_du)) + _safe_log(np.abs(dy_dv))
        )
        pdf = _safe_exp(log_pdf)
        # Clamp tiny negative due to rounding
        return np.maximum(pdf, 0.0)

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """
        Compute the partial derivative ∂C(u,v)/∂u of the copula CDF.

        Args:
            u (float or array-like): First margin in (0,1).
            v (float or array-like): Second margin in (0,1).
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float or numpy.ndarray: Value of ∂C/∂u at (u, v).
        """

        if param is None:
            param = self.get_parameters()
        theta, delta = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        # reuse transform & log helpers
        x, ubar, _, one_minus_u_pow = self._transform(u, theta)
        y, _, _, _ = self._transform(v, theta)
        log_xd = delta * _safe_log(x)
        log_yd = delta * _safe_log(y)
        log_sum = np.logaddexp(log_xd, log_yd)
        log_tem = (1.0 / delta) * log_sum
        tem = _safe_exp(log_tem)
        w = _safe_exp(-tem)

        pow_factor = _safe_pow(1.0 - w, 1.0 / theta - 1.0)
        dtem_dx = _safe_exp((delta-1.0)*_safe_log(x) + (1.0/delta-1.0)*log_sum)
        g1 = (1.0 / theta) * pow_factor * (-w) * dtem_dx
        dx_du = -theta * _safe_exp((theta - 1) * _safe_log(ubar)) / one_minus_u_pow
        return g1 * dx_du

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """
        Compute the partial derivative ∂C(u,v)/∂v of the copula CDF.

        Args:
            u (float or array-like): First margin in (0,1).
            v (float or array-like): Second margin in (0,1).
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float or numpy.ndarray: Value of ∂C/∂v at (u, v).
        """

        return self.partial_derivative_C_wrt_u(v, u, param)

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
        raise NotImplementedError("Sampling not implemented for BB6Copula")

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
        theta, delta = param
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