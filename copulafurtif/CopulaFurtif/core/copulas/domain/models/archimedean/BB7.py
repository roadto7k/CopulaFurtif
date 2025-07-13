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


class BB7Copula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """
    BB7 Copula (Joe-Clayton) Archimedean copula.

    Attributes:
        name (str): Name of the copula.
        type (str): Identifier for the copula family.
        bounds_param (list of tuple): Parameter bounds for optimization.
        parameters (np.ndarray): Copula parameters [theta, delta].
        default_optim_method (str): Optimization method.
    """

    def __init__(self):
        """Initialize the BB7 copula with default parameters."""
        super().__init__()
        self.name = "BB7 Copula"
        self.type = "bb7"
        self.default_optim_method = "Powell"
        self.init_parameters(CopulaParameters([2.0, 2.0], [(1e-6, np.inf), (1e-6, np.inf)], ["theta", "delta"]))

    @staticmethod
    def _phi(t, theta, delta):
        """
        Compute the φ-transform for the BB7 copula.

        Args:
            t (float or array-like): Input variable in (0,1).
            theta (float): Copula parameter θ.
            delta (float): Copula parameter δ.

        Returns:
            float or numpy.ndarray: Value of φ(t) = (φJ⁻ᵟ − 1)/δ, where φJ = 1 − (1−t)ᵗʰᵉᵗᵃ.
        """

        t = np.clip(t, 1e-12, 1 - 1e-12)
        log_one_minus_t = _safe_log(1.0 - t)  # ≤0
        # log((1-t)^θ) = θ*log(1-t)
        log_pow = theta * log_one_minus_t
        pow_term = _safe_exp(log_pow)  # (1-t)^θ, in (0,1)
        phiJ = 1.0 - pow_term  # ∈(0,1)
        log_phiJ = _safe_log(phiJ)
        phiJ_neg_delta = _safe_exp(-delta * log_phiJ)
        return (phiJ_neg_delta - 1.0) / delta

    @staticmethod
    def _phi_prime(t, theta, delta):
        """First derivative φ'(t)."""
        t = np.clip(t, 1e-12, 1 - 1e-12)
        log_one_minus_t = _safe_log(1.0 - t)
        log_pow = theta * log_one_minus_t  # log((1-t)^θ)
        pow_term = _safe_exp(log_pow)  # (1-t)^θ
        phiJ = 1.0 - pow_term
        log_phiJ = _safe_log(phiJ)

        # g'(t) for g(t)=phiJ
        g_prime = theta * _safe_exp((theta - 1.0) * log_one_minus_t) * (-1.0)
        # φ'(t) = -phiJ^{-δ-1} * g'(t)
        phi_prime = -_safe_exp((-delta - 1.0) * log_phiJ) * g_prime
        # ensure positivity (numerically phi_prime>0)
        return np.abs(phi_prime)

    @staticmethod
    def _phi_inv(s, theta, delta):
        """
        Compute the inverse φ-transform for the BB7 copula.

        Args:
            s (float or array-like): Transformed variable ≥ 0.
            theta (float): Copula parameter θ.
            delta (float): Copula parameter δ.

        Returns:
            float or numpy.ndarray: Value of φ⁻¹(s).
        """

        s = np.maximum(s, 0.0)
        temp = _safe_pow(1.0 + delta * s, -1.0 / delta)  # (1+δs)^{-1/δ}
        one_minus_temp = 1.0 - temp
        # Return 1 - (1 - temp)^{1/θ}
        return 1.0 - _safe_pow(one_minus_temp, 1.0 / theta)

    @staticmethod
    def _psi(s, theta, delta):
        return BB7Copula._phi_inv(s, theta, delta)

    @staticmethod
    def _psi_prime(s, theta, delta):
        s = np.maximum(s, 0.0)
        # reuse intermediate values for efficiency
        temp = _safe_pow(1.0 + delta * s, -1.0 / delta)
        one_minus_temp = 1.0 - temp
        dtemp_ds = -(1.0 + delta * s) ** (-1.0 / delta - 1.0)
        return (1.0 / theta) * _safe_pow(one_minus_temp, 1.0 / theta - 1.0) * (-dtemp_ds)

    @staticmethod
    def _psi_second(s, theta, delta):
        """Second derivative ψ'' needed for joint density."""
        s = np.maximum(s, 0.0)
        temp = _safe_pow(1.0 + delta * s, -1.0 / delta)
        one_minus_temp = 1.0 - temp
        dtemp_ds = -(1.0 + delta * s) ** (-1.0 / delta - 1.0)
        d2temp_ds2 = (1.0 / delta + 1.0) * delta * (1.0 + delta * s) ** (-1.0 / delta - 2.0)

        term1 = (1.0 / theta) * (1.0 / theta - 1.0) * _safe_pow(one_minus_temp, 1.0 / theta - 2.0) * (dtemp_ds ** 2)
        term2 = (1.0 / theta) * _safe_pow(one_minus_temp, 1.0 / theta - 1.0) * (-d2temp_ds2)
        return term1 + term2

    def get_cdf(self, u, v, param=None):
        """
        Evaluate the BB7 copula cumulative distribution function at (u, v).

        Args:
            u (float or array-like): First uniform margin in (0,1).
            v (float or array-like): Second uniform margin in (0,1).
            param (Sequence[float], optional): Copula parameters (θ, δ). Defaults to self.get_parameters().

        Returns:
            float or numpy.ndarray: CDF value C(u, v).
        """

        if param is None:
            param = self.get_parameters()
        theta, delta = param
        s = self._phi(u, theta, delta) + self._phi(v, theta, delta)
        return self._psi(s, theta, delta)

    def get_pdf(self, u, v, param=None):
        """
        Approximate the BB7 copula probability density function at (u, v) via finite differences.

        Args:
            u (float or array-like): First uniform margin in (0,1).
            v (float or array-like): Second uniform margin in (0,1).
            param (Sequence[float], optional): Copula parameters (θ, δ). Defaults to self.get_parameters().

        Returns:
            float or numpy.ndarray: Approximate PDF c(u, v).
        """

        if param is None:
            param = self.get_parameters()
        theta, delta = param

        # compute φ, φ', ψ''
        phi_u = self._phi(u, theta, delta)
        phi_v = self._phi(v, theta, delta)
        s = phi_u + phi_v
        log_phi_u_prime = _safe_log(self._phi_prime(u, theta, delta))
        log_phi_v_prime = _safe_log(self._phi_prime(v, theta, delta))
        log_psi_second = _safe_log(np.abs(self._psi_second(s, theta, delta)))

        log_pdf = log_psi_second + log_phi_u_prime + log_phi_v_prime
        pdf = _safe_exp(log_pdf)
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
        Not implemented for BB7Copula.
        """
        raise NotImplementedError("Sampling not implemented for BB7Copula")

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """
        The partial derivative ∂C(u,v)/∂u for the BB7 copula.

        Args:
            u (float or array-like): First uniform margin in (0,1).
            v (float or array-like): Second uniform margin in (0,1).
            param (Sequence[float], optional): Copula parameters (θ, δ). Defaults to self.get_parameters().

        Returns:
            float or numpy.ndarray: Approximate ∂C/∂u at (u, v).
        """

        if param is None:
            param = self.get_parameters()
        theta, delta = param
        # φ(u), φ(v)
        phi_u = self._phi(u, theta, delta)
        phi_v = self._phi(v, theta, delta)
        s = phi_u + phi_v
        # ψ(s)
        psi_val = self._psi(s, theta, delta)
        # φ'(u) and φ'(ψ(s))
        phi_u_prime = self._phi_prime(u, theta, delta)
        phi_psi_prime = self._phi_prime(psi_val, theta, delta)
        return phi_u_prime / phi_psi_prime

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """
        Approximate the partial derivative ∂C(u,v)/∂v for the BB7 copula.

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
        Compute the lower tail dependence coefficient (LTDC) for the BB7 copula.

        Args:
            param (Sequence[float], optional): Copula parameters (θ, δ). Defaults to self.get_parameters().

        Returns:
            float: LTDC value.
        """

        if param is None:
            param = self.get_parameters()
        delta = param[1]
        # 2^(-1/δ) en safe
        return _safe_pow(2.0, -1.0 / delta)

    def UTDC(self, param=None):
        """
        Compute the upper tail dependence coefficient (UTDC) for the BB7 copula.

        Args:
            param (Sequence[float], optional): Copula parameters (θ, δ). Defaults to self.get_parameters().

        Returns:
            float: UTDC value = 2 − lim₍u→1₎ [1 − 2u + C(u,u)]/(1−u).
        """

        if param is None:
            param = self.get_parameters()
        theta = param[0]
        # 2^(1/θ) en safe, puis 2 - …
        two_pow = _safe_pow(2.0, 1.0 / theta)
        return 2.0 - two_pow


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
