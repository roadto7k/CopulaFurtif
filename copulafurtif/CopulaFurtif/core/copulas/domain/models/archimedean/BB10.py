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


class BB10Copula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """
    BB10 Copula (Crowder) Archimedean copula.

    Attributes:
        name (str): Name of the copula.
        type (str): Identifier for the copula family.
        bounds_param (list of tuple): Parameter bounds for optimization.
        parameters (np.ndarray): Copula parameters [theta, delta].
        default_optim_method (str): Optimization method.
    """

    def __init__(self):
        """Initialize the BB10 copula with default parameters."""
        super().__init__()
        self.name = "BB10 Copula"
        self.type = "bb10"
        self.default_optim_method = "Powell"
        self.init_parameters(CopulaParameters(np.array([2.0, 0.5]), [(1e-6, np.inf), (1e-6, 1.0)], ["theta", "pi"]))


    def get_cdf(self, u, v, param=None):
        """Return :math:`C(u,v)=uv\\,[1-\\pi(1-u^{\\theta})(1-v^{\\theta})]^{-1/\\theta}`.

        Args:
            u (float | np.ndarray): First uniform margin *(0, 1)*.
            v (float | np.ndarray): Second uniform margin *(0, 1)*.
            param (Sequence[float], optional):
                Copula parameters ``[theta, pi]``.
                Defaults to current model parameters.

        Returns:
            float | np.ndarray: Copula CDF :math:`C(u,v)`.
        """
        if param is None:
            param = self.get_parameters()
        theta, pi = param

        u = np.clip(u, 1e-12, 1 - 1e-12)
        v = np.clip(v, 1e-12, 1 - 1e-12)

        u_theta = _safe_pow(u, theta)           # u^{θ}
        v_theta = _safe_pow(v, theta)           # v^{θ}
        T = 1.0 - pi * (1.0 - u_theta) * (1.0 - v_theta)
        T = np.clip(T, 1e-12, None)             # positivity guard

        return u * v * _safe_pow(T, -1.0 / theta)

    def get_pdf(self, u, v, param=None):
        r"""Return analytic density

        .. math::

            c(u,v)=\,[1-\pi(1-u^{\theta})(1-v^{\theta})]^{-1/\theta-2}\,
                    \{\,
                        1-\pi+\pi(1+\theta)u^{\theta}v^{\theta}
                        -\pi(1-u^{\theta})(1-v^{\theta})
                    \}.

        Args:
            u, v (float | np.ndarray): Uniform margins.
            param (Sequence[float], optional): ``[theta, pi]``.

        Returns:
            float | np.ndarray: Joint density :math:`c(u,v)\\ge0`.
        """
        if param is None:
            param = self.get_parameters()
        theta, pi = param

        u = np.clip(u, 1e-12, 1 - 1e-12)
        v = np.clip(v, 1e-12, 1 - 1e-12)

        u_theta = _safe_pow(u, theta)
        v_theta = _safe_pow(v, theta)

        T = 1.0 - pi * (1.0 - u_theta) * (1.0 - v_theta)
        T = np.clip(T, 1e-12, None)

        log_pref = (-1.0 / theta - 2.0) * _safe_log(T)

        inner = (
            1.0 - pi
            + pi * (1.0 + theta) * u_theta * v_theta
            - pi * (1.0 - u_theta) * (1.0 - v_theta)
        )
        inner = np.maximum(inner, 0.0)          # avoid tiny negatives by round‑off

        return _safe_exp(log_pref) * inner

    def kendall_tau(self, param=None):
        """
        Placeholder for Kendall's tau.
        Currently unimplemented; returns NaN.
        """
        return np.nan

    def sample(self, n, random_state=None):
        """
        Placeholder sampler.
        Not implemented for BB10Copula.
        """
        raise NotImplementedError("Sampling not implemented for BB10Copula")

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """Compute :math:`\\partial C/\\partial u`.

        Formula
        .. math::

            C_u = C\\Bigl[\\tfrac1u
                     + \\frac{\\pi u^{\\theta-1}(1-v^{\\theta})}{T}\\Bigr],
            \\qquad
            T=1-\\pi(1-u^{\\theta})(1-v^{\\theta}).

        Returns:
            float | np.ndarray: :math:`\\partial C/\\partial u`.
        """
        if param is None:
            param = self.get_parameters()
        theta, pi = param

        u = np.clip(u, 1e-12, 1 - 1e-12)
        v = np.clip(v, 1e-12, 1 - 1e-12)

        utheta = _safe_pow(u, theta)
        vtheta = _safe_pow(v, theta)
        T = 1.0 - pi * (1.0 - utheta) * (1.0 - vtheta)
        T = np.clip(T, 1e-12, None)

        C = u * v * _safe_pow(T, -1.0 / theta)

        return C * (1.0 / u - pi * utheta / u * (1.0 - vtheta) / T)

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """
        Approximate the partial derivative ∂C(u,v)/∂v for the BB10 copula.

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
        Compute the lower tail dependence coefficient (LTDC) for the BB10 copula.

        Args:
            param (Sequence[float], optional): Copula parameters (θ, δ). Defaults to self.get_parameters().

        Returns:
            float: LTDC value.
        """

        return 0.0

    def UTDC(self, param=None):
        """
        Compute the upper tail dependence coefficient (UTDC) for the BB10 copula.

        Args:
            param (Sequence[float], optional): Copula parameters (θ, δ). Defaults to self.get_parameters().

        Returns:
            float: UTDC value = 2 − lim₍u→1₎ [1 − 2u + C(u,u)]/(1−u).
        """

        return 0.0


    def IAD(self, data):
        """
        Return NaN for the Integrated Anderson-Darling (IAD) statistic for BB10.

        Args:
            data (Sequence[array-like, array-like]): Ignored pseudo-observations.

        Returns:
            float: Always returns numpy.nan.
        """

        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        """
        Return NaN for the Anderson-Darling (AD) statistic for BB10.

        Args:
            data (Sequence[array-like, array-like]): Ignored pseudo-observations.

        Returns:
            float: Always returns numpy.nan.
        """

        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan
