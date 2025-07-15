import math

import numpy as np
from numpy.random import default_rng
from scipy.optimize import brentq
from scipy.stats import levy_stable

from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel, CopulaParameters
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence

_FLOAT_MIN = 1e-308  # min positive normal float64
_FLOAT_MIN_LOG = -745.0   # ~ log(min float64)
_FLOAT_MAX_LOG =  710.0   # ~ log(max float64)

def _safe_log(x):
    """Compute log(x) with clipping to avoid –inf when x → 0."""
    return np.log(np.clip(x, 1e-308, None))

def _safe_exp(x):
    """Compute exp(x) with clipping to avoid overflow."""
    x = np.clip(x, None, _FLOAT_MAX_LOG)
    return np.exp(x)

def _safe_pow(x, p):
    """Compute x**p safely for x > 0."""
    return np.exp(p * _safe_log(x))

def _safe_log1p(x):
    """Compute log(1 + x) in a numerically stable way (x may be small)."""
    return np.log1p(x)


class BB3Copula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """
    BB3 (Joe–Hu 1996) – Positive-stable stopped-gamma LT copula..

    Attributes:
        name (str): Human-readable name of the copula.
        type (str): Identifier for the copula family.
        bounds_param (list of tuple): Parameter bounds [d > 0, q >= 1].
        parameters (np.ndarray): Current copula parameters.
        default_optim_method (str): Optimization method.
    """

    def __init__(self):
        """Initialize BB3 copula with default parameters d=1.0, q=1.0."""
        super().__init__()
        self.name = "BB3 Copula"
        self.type = "bb3"
        self.default_optim_method = "Powell"
        self.init_parameters(CopulaParameters([2, 1.5], [(1.0, 10.0),(0.05, 10.0) ], ["theta", "delta"]))


    def _h(self, s, param=None):
        """
        Compute the generator function h(s) = (log1p(s) / theta)^(1/delta).

        Args:
            s (float or array-like): Input to the generator.
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float or np.ndarray: Value of h(s).
        """

        theta, delta = self.get_parameters() if param is None else param
        return _safe_pow(_safe_log1p(s) / delta, 1.0 / theta)

    def _h_prime(self, s, param=None):
        """
        Compute the first derivative hʼ(s) of the generator function.

        Args:
            s (float or array-like): Input to the generator.
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float or np.ndarray: Value of hʼ(s).
        """

        theta, delta = self.get_parameters() if param is None else param
        g = _safe_log1p(s) / delta  # g>0
        log_hp = (1.0 / theta - 1.0) * _safe_log(g) \
                 - _safe_log(theta) - _safe_log(delta) \
                 - _safe_log1p(s)
        return _safe_exp(log_hp)

    def _h_double(self, s, param=None):
        """
        Compute the second derivative h″(s) of the generator function.

        Args:
            s (float or array-like): Input to the generator.
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float or np.ndarray: Value of h″(s).
        """

        theta, delta = self.get_parameters() if param is None else param
        inv_theta = 1.0 / theta
        g = _safe_log1p(s) / delta
        # log(h'') = log(h') + log((inv_theta-1)-δg) - log(δ(1+s))
        hp = self._h_prime(s, param)
        tmp = (inv_theta - 1.0) - delta * g
        return hp * tmp / (delta * (1.0 + s))

    def get_cdf(self, u, v, param=None):
        """
        Evaluate the copula CDF at points (u, v) using the  BB3 positive-stable stopped-gamma.

        Args:
            u (float or array-like): First uniform margin in (0,1).
            v (float or array-like): Second uniform margin in (0,1).
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float or np.ndarray: CDF value C(u, v).
        """

        theta, delta = self.get_parameters() if param is None else param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        # s_u = expm1( δ (‑log u)^θ )
        log_su = _safe_log(_safe_exp(delta * _safe_pow(-_safe_log(u), theta)) - 1.0)
        log_sv = _safe_log(_safe_exp(delta * _safe_pow(-_safe_log(v), theta)) - 1.0)

        s = _safe_exp(log_su) + _safe_exp(log_sv)
        return _safe_exp(-self._h(s, param))

    def get_pdf(self, u, v, param=None):
        """
        Evaluate the copula PDF at points (u, v) using the  BB3 positive-stable stopped-gamma.

        Args:
            u (float or array-like): First uniform margin in (0,1).
            v (float or array-like): Second uniform margin in (0,1).
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float or np.ndarray: PDF value c(u, v).
        """

        # ------------------------------------------------------------------ #
        # 0 – parameters & sanitising                                        #
        # ------------------------------------------------------------------ #
        if param is None:
            param = self.get_parameters()
        theta, delta = map(float, param)
        inv_theta = 1.0 / theta

        eps = 1e-12
        u = np.clip(u, eps, 1.0 - eps)
        v = np.clip(v, eps, 1.0 - eps)

        # ------------------------------------------------------------------ #
        # 1 – s,  h, h', h''                                                 #
        # ------------------------------------------------------------------ #
        su = _safe_exp(delta * _safe_pow(-_safe_log(u), theta)) - 1.0
        sv = _safe_exp(delta * _safe_pow(-_safe_log(v), theta)) - 1.0
        s = su + sv

        h = self._h(s, param)  # scalar / ndarray
        h1 = self._h_prime(s, param)
        h2 = self._h_double(s, param)

        # log φʺ(s)  (guard against inf / negative)
        core = h1 * h1 - h2
        core = np.where((core <= 0) | (~np.isfinite(core)), _FLOAT_MIN, core)
        log_phi_dd = -h + _safe_log(core)

        # ------------------------------------------------------------------ #
        # 2 – log ∂s/∂u  and  log ∂s/∂v                                      #
        # ------------------------------------------------------------------ #
        log_du_s = (_safe_log(delta) + _safe_log(theta)
                    + (theta - 1.0) * _safe_log(-_safe_log(u))
                    + delta * _safe_pow(-_safe_log(u), theta)
                    - _safe_log(u))

        log_dv_s = (_safe_log(delta) + _safe_log(theta)
                    + (theta - 1.0) * _safe_log(-_safe_log(v))
                    + delta * _safe_pow(-_safe_log(v), theta)
                    - _safe_log(v))

        # ------------------------------------------------------------------ #
        # 3 – assemble log‑pdf and exp                                       #
        # ------------------------------------------------------------------ #
        log_pdf = log_phi_dd + log_du_s + log_dv_s

        # under/overflow guards, then exp
        log_pdf = np.where(log_pdf < _FLOAT_MIN_LOG, -np.inf, log_pdf)
        log_pdf = np.where(log_pdf > _FLOAT_MAX_LOG, np.inf, log_pdf)
        pdf = _safe_exp(log_pdf)

        # make absolutely sure: replace NaN with 0, Inf with max‑float
        pdf = np.nan_to_num(pdf,nan=0.0,posinf=np.finfo(float).max,neginf=0.0)

        return pdf

    def sample(self, n, param=None, rng=None, eps=1e-12):
        raise NotImplementedError("BB3 sampling not implemented")

    def kendall_tau(self, param=None, m: int = 800) -> float:
        raise NotImplementedError("BB3 kendall_tau not implemented")

    def LTDC(self, param=None):
        """
        Compute the lower tail dependence coefficient (LTDC) of the copula.

        Args:
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float: LTDC value
        """
        theta, delta = self.get_parameters() if param is None else param
        return 1.0 if theta > 1.0 else 2.0 ** (-1.0 / delta)

    def UTDC(self, param=None):
        """
        Compute the upper tail dependence coefficient (UTDC) of the copula.

        Args:
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float: UTDC value (2 − 2^(1/q)).
        """

        theta, delta = self.get_parameters() if param is None else param
        return 2.0 - 2.0 ** (1.0 / theta)

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """
        ∂C(u,v)/∂u for the BB3 (Joe–Hu) copula.

        Parameters
        ----------
        u, v : float or np.ndarray
            Margins in (0, 1).
        param : (theta, delta) tuple, optional
            Copula parameters. If None -> self.get_parameters().

        Returns
        -------
        float or np.ndarray
            The partial derivative dC/du evaluated at (u, v).
        """
        # ------------------------------------------------------------------ #
        # Parameters & constants                                         #
        # ------------------------------------------------------------------ #
        if param is None:
            param = self.get_parameters()
        theta, delta = map(float, param)
        inv_theta = 1.0 / theta

        # ------------------------------------------------------------------ #
        # Sanitise inputs                                                #
        # ------------------------------------------------------------------ #
        eps = 1e-12
        u = np.clip(u, eps, 1.0 - eps)
        v = np.clip(v, eps, 1.0 - eps)

        # ------------------------------------------------------------------ #
        # Pre‑compute logs & intermediates                               #
        # ------------------------------------------------------------------ #
        log_u_neg = -_safe_log(u)  # -log u  > 0
        log_v_neg = -_safe_log(v)

        # δ(−log u)^θ and δ(−log v)^θ
        exp_u = delta * _safe_pow(log_u_neg, theta)
        exp_v = delta * _safe_pow(log_v_neg, theta)

        # gardés pour ds/du et ds/dv
        log_su = exp_u  # = log(1+su)−log(1)
        log_sv = exp_v  # (utilisé ailleurs)

        # su = e^{…}−1  (expm1‑like pour la précision)
        su = _safe_exp(exp_u) - 1.0
        sv = _safe_exp(exp_v) - 1.0
        s = su + sv

        log1p_s = _safe_log1p(s)  # log(1+s)
        g = log1p_s / delta
        log_g = _safe_log(g)

        # ------------------------------------------------------------------ #
        # h(g) et h′(g)                                                  #
        # ------------------------------------------------------------------ #
        h = _safe_pow(g, inv_theta)  # g^{1/θ}
        log_hprime = ((inv_theta - 1.0) * log_g
                      - _safe_log(theta)
                      - _safe_log(delta)
                      - log1p_s)

        # ------------------------------------------------------------------ #
        # log(ds/du)                                                     #
        # ------------------------------------------------------------------ #
        log_dsdu = (_safe_log(delta) + _safe_log(theta)
                    + log_su
                    + (theta - 1.0) * _safe_log(log_u_neg)
                    - _safe_log(u))

        # ------------------------------------------------------------------ #
        # Assemble log‑derivative                                        #
        # ------------------------------------------------------------------ #
        log_deriv = -h + log_hprime + log_dsdu

        #  underflow / overflow -> ±inf
        log_deriv = np.where(log_deriv < _FLOAT_MIN_LOG, -np.inf, log_deriv)
        log_deriv = np.where(log_deriv > _FLOAT_MAX_LOG, np.inf, log_deriv)

        return _safe_exp(log_deriv)

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """
        Compute the conditional CDF P(V ≤ v | U = u).

        Args:
            u (float or array-like): Conditioning value of U in (0,1).
            v (float or array-like): Value of V in (0,1).
            param (Sequence[float], optional): Copula parameters (d, q). Defaults to self.get_parameters().

        Returns:
            float or numpy.ndarray: Conditional CDF of V given U.
        """

        return self.partial_derivative_C_wrt_u(v, u, param)

    def IAD(self, data):
        """
        Return NaN for the IAD statistic, as it is disabled for this copula.

        Args:
            data (Sequence[array-like, array-like]): Ignored pseudo-observations.

        Returns:
            float: Always returns numpy.nan.
        """

        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        """
        Return NaN for the AD statistic, as it is disabled for this copula.

        Args:
            data (Sequence[array-like, array-like]): Ignored pseudo-observations.

        Returns:
            float: Always returns numpy.nan.
        """

        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan
