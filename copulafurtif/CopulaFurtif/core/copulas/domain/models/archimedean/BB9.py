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

    # ------------------------------------------------------------------
    # Cached GL-64 nodes for kendall_tau (class-level, computed at import)
    # ------------------------------------------------------------------
    _GL_N_TAU = 64
    _xi_tau, _wi_tau = __import__(
        'numpy.polynomial.legendre', fromlist=['leggauss']
    ).leggauss(_GL_N_TAU)
    _GL_HALF_TAU = 0.5 * (1.0 - 2e-6)
    _GL_MID_TAU  = 0.5 * (1.0 + 2e-6 - 1e-6)
    _GL_T_TAU    = _GL_HALF_TAU * _xi_tau + _GL_MID_TAU
    _GL_W_TAU    = _GL_HALF_TAU * _wi_tau

    def kendall_tau(self, param=None):
        """
        Kendall's τ via the Archimedean concordance formula.

        BB9 IS Archimedean (Joe 2014 §4.25.1) with generator:
            φ(t) = (δ⁻¹ − log t)^ϑ − δ⁻ϑ     (maps (0,1] to [0,∞))
            φ'(t) = −ϑ · (δ⁻¹ − log t)^{ϑ−1} / t   (< 0)

        τ = 1 + 4·∫₀¹ φ(t)/φ'(t) dt

        Evaluated on a vectorised 64-point Gauss-Legendre grid (~0.02ms).

        Parameters
        ----------
        param : [theta, delta], optional

        Returns
        -------
        float ∈ (0, 1)
        """
        if param is None:
            param = self.get_parameters()
        theta, delta = float(param[0]), float(param[1])
        delta_inv = 1.0 / delta

        T = self._GL_T_TAU
        W = self._GL_W_TAU

        x_t    = delta_inv - _safe_log(T)              # δ⁻¹ − log t > 0
        phi_t  = _safe_pow(x_t, theta) - delta_inv**theta
        phi_p  = -theta * _safe_pow(x_t, theta - 1.0) / T   # < 0

        result = float(np.dot(W, phi_t / phi_p))
        return float(np.clip(1.0 + 4.0 * result, 0.0, 1.0 - 1e-15))

    def blomqvist_beta(self, param=None):
        """
        Blomqvist's β = 4·C(½,½) − 1.

        Parameters
        ----------
        param : [theta, delta], optional

        Returns
        -------
        float
        """
        if param is None:
            param = self.get_parameters()
        return float(4.0 * self.get_cdf(0.5, 0.5, param) - 1.0)

    def sample(self, n, rng=None):
        """
        Draw n samples from BB9 via conditional inversion.

        For each u_i ~ U(0,1), draw w_i ~ U(0,1) and solve
            ∂C/∂u(u_i, v) = w_i  for v  using Brentq.

        Parameters
        ----------
        n   : int
        rng : np.random.Generator, optional

        Returns
        -------
        np.ndarray, shape (n, 2)
        """
        from scipy.optimize import brentq

        if rng is None:
            rng = np.random.default_rng()

        U = rng.uniform(1e-6, 1 - 1e-6, n)
        W_uni = rng.uniform(1e-6, 1 - 1e-6, n)
        V = np.empty(n)

        for i in range(n):
            u_i = float(U[i])
            w_i = float(W_uni[i])

            def obj(v_):
                return float(self.partial_derivative_C_wrt_u(u_i, v_)) - w_i

            try:
                V[i] = brentq(obj, 1e-8, 1 - 1e-8, xtol=1e-10, maxiter=100)
            except ValueError:
                V[i] = 1e-6 if abs(obj(1e-8)) < abs(obj(1 - 1e-8)) else 1 - 1e-6

        return np.column_stack([U, V])

    def init_from_data(self, u, v):
        """
        Moment-matching initialisation from pseudo-observations.

        Strategy
        --------
        1. Estimate ϑ and δ jointly by grid search on empirical Kendall τ
           (τ depends on both), using Blomqvist β̂ to pin δ first.

        Parameters
        ----------
        u, v : array-like, pseudo-observations in (0, 1)

        Returns
        -------
        np.ndarray [theta0, delta0]
        """
        from scipy.optimize import brentq
        from scipy.stats import kendalltau as sp_kendalltau

        u = np.asarray(u, float)
        v = np.asarray(v, float)

        # ---- 1. Estimate δ from Blomqvist β̂ (prior ϑ=2) ------------------
        # beta_emp = float(np.clip(4.0 * np.mean((u > 0.5) == (v > 0.5)) - 1.0, -0.99, 0.99))
        beta_emp = 2.0 * np.mean((u > 0.5) == (v > 0.5)) - 1.0 # THEO CHANGE
        delta0 = 1.0
        best_err = np.inf
        for de in np.linspace(0.1, 8.0, 25):
            try:
                beta_try = 4.0 * float(self.get_cdf(0.5, 0.5, [2.0, de])) - 1.0
                err = abs(beta_try - beta_emp)
                if err < best_err:
                    best_err = err; delta0 = de
            except Exception:
                pass

        # ---- 2. Estimate ϑ from τ̂ given δ₀ --------------------------------
        tau_emp_val, _ = sp_kendalltau(u, v)
        tau_emp_val = float(np.clip(tau_emp_val, 0.01, 0.99))

        theta_grid = [1.1, 1.5, 2.0, 3.0, 5.0, 8.0, 15.0]
        tau_grid = []
        for th in theta_grid:
            try:
                tau_grid.append((th, self.kendall_tau([th, delta0])))
            except Exception:
                tau_grid.append((th, np.nan))

        theta0 = theta_grid[0]
        for i in range(len(tau_grid) - 1):
            th_lo_br, t_lo_br = tau_grid[i]
            th_hi_br, t_hi_br = tau_grid[i + 1]
            if np.isfinite(t_lo_br) and np.isfinite(t_hi_br):
                if (t_lo_br - tau_emp_val) * (t_hi_br - tau_emp_val) <= 0:
                    try:
                        theta0 = brentq(
                            lambda th: self.kendall_tau([th, delta0]) - tau_emp_val,
                            th_lo_br, th_hi_br,
                            xtol=1e-4, rtol=1e-4, maxiter=30,
                        )
                    except Exception:
                        theta0 = 0.5 * (th_lo_br + th_hi_br)
                    break

        theta0 = float(np.clip(theta0, 1.001, 200.0))
        delta0 = float(np.clip(delta0, 1e-4, 300.0))
        return np.array([theta0, delta0], dtype=float)

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