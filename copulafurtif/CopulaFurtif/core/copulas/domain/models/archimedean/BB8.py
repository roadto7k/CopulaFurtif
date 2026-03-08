import numpy as np
from numpy import log as _np_log, exp as _np_exp

from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel, CopulaParameters
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence

_FLOAT_MIN = 1e-308
_FLOAT_MAX_LOG = 709.0
_FLOAT_MIN_LOG = -745.0


def _safe_log(x):
    return _np_log(np.clip(x, _FLOAT_MIN, None))


def _safe_exp(log_x):
    return _np_exp(np.clip(log_x, _FLOAT_MIN_LOG, _FLOAT_MAX_LOG))


def _safe_pow(base, exponent):
    return _safe_exp(exponent * _safe_log(base))


class BB8Copula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """
    BB8 Copula (Joe 1993 / Joe 2014 §4.24.1).

    Two-parameter Archimedean copula based on a two-parameter LT family:

        C(u,v; ϑ,δ) = δ⁻¹(1 − {1 − η⁻¹·x·y}^{1/ϑ})

    where:
        η  = 1 − (1−δ)^ϑ
        x  = 1 − (1−δu)^ϑ
        y  = 1 − (1−δv)^ϑ
        ϑ ≥ 1,  0 < δ ≤ 1.

    Properties:
        • C⊥ as δ→0⁺ or ϑ→1⁺ (in appropriate limits).
        • Frank family as ϑ→∞ with η held constant.
        • No tail dependence for 0 < δ < 1 (λ_L = λ_U = 0).
        • Concordance increases with both ϑ and δ.
        • Archimedean generator: φ(t) = −log{[1−(1−δt)^ϑ]/η}.
    """

    # ------------------------------------------------------------------
    # Cached GL-64 nodes for kendall_tau (class-level, computed at import)
    # ------------------------------------------------------------------
    _GL_N_TAU = 64
    _xi_tau, _wi_tau = __import__(
        'numpy.polynomial.legendre', fromlist=['leggauss']
    ).leggauss(_GL_N_TAU)
    _GL_HALF_TAU = 0.5 * (1.0 - 2e-6)
    _GL_MID_TAU  = 0.5 * (1.0 + 2e-6 - 1e-6)
    _GL_T_TAU    = _GL_HALF_TAU * _xi_tau + _GL_MID_TAU   # shape (64,)
    _GL_W_TAU    = _GL_HALF_TAU * _wi_tau

    def __init__(self):
        super().__init__()
        self.name = "BB8 Copula"
        self.type = "bb8"
        self.default_optim_method = "Powell"
        self.init_parameters(CopulaParameters(
            np.array([2.0, 0.7]),
            [(1.0, np.inf), (1e-8, 1.0)],
            ["theta", "delta"],
        ))

    # ------------------------------------------------------------------
    # Core intermediates (shared across methods)
    # ------------------------------------------------------------------

    @staticmethod
    def _intermediates(u, v, theta, delta):
        """
        Return (η, x, y, w) for the BB8 copula.

        η  = 1 − (1−δ)^ϑ
        x  = 1 − (1−δu)^ϑ
        y  = 1 − (1−δv)^ϑ
        w  = 1 − η⁻¹·x·y   (clipped to [1e-300, 1])
        """
        eta = 1.0 - (1.0 - delta) ** theta
        x = 1.0 - _safe_exp(theta * _safe_log(1.0 - delta * u))
        y = 1.0 - _safe_exp(theta * _safe_log(1.0 - delta * v))
        w = np.clip(1.0 - x * y / eta, 1e-300, 1.0)
        return eta, x, y, w

    def get_cdf(self, u, v, param=None):
        """
        Evaluate C(u,v; ϑ,δ) = δ⁻¹(1 − {1 − η⁻¹·x·y}^{1/ϑ}).

        Args:
            u, v (float or array-like): Uniform margins in (0,1).
            param ([theta, delta], optional): Parameters. Default: self.get_parameters().

        Returns:
            float or np.ndarray
        """
        if param is None:
            param = self.get_parameters()
        theta, delta = float(param[0]), float(param[1])
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        eta, x, y, w = self._intermediates(u, v, theta, delta)
        return (1.0 - _safe_pow(w, 1.0 / theta)) / delta

    def get_pdf(self, u, v, param=None):
        """
        Copula density (Joe 2014, §4.24.1):

            c(u,v) = η⁻¹δ · (1−η⁻¹xy)^{1/ϑ−2} · (ϑ − η⁻¹xy)
                     · (1−δu)^{ϑ−1} · (1−δv)^{ϑ−1}

        Args:
            u, v (float or array-like): Uniform margins in (0,1).
            param ([theta, delta], optional): Parameters.

        Returns:
            float or np.ndarray (≥ 0)
        """
        if param is None:
            param = self.get_parameters()
        theta, delta = float(param[0]), float(param[1])
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        eta, x, y, w = self._intermediates(u, v, theta, delta)
        xy_over_eta = x * y / eta

        # log-space for numerical stability
        log_pdf = (
            -_safe_log(eta)
            + _safe_log(delta)
            + (1.0 / theta - 2.0) * _safe_log(w)
            + _safe_log(np.clip(theta - xy_over_eta, 1e-300, None))
            + (theta - 1.0) * _safe_log(1.0 - delta * u)
            + (theta - 1.0) * _safe_log(1.0 - delta * v)
        )
        return np.maximum(_safe_exp(log_pdf), 0.0)

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """
        ∂C/∂u = η⁻¹ · y · (1−δu)^{ϑ−1} · (1−η⁻¹xy)^{1/ϑ−1}

        Args:
            u, v (float or array-like): Uniform margins in (0,1).
            param ([theta, delta], optional): Parameters.

        Returns:
            float or np.ndarray
        """
        if param is None:
            param = self.get_parameters()
        theta, delta = float(param[0]), float(param[1])
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        eta, x, y, w = self._intermediates(u, v, theta, delta)
        return (y / eta) * _safe_pow(1.0 - delta * u, theta - 1.0) * _safe_pow(w, 1.0 / theta - 1.0)

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """∂C/∂v — by symmetry, equals ∂C/∂u with u and v swapped."""
        return self.partial_derivative_C_wrt_u(v, u, param)

    # ------------------------------------------------------------------
    # Kendall's tau
    # ------------------------------------------------------------------

    def kendall_tau(self, param=None):
        """
        Kendall's τ via the Archimedean concordance formula.

        BB8 IS Archimedean with generator:
            φ(t) = −log{[1−(1−δt)^ϑ]/η}   (maps [0,1] to [0,∞), φ(1)=0)
            φ'(t) = −ϑδ(1−δt)^{ϑ−1} / [1−(1−δt)^ϑ]  < 0

        τ = 1 + 4·∫₀¹ φ(t)/φ'(t) dt

        Evaluated on a vectorised 64-point Gauss-Legendre grid (~0.03ms).

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

        T  = self._GL_T_TAU
        W  = self._GL_W_TAU

        eta = 1.0 - (1.0 - delta) ** theta
        # φ(t) = −log([1−(1−δt)^ϑ] / η)
        x_t   = 1.0 - _safe_pow(1.0 - delta * T, theta)
        phi_t = -_safe_log(x_t / eta)                   # > 0 on (0,1)
        # φ'(t) = −ϑδ(1−δt)^{ϑ−1}/[1−(1−δt)^ϑ]       (< 0)
        phi_prime = -theta * delta * _safe_pow(1.0 - delta * T, theta - 1.0) / x_t

        integrand = phi_t / phi_prime   # < 0

        result = float(np.dot(W, integrand))
        return float(np.clip(1.0 + 4.0 * result, 0.0, 1.0 - 1e-15))

    # ------------------------------------------------------------------
    # Blomqvist's beta
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, n, rng=None):
        """
        Draw n samples via conditional inversion.

        For each u_i ~ U(0,1), draw w_i ~ U(0,1) and solve
            ∂C/∂u(u_i, v) = w_i  for v ∈ (0,1)
        using Brent's method.

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
        W = rng.uniform(1e-6, 1 - 1e-6, n)
        V = np.empty(n)

        for i in range(n):
            u_i = float(U[i])
            w_i = float(W[i])

            def obj(v_):
                return float(self.partial_derivative_C_wrt_u(u_i, v_)) - w_i

            try:
                V[i] = brentq(obj, 1e-8, 1 - 1e-8, xtol=1e-10, maxiter=100)
            except ValueError:
                V[i] = 1e-6 if abs(obj(1e-8)) < abs(obj(1 - 1e-8)) else 1 - 1e-6

        return np.column_stack([U, V])

    # ------------------------------------------------------------------
    # init_from_data
    # ------------------------------------------------------------------

    def init_from_data(self, u, v):
        """
        Moment-matching initialisation from pseudo-observations.

        Strategy
        --------
        1. Estimate δ from empirical Blomqvist β̂ (grid search).
        2. Estimate ϑ from empirical Kendall τ̂ via grid search + Brentq.

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

        # ---- 1. Estimate δ from Blomqvist β ---------------------------
        beta_emp = float(np.clip(4.0 * np.mean((u > 0.5) == (v > 0.5)) - 1.0, -0.99, 0.99))
        # Grid over δ at θ=2 (prior)
        delta0 = 0.5  # safe default
        best_err = np.inf
        for de in np.linspace(0.05, 0.99, 20):
            try:
                beta_try = 4.0 * self.get_cdf(0.5, 0.5, [2.0, de]) - 1.0
                err = abs(beta_try - beta_emp)
                if err < best_err:
                    best_err = err; delta0 = de
            except Exception:
                pass

        # ---- 2. Estimate ϑ from τ̂ ------------------------------------
        tau_emp_val, _ = sp_kendalltau(u, v)
        tau_emp_val = float(np.clip(tau_emp_val, 0.01, 0.99))

        theta_grid = [1.05, 1.3, 1.7, 2.5, 4.0, 7.0, 15.0]
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
        delta0 = float(np.clip(delta0, 1e-4, 1.0 - 1e-8))
        return np.array([theta0, delta0], dtype=float)

    # ------------------------------------------------------------------
    # Tail dependence
    # ------------------------------------------------------------------

    def LTDC(self, param=None):
        """λ_L = 0 for 0 < δ < 1 (Joe 2014, §4.24.1)."""
        return 0.0

    def UTDC(self, param=None):
        """λ_U = 0 for 0 < δ < 1 (Joe 2014, §4.24.1)."""
        return 0.0

    # ------------------------------------------------------------------
    # Disabled statistics
    # ------------------------------------------------------------------

    def IAD(self, data):
        """IAD disabled for BB8."""
        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        """AD disabled for BB8."""
        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan