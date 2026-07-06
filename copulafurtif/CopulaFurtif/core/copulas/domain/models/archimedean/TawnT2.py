"""
Tawn Type-2 extreme-value copula.

The Tawn Type-2 copula is obtained from the three-parameter Tawn copula by
fixing α = 1.  It is an *asymmetric* extreme-value copula parameterised by
θ ≥ 1 (shape) and β ∈ [0, 1] (asymmetry weight).

Pickands dependence function (with t = x/(x+y), x = −ln u, y = −ln v):

    A(t) = (1 − β)(1 − t) + [t^θ + (β(1 − t))^θ]^{1/θ}

Properties:
    - A(0) = A(1) = 1  (proper uniform marginals).
    - β = 0  ⇒  A(t) ≡ 1  ⇒  independence copula C⊥.
    - β = 1  ⇒  Gumbel copula with parameter θ.
    - θ = 1, any β  ⇒  independence copula.
    - Lower-tail dependence:  λ_L = 0 always.
    - Upper-tail dependence:  λ_U = 2 − 2·A(½) > 0 when β > 0 and θ > 1.
    - C(u, v) ≠ C(v, u) in general (asymmetric).
    - Kendall's tau belongs to [0, 1) and is evaluated numerically using
      the bounded conditional-CDF identity

          tau = 1 - 4 * integral integral
                C_{2|1}(v|u) * C_{1|2}(u|v) du dv.

      This representation avoids the numerical concentration of A''(t)
      that occurs for large theta.

Structural relationship to Tawn Type-1:
    TawnT2 has the same Pickands functional form as TawnT1, but uses
    t = x/s (x = −ln u) instead of t = y/s (y = −ln v).  This swaps
    the roles of u and v:  C_T2(u,v; θ,β) = C_T1(v,u; θ,β).

References:
    - Joe (2014), §4.15.1.
    - Gudendorf & Segers (2010), "Extreme-value copulas".
    - Tadi & Witzany (2025), Table 2.

Attributes:
    name (str): Human-readable name.
    type (str): Identifier for the copula family.
    default_optim_method (str): Default optimiser for fitting.
"""

import numpy as np
from numpy import log as _np_log, exp as _np_exp
# from scipy.optimize import brentq
from scipy.stats import kendalltau as sp_kendalltau

from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel, CopulaParameters
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


# ── IEEE-754 guards ──────────────────────────────────────────────────────────
_FLOAT_MIN = 1e-308
_FLOAT_MAX_LOG = 709.0
_FLOAT_MIN_LOG = -745.0


def _safe_log(x):
    """Natural log with clipping away from zero."""
    return _np_log(np.clip(x, _FLOAT_MIN, None))


def _safe_exp(log_x):
    """Exponentiation with clipping inside the float64 range."""
    return _np_exp(np.clip(log_x, _FLOAT_MIN_LOG, _FLOAT_MAX_LOG))


# ── Copula ───────────────────────────────────────────────────────────────────

class TawnT2Copula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    r"""
    Tawn Type-2 extreme-value copula.

    Pickands dependence function (with :math:`t = x/(x+y)`):

    .. math::

        A(t) = (1-\beta)(1-t)
               + \bigl[t^{\theta} + (\beta(1-t))^{\theta}\bigr]^{1/\theta},

    with :math:`\theta \ge 1` and :math:`\beta \in [0,1]`.

    Special cases (limits, not settable at exact boundary):
        - β → 0 → independence.
        - β → 1 → Gumbel(θ).
        - θ → 1 → independence (for any β).
    """

    _EPS = 1e-12

    # Gauss-Legendre quadrature for the bounded Kendall's tau identity
    _GL_N_TAU = 64
    _xi_tau, _wi_tau = np.polynomial.legendre.leggauss(_GL_N_TAU)

    _GL_U_TAU = 0.5 * (_xi_tau + 1.0)
    _GL_W_TAU = 0.5 * _wi_tau

    _GL_UU_TAU, _GL_VV_TAU = np.meshgrid(
        _GL_U_TAU,
        _GL_U_TAU,
        indexing="ij",
    )

    _GL_W2D_TAU = np.outer(_GL_W_TAU, _GL_W_TAU)

    def __init__(self):
        super().__init__()
        self.name = "Tawn Type-2 Copula"
        self.type = "tawn2"
        self.default_optim_method = "SLSQP"
        self.init_parameters(
            CopulaParameters(
                np.array([2.0, 0.5], dtype=float),
                [(1.0, 500.0), (0.0, 1.0)],
                ["theta", "beta"],
            )
        )

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _clip_open_unit(x, eps=_EPS):
        return np.clip(np.asarray(x, dtype=float), eps, 1.0 - eps)

    # ── Pickands function & derivatives (vectorised) ──────────────────────

    def _A(self, t, param=None):
        r"""Pickands dependence function A(t) on [0, 1], with t = x/s."""
        if param is None:
            param = self.get_parameters()
        theta, beta = float(param[0]), float(param[1])

        t = np.asarray(t, dtype=float)
        omt = np.clip(1.0 - t, 0.0, 1.0)
        h = np.power(np.clip(t, 0.0, 1.0), theta) + np.power(beta * omt, theta)
        # h ≥ 0 by construction; do NOT clip to _EPS — it corrupts A(t) for
        # small β and large θ where h is legitimately ≪ 1e-12.
        return (1.0 - beta) * omt + np.power(np.maximum(h, 0.0), 1.0 / theta)

    def _A_prime(self, t, param=None):
        """First derivative A'(t) on (0, 1)."""
        if param is None:
            param = self.get_parameters()
        theta, beta = float(param[0]), float(param[1])

        t = np.asarray(t, dtype=float)
        t_cl = np.clip(t, self._EPS, 1.0 - self._EPS)
        omt = np.clip(1.0 - t_cl, self._EPS, 1.0)

        h = np.power(t_cl, theta) + np.power(beta * omt, theta)
        hp = (
            theta * np.power(t_cl, theta - 1.0)
            - theta * (beta ** theta) * np.power(omt, theta - 1.0)
        )
        h_safe = np.maximum(h, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            factor = np.where(
                h_safe > 0.0,
                (1.0 / theta) * np.power(h_safe, 1.0 / theta - 1.0),
                0.0,
            )
        return -(1.0 - beta) + factor * hp

    def _A_double(self, t, param=None):
        """Second derivative A''(t) on (0, 1)."""
        if param is None:
            param = self.get_parameters()
        theta, beta = float(param[0]), float(param[1])

        t = np.asarray(t, dtype=float)
        t_cl = np.clip(t, self._EPS, 1.0 - self._EPS)
        omt = np.clip(1.0 - t_cl, self._EPS, 1.0)

        h = np.power(t_cl, theta) + np.power(beta * omt, theta)
        hp = (
            theta * np.power(t_cl, theta - 1.0)
            - theta * (beta ** theta) * np.power(omt, theta - 1.0)
        )
        hpp = (
            theta * (theta - 1.0) * np.power(t_cl, theta - 2.0)
            + theta * (theta - 1.0) * (beta ** theta) * np.power(omt, theta - 2.0)
        )

        h_safe = np.maximum(h, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            term1 = np.where(
                h_safe > 0.0,
                (1.0 / theta)
                * (1.0 / theta - 1.0)
                * np.power(h_safe, 1.0 / theta - 2.0)
                * hp ** 2,
                0.0,
            )
            term2 = np.where(
                h_safe > 0.0,
                (1.0 / theta) * np.power(h_safe, 1.0 / theta - 1.0) * hpp,
                0.0,
            )
        return term1 + term2

    # ── CDF ───────────────────────────────────────────────────────────────

    def get_cdf(self, u, v, param=None):
        r"""CDF: C(u,v) = exp{−s·A(t)},  s = x+y, t = x/s, x = −ln u, y = −ln v."""
        if param is None:
            param = self.get_parameters()

        u = self._clip_open_unit(u)
        v = self._clip_open_unit(v)

        x = -_safe_log(u)
        y = -_safe_log(v)
        s = x + y
        t = x / s

        return _safe_exp(-s * self._A(t, param))

    # ── PDF ───────────────────────────────────────────────────────────────

    def get_pdf(self, u, v, param=None):
        r"""
        Copula density  c(u,v) = ∂²C/(∂u ∂v).

        With t = x/s (x = −ln u, y = −ln v, s = x+y):

            ℓ_x = A + (y/s)·A'
            ℓ_y = A − (x/s)·A'
            ℓ_{xy} = −(xy/s³)·A''

            c(u,v) = C(u,v)/(u·v) · (ℓ_x·ℓ_y − ℓ_{xy})
        """
        if param is None:
            param = self.get_parameters()

        u = self._clip_open_unit(u)
        v = self._clip_open_unit(v)

        x = -_safe_log(u)
        y = -_safe_log(v)
        s = x + y
        t = x / s

        A = self._A(t, param)
        Ap = self._A_prime(t, param)
        App = self._A_double(t, param)
        c_val = self.get_cdf(u, v, param)

        ell_x = A + (y / s) * Ap
        ell_y = A - (x / s) * Ap
        ell_xy = -(x * y / s ** 3) * App

        return np.maximum(c_val * (ell_x * ell_y - ell_xy) / (u * v), 0.0)

    # ── Partial derivatives (h-functions) ─────────────────────────────────

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        r"""∂C(u,v)/∂u.  With t = x/s:  = C/u · [A + (y/s)·A']."""
        if param is None:
            param = self.get_parameters()

        u = self._clip_open_unit(u)
        v = self._clip_open_unit(v)

        x = -_safe_log(u)
        y = -_safe_log(v)
        s = x + y
        t = x / s

        c_val = self.get_cdf(u, v, param)
        A = self._A(t, param)
        Ap = self._A_prime(t, param)

        out = c_val * (A + (y / s) * Ap) / u
        return np.clip(out, 0.0, 1.0)

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        r"""∂C(u,v)/∂v.  With t = x/s:  = C/v · [A − (x/s)·A']."""
        if param is None:
            param = self.get_parameters()

        u = self._clip_open_unit(u)
        v = self._clip_open_unit(v)

        x = -_safe_log(u)
        y = -_safe_log(v)
        s = x + y
        t = x / s

        c_val = self.get_cdf(u, v, param)
        A = self._A(t, param)
        Ap = self._A_prime(t, param)

        out = c_val * (A - (x / s) * Ap) / v
        return np.clip(out, 0.0, 1.0)

    def conditional_cdf_v_given_u(self, u, v, param=None):
        r"""P(V ≤ v | U = u)."""
        return self.partial_derivative_C_wrt_u(u, v, param)

    def conditional_cdf_u_given_v(self, u, v, param=None):
        r"""P(U ≤ u | V = v)."""
        return self.partial_derivative_C_wrt_v(u, v, param)

    # ── Dependence measures ───────────────────────────────────────────────

    def kendall_tau(self, param=None):
        r"""
        Compute Kendall's tau using the bounded conditional-CDF identity.

        For a bivariate copula C,

            tau = 1 - 4 * integral_0^1 integral_0^1
                  C_{2|1}(v|u) * C_{1|2}(u|v) du dv

        where

            C_{2|1}(v|u) = dC(u,v) / du
            C_{1|2}(u|v) = dC(u,v) / dv

        This representation is numerically preferable to the extreme-value
        identity involving A''(t) when theta is large. In that regime, the
        curvature of the Pickands dependence function becomes increasingly
        concentrated around its kink, and a fixed one-dimensional quadrature
        may fail to capture the resulting narrow peak.

        The conditional CDFs remain bounded in [0, 1], giving a more stable
        numerical integrand across the full admissible parameter domain.

        Parameters
        ----------
        param : array-like, optional
            Copula parameters [theta, beta]. If omitted, the current model
            parameters are used.

        Returns
        -------
        float
            Kendall's tau implied by the copula parameters.
        """
        if param is None:
            param = self.get_parameters()

        U = self._GL_UU_TAU.ravel()
        V = self._GL_VV_TAU.ravel()

        partial_u = np.asarray(
            self.partial_derivative_C_wrt_u(U, V, param),
            dtype=float,
        ).reshape(self._GL_N_TAU, self._GL_N_TAU)

        partial_v = np.asarray(
            self.partial_derivative_C_wrt_v(U, V, param),
            dtype=float,
        ).reshape(self._GL_N_TAU, self._GL_N_TAU)

        integral = float(
            np.sum(
                self._GL_W2D_TAU
                * partial_u
                * partial_v
            )
        )

        return float(1.0 - 4.0 * integral)

    def blomqvist_beta(self, param=None):
        r"""Blomqvist's β = 4·C(½, ½) − 1."""
        if param is None:
            param = self.get_parameters()
        return float(4.0 * self.get_cdf(0.5, 0.5, param) - 1.0)

    # ── Tail dependence ───────────────────────────────────────────────────

    def LTDC(self, param=None):
        """Lower-tail dependence coefficient — identically zero."""
        return 0.0

    def UTDC(self, param=None):
        r"""Upper-tail dependence:  λ_U = 2 − 2·A(½)."""
        if param is None:
            param = self.get_parameters()
        return float(np.clip(2.0 - 2.0 * self._A(0.5, param), 0.0, 1.0))

    # ── Sampling ──────────────────────────────────────────────────────────

    def sample(self, n, param=None, rng=None):
        r"""
        Draw samples by vectorized conditional inversion.

        Let

            h(v | u) = P(V <= v | U = u)
                     = dC(u, v) / du.

        For independent draws

            U ~ Uniform(0, 1),
            W ~ Uniform(0, 1),

        the second copula coordinate is obtained from

            V = h^{-1}(W | U).

        The inverse conditional CDF is computed by vectorized bisection.
        All observations are updated simultaneously with NumPy arrays, avoiding
        one scalar root solver per sample.

        Because h(v | u) is non-decreasing in v, bisection converges to the
        conditional quantile. After ``n_iter`` iterations, the remaining
        interval width is approximately 2**(-n_iter).

        Parameters
        ----------
        n : int
            Number of samples.

        param : array-like, optional
            Tawn Type-2 parameters [theta, beta]. If omitted, the current
            model parameters are used.

        rng : numpy.random.Generator, optional
            NumPy random number generator.

        Returns
        -------
        numpy.ndarray
            Array of shape (n, 2) containing the sampled copula coordinates.
        """
        if param is None:
            param = self.get_parameters()

        if rng is None:
            rng = np.random.default_rng()

        n = int(n)

        if n < 0:
            raise ValueError("n must be non-negative.")

        if n == 0:
            return np.empty((0, 2), dtype=float)

        eps = 1e-8
        n_iter = 40

        u = rng.uniform(
            eps,
            1.0 - eps,
            size=n,
        )

        w = rng.uniform(
            eps,
            1.0 - eps,
            size=n,
        )

        lo = np.full(
            n,
            eps,
            dtype=float,
        )

        hi = np.full(
            n,
            1.0 - eps,
            dtype=float,
        )

        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)

            h_mid = np.asarray(
                self.conditional_cdf_v_given_u(
                    u,
                    mid,
                    param,
                ),
                dtype=float,
            )

            go_left = h_mid >= w

            hi = np.where(
                go_left,
                mid,
                hi,
            )

            lo = np.where(
                go_left,
                lo,
                mid,
            )

        v = 0.5 * (lo + hi)

        return np.column_stack(
            [
                u,
                v,
            ]
        )

    # ── init_from_data ────────────────────────────────────────────────────

    def init_from_data(self, u, v):
        """
        Moment-based initialisation from pseudo-observations.

        Two-stage grid search matching empirical Kendall's τ and Blomqvist's β,
        then local refinement.
        """
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)

        tau_emp, _ = sp_kendalltau(u, v)
        tau_emp = 0.0 if np.isnan(tau_emp) else float(np.clip(tau_emp, 0.0, 0.995))
        beta_emp = float(
            np.clip(2.0 * np.mean((u > 0.5) == (v > 0.5)) - 1.0, 0.0, 0.995)
        )

        if tau_emp < 0.03 and beta_emp < 0.03:
            return np.array([2.0, 0.0], dtype=float)

        theta_grid = np.array([1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0])
        beta_grid = np.linspace(0.0, 1.0, 41)

        best_score = np.inf
        best = np.array([2.0, 0.5], dtype=float)

        def score(theta, beta):
            tau_th = self.kendall_tau([theta, beta])
            beta_th = self.blomqvist_beta([theta, beta])
            return ((tau_th - tau_emp) / 0.05) ** 2 + (
                (beta_th - beta_emp) / 0.05
            ) ** 2

        for theta in theta_grid:
            for beta in beta_grid:
                s = score(float(theta), float(beta))
                if s < best_score:
                    best_score = s
                    best = np.array([float(theta), float(beta)], dtype=float)

        theta0, beta0 = float(best[0]), float(best[1])
        theta_refined = np.linspace(max(1.0, 0.7 * theta0), 1.4 * theta0 + 0.05, 25)
        beta_refined = np.linspace(
            max(0.0, beta0 - 0.15), min(1.0, beta0 + 0.15), 31
        )

        for theta in theta_refined:
            for beta in beta_refined:
                s = score(float(theta), float(beta))
                if s < best_score:
                    best_score = s
                    best = np.array([float(theta), float(beta)], dtype=float)

        best[0] = np.clip(best[0], 1.0, 500.0)
        best[1] = np.clip(best[1], 0.0, 1.0)
        return best.astype(float)

    # ── Diagnostics (disabled) ────────────────────────────────────────────

    def IAD(self, data):
        """Return NaN — IAD is disabled for Tawn Type-2."""
        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        """Return NaN — AD is disabled for Tawn Type-2."""
        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan