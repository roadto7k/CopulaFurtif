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
    - Kendall's τ ∈ [0, 1), computed via Gauss–Legendre quadrature of
      τ = ∫₀¹ t(1−t)A″(t)/A(t) dt.

Structural relationship to Tawn Type-1:
    TawnT2 has the same Pickands functional form as TawnT1, but uses
    t = x/s (x = −ln u) instead of t = y/s (y = −ln v).  This swaps
    the roles of u and v:  C_T2(u,v; θ,β) = C_T1(v,u; θ,β).

References:
    - Joe (2014), §4.8.1.
    - Gudendorf & Segers (2010), "Extreme-value copulas".
    - Tadi & Witzany (2025), Table 2.

Attributes:
    name (str): Human-readable name.
    type (str): Identifier for the copula family.
    default_optim_method (str): Default optimiser for fitting.
"""

import numpy as np
from numpy import log as _np_log, exp as _np_exp
from scipy.optimize import brentq
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

    # Gauss–Legendre nodes for Kendall's tau quadrature
    _GL_N_TAU = 96
    _xi_tau, _wi_tau = np.polynomial.legendre.leggauss(_GL_N_TAU)
    _TAU_EPS = 1e-6
    _GL_HALF_TAU = 0.5 * (1.0 - 2.0 * _TAU_EPS)
    _GL_MID_TAU = 0.5
    _GL_T_TAU = _GL_HALF_TAU * _xi_tau + _GL_MID_TAU
    _GL_W_TAU = _GL_HALF_TAU * _wi_tau

    def __init__(self):
        super().__init__()
        self.name = "Tawn Type-2 Copula"
        self.type = "tawn2"
        self.default_optim_method = "Powell"
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
        Kendall's τ via Gauss–Legendre quadrature:

            τ = ∫₀¹ t(1−t)A″(t) / A(t) dt.
        """
        if param is None:
            param = self.get_parameters()

        t = self._GL_T_TAU
        w = self._GL_W_TAU
        A = np.clip(self._A(t, param), self._EPS, None)
        App = np.maximum(self._A_double(t, param), 0.0)

        tau = float(np.dot(w, t * (1.0 - t) * App / A))
        return float(np.clip(tau, 0.0, 1.0))

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
        Draw *n* samples via conditional inversion (Brent's method).

        For each u_i ~ U(0,1) and w_i ~ U(0,1), solve
            ∂C(u_i, v)/∂u = w_i  for v.
        """
        if param is None:
            param = self.get_parameters()
        if rng is None:
            rng = np.random.default_rng()

        n = int(n)
        u = rng.uniform(1e-6, 1.0 - 1e-6, n)
        w_uni = rng.uniform(1e-6, 1.0 - 1e-6, n)
        v = np.empty(n, dtype=float)

        for i in range(n):
            u_i, w_i = float(u[i]), float(w_uni[i])

            def obj(v_):
                return float(self.conditional_cdf_v_given_u(u_i, v_, param)) - w_i

            try:
                v[i] = brentq(obj, 1e-8, 1.0 - 1e-8, xtol=1e-10, maxiter=100)
            except ValueError:
                left = abs(obj(1e-8))
                right = abs(obj(1.0 - 1e-8))
                v[i] = 1e-6 if left <= right else 1.0 - 1e-6

        return np.column_stack([u, v])

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