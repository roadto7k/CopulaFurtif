"""
BB6 Copula implementation (Joe & Hu 1996).

The BB6 is a 2-parameter Archimedean copula with generator
    φ(t) = [−log(1 − (1−t)^θ)]^(1/δ),   θ ≥ 1, δ ≥ 1.

It nests:
  • Gumbel copula when θ = 1
  • Joe/B5 copula when δ = 1
  • Comonotone (C⁺) as θ → ∞ or δ → ∞

Properties:
  • Upper tail dependence:  λ_U = 2 − 2^(1/(θδ))
  • Lower tail dependence:  λ_L = 0  (lower tail order κ_L = 2^(1/δ) > 1)
  • Blomqvist β = 4β* − 1  where β* = 1 − [1 − (1−2^{−θ})^{(2−λ_U)}]^{1/θ}

References:
    Joe H. (1997) *Multivariate Models and Dependence Concepts*. § 4.22.
    Joe H., Hu T. (1996) Multivariate distributions from mixtures of max‑infinitely
    divisible distributions. *J. Multivariate Anal.* 57, 240–265.
"""
from __future__ import annotations

import numpy as np
from numpy import log as _np_log, exp as _np_exp
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import kendalltau

from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel, CopulaParameters
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence

# ---------------------------------------------------------------------------
# Floating‑point sentinels
# ---------------------------------------------------------------------------
_FLOAT_MIN = 1e-308
_FLOAT_MAX_LOG = 709.0   # exp(709) ≈ 8e307 < float64 max
_FLOAT_MIN_LOG = -745.0  # exp(-745) ≈ 5e-324 > 0


# ---------------------------------------------------------------------------
# Internal helpers — all accept / return numpy scalars or arrays
# ---------------------------------------------------------------------------

def _safe_log(x):
    """log clipped away from 0 to avoid −∞."""
    return _np_log(np.clip(x, _FLOAT_MIN, None))


def _safe_exp(log_x):
    """exp with exponent clipped so the result stays finite."""
    return _np_exp(np.clip(log_x, _FLOAT_MIN_LOG, _FLOAT_MAX_LOG))


def _safe_pow(base, exponent):
    """Stable b^e = exp(e·log b). base must be > 0."""
    return _safe_exp(exponent * _safe_log(base))


def _transform(u, theta):
    """
    x(u) = −log(1 − (1−u)^θ)   [Joe 1997, p. 201]

    Returns (x, ubar, log_ubar_theta, one_minus_ubar_theta) with:
      ubar              = 1 − u
      log_ubar_theta    = θ · log(ubar)   [= log( (1−u)^θ )]
      one_minus_ubar_theta = 1 − (1−u)^θ  [= 1 − exp(log_ubar_theta)]
    """
    ubar = 1.0 - u
    log_ubar_theta = theta * _safe_log(ubar)
    one_minus_ubar_theta = -np.expm1(log_ubar_theta)      # numerically stable 1−exp(·)
    x = -_safe_log(one_minus_ubar_theta)
    return x, ubar, log_ubar_theta, one_minus_ubar_theta


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class BB6Copula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """
    BB6 Copula (Joe & Hu 1996).

    Parameters
    ----------
    theta : float ≥ 1   (1 → Gumbel margin)
    delta : float ≥ 1   (1 → Joe/B5 margin)

    Bounds: θ ∈ [1, ∞),  δ ∈ [1, ∞).  Both lower bounds are *excluded* by the
    parent class validator so θ = 1 and δ = 1 are boundary / degenerate cases.
    """

    def __init__(self):
        super().__init__()
        self.name = "BB6 Copula"
        self.type = "bb6"
        self.default_optim_method = "Powell"
        self.init_parameters(
            CopulaParameters(
                np.array([2.0, 2.0]),
                [(1.0, np.inf), (1.0, np.inf)],
                ["theta", "delta"],
            )
        )

    # ------------------------------------------------------------------
    # Generator & its derivative (used in PDF)
    # ------------------------------------------------------------------

    def _phi(self, t, theta, delta):
        """φ(t) = (−log(1−(1−t)^θ))^(1/δ)."""
        return (-np.log(1.0 - (1.0 - t) ** theta)) ** (1.0 / delta)

    def _phi_prime(self, t, theta, delta):
        """φ'(t)  — analytical derivative of the generator."""
        g = 1.0 - (1.0 - t) ** theta
        gp = theta * (1.0 - t) ** (theta - 1.0)
        L = -np.log(g)
        Lp = -gp / g
        return (1.0 / delta) * L ** (1.0 / delta - 1.0) * Lp

    # ------------------------------------------------------------------
    # CDF
    # ------------------------------------------------------------------

    def get_cdf(self, u, v, param=None):
        """
        C(u, v) = 1 − (1 − exp{−[(x^δ + y^δ)^{1/δ}]})^{1/θ}

        with x = −log(1−(1−u)^θ),  y = −log(1−(1−v)^θ).
        """
        if param is None:
            param = self.get_parameters()
        theta, delta = float(param[0]), float(param[1])

        eps = 1e-12
        u = np.clip(u, eps, 1.0 - eps)
        v = np.clip(v, eps, 1.0 - eps)

        x, _, _, _ = _transform(u, theta)
        y, _, _, _ = _transform(v, theta)

        log_xd = delta * _safe_log(x)
        log_yd = delta * _safe_log(y)
        log_sum = np.logaddexp(log_xd, log_yd)          # log(x^δ + y^δ)
        log_tem = (1.0 / delta) * log_sum               # log( (x^δ+y^δ)^{1/δ} )

        w = _safe_exp(-_safe_exp(log_tem))               # exp(−tem)
        C = 1.0 - _safe_pow(1.0 - w, 1.0 / theta)
        return C

    # ------------------------------------------------------------------
    # PDF  — c(u,v) = ∂²C/∂u∂v
    # ------------------------------------------------------------------

    def get_pdf(self, u, v, param=None):
        """
        Copula density c(u, v) via Joe (1997) eq. (4.68) / p. 201:

            c(u,v) = [1−w]^{1/θ−2} · w · (x^δ+y^δ)^{1/δ−2}
                     · [(θ−1)(x^δ+y^δ)^{1/δ} + (δ−1)] · (1−w)  ← leading (1-w) term
                     · (xy)^{δ−1} · (1−ū^θ)^{−1} · (1−v̄^θ)^{−1} · (ū·v̄)^{θ−1}

        Using the fully expanded form from the reference (image p.201):

            c = [1−w]^{1/θ−2} · w · (x^δ+y^δ)^{1/δ−2}
                · [(θ−1)(x^δ+y^δ)^{1/δ} + (δ−1)(1−w)]
                · (xy)^{δ−1} · (1−ū^θ)^{−1} · (1−v̄^θ)^{−1} · (ū·v̄)^{θ−1}

        where ū = 1−u, v̄ = 1−v, w = exp{−(x^δ+y^δ)^{1/δ}}.
        """
        if param is None:
            param = self.get_parameters()
        theta, delta = float(param[0]), float(param[1])

        eps = 1e-12
        u = np.clip(u, eps, 1.0 - eps)
        v = np.clip(v, eps, 1.0 - eps)

        x, ubar, _, one_minus_u_pow = _transform(u, theta)   # one_minus_u_pow = 1−ū^θ
        y, vbar, _, one_minus_v_pow = _transform(v, theta)

        # --- core quantities in log‑space for numerical safety ---
        log_x = _safe_log(x)
        log_y = _safe_log(y)
        log_xd = delta * log_x
        log_yd = delta * log_y
        log_sum = np.logaddexp(log_xd, log_yd)               # log(x^δ+y^δ)
        S = _safe_exp(log_sum)                               # x^δ+y^δ  (scalar / array)
        log_tem = (1.0 / delta) * log_sum
        tem = _safe_exp(log_tem)                             # (x^δ+y^δ)^{1/δ}
        w = _safe_exp(-tem)                                  # exp(−tem)
        one_minus_w = np.clip(1.0 - w, _FLOAT_MIN, None)

        # --- bracket term (derived from ∂²C/∂u∂v, see Joe 1997 p.201):
        #     tem·(1 − w/θ) + (δ−1)·(1−w)
        # Note: NOT (θ−1)·tem — that was wrong. ---
        bracket = tem * (1.0 - w / theta) + (delta - 1.0) * one_minus_w

        # --- log of all positive multiplicative factors ---
        # Full formula: c = θ·(1−w)^{1/θ−2}·w·(xy)^{δ−1}·S^{1/δ−2}·bracket·(ūv̄)^{θ−1}/[(1−ū^θ)(1−v̄^θ)]
        log_pdf = (
            np.log(theta)                                     # θ  ← was missing
            + (1.0 / theta - 2.0) * _safe_log(one_minus_w)   # · (1−w)^{1/θ−2}
            + _safe_log(w)                                    # · w
            + (1.0 / delta - 2.0) * log_sum                  # · (x^δ+y^δ)^{1/δ−2}
            + _safe_log(np.abs(bracket))                      # · bracket
            + (delta - 1.0) * (log_x + log_y)                # · (xy)^{δ−1}
            + (theta - 1.0) * (_safe_log(ubar) + _safe_log(vbar))  # · (ū·v̄)^{θ−1}
            - _safe_log(one_minus_u_pow)                      # / (1−ū^θ)
            - _safe_log(one_minus_v_pow)                      # / (1−v̄^θ)
        )

        pdf = _safe_exp(log_pdf)
        return np.maximum(pdf, 0.0)

    # ------------------------------------------------------------------
    # Conditional CDF  h(v|u) = ∂C/∂u
    # ------------------------------------------------------------------

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """
        ∂C(u,v)/∂u  (= h-function h(v|u)).

        From Joe (1997) p.201:
            C_{2|1}(v|u) = [1−w]^{1/θ−1} · w · (x^δ+y^δ)^{1/δ−1} · x^{δ−1}
                           · e^x · [1−e^{−x}]^{−1/θ} · ∂x/∂u          (*)

        where ∂x/∂u = −θ·(1−u)^{θ−1} · [1−(1−u)^θ]^{−1}
                     = −θ·(1−u)^{θ−1} / (1−ū^θ).
        """
        if param is None:
            param = self.get_parameters()
        theta, delta = float(param[0]), float(param[1])

        eps = 1e-12
        u = np.clip(u, eps, 1.0 - eps)
        v = np.clip(v, eps, 1.0 - eps)

        x, ubar, _, one_minus_u_pow = _transform(u, theta)
        y, _, _, _ = _transform(v, theta)

        log_x = _safe_log(x)
        log_xd = delta * log_x
        log_yd = delta * _safe_log(y)
        log_sum = np.logaddexp(log_xd, log_yd)
        log_tem = (1.0 / delta) * log_sum
        tem = _safe_exp(log_tem)
        w = _safe_exp(-tem)
        one_minus_w = np.clip(1.0 - w, _FLOAT_MIN, None)

        # ∂tem/∂x = x^{δ−1} / (x^δ+y^δ)^{1−1/δ}  = exp((δ−1)log_x − (1−1/δ)log_sum)
        log_dtem_dx = (delta - 1.0) * log_x - (1.0 - 1.0 / delta) * log_sum
        dtem_dx = _safe_exp(log_dtem_dx)

        # ∂C/∂tem = −(1/θ)·(1−w)^{1/θ−1}·w  (from ∂C/∂w · ∂w/∂tem, ∂w/∂tem=−w)
        dC_dtem = -(1.0 / theta) * _safe_pow(one_minus_w, 1.0 / theta - 1.0) * w

        # ∂x/∂u = −θ·ū^{θ−1} / (1−ū^θ)
        dx_du = -theta * _safe_exp((theta - 1.0) * _safe_log(ubar)) / one_minus_u_pow

        h = dC_dtem * dtem_dx * dx_du
        return np.clip(h, 0.0, 1.0)

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """∂C(u,v)/∂v  by symmetry of BB6."""
        return self.partial_derivative_C_wrt_u(v, u, param)

    # ------------------------------------------------------------------
    # Tail dependence
    # ------------------------------------------------------------------

    def LTDC(self, param=None):
        """Lower tail dependence = 0 for BB6 (lower tail order κ_L = 2^{1/δ} > 1)."""
        return 0.0

    def UTDC(self, param=None):
        """
        Upper tail dependence:  λ_U = 2 − 2^{1/(θδ)}.

        (Joe 1997 p.201 states λ_U = 2 − 2^{1/(θδ)}; note some sources write
        2 − 2^{1/(δθ)} which is the same thing.)
        """
        if param is None:
            param = self.get_parameters()
        theta, delta = float(param[0]), float(param[1])
        return 2.0 - 2.0 ** (1.0 / (theta * delta))

    # ------------------------------------------------------------------
    # Blomqvist β
    # ------------------------------------------------------------------

    def blomqvist_beta(self, param=None) -> float:
        """
        β = 4·C(½, ½) − 1.

        From Joe (1997) p.201 closed form:
            β* = 1 − [1 − (1−2^{−θ})^{(2−λ_U)}]^{1/θ}
            β  = 4β* − 1

        We evaluate it directly via get_cdf for safety.
        """
        if param is None:
            param = self.get_parameters()
        return float(4.0 * self.get_cdf(0.5, 0.5, param) - 1.0)

    # ------------------------------------------------------------------
    # Kendall's τ  (numerical)
    # ------------------------------------------------------------------

    def kendall_tau(self, param=None):
        """
        Kendall's τ via the Archimedean generator formula.

        BB6 IS Archimedean with generator ψ(t) = x^δ where x = −log(1−(1−t)^θ).
        Verification: ψ⁻¹(S) = 1−(1−exp(−S^{1/δ}))^{1/θ} = C(u,v). ✓

        For any Archimedean copula:
            τ = 1 + 4 ∫₀¹ ψ(t)/ψ'(t) dt

        With:
            ψ(t)  = x^δ                                              (positive)
            ψ'(t) = δ · x^{δ−1} · (−θ · ū^{θ−1} / (1−ū^θ))        (negative)

        So ψ(t)/ψ'(t) = −x · (1−ū^θ) / (δ · θ · ū^{θ−1})          (negative)

        The integral is therefore negative, giving τ ∈ (0,1) for θ,δ ≥ 1.
        """
        if param is None:
            param = self.get_parameters()
        theta, delta = float(param[0]), float(param[1])

        def integrand(t):
            eps = 1e-14
            t = float(np.clip(t, eps, 1.0 - eps))
            ubar = 1.0 - t
            ubar_th = ubar ** theta                       # (1−t)^θ
            one_minus = 1.0 - ubar_th                    # 1−(1−t)^θ  ∈ (0,1)
            if one_minus <= 0.0:
                return 0.0
            x = -np.log(one_minus)                       # −log(1−(1−t)^θ) > 0

            # ψ(t)/ψ'(t) = −x·(1−ū^θ) / (δ·θ·ū^{θ−1})
            denom = delta * theta * (ubar ** (theta - 1.0))
            if denom < 1e-300:
                return 0.0
            return -x * one_minus / denom

        try:
            val, _ = quad(integrand, 0.0, 1.0,
                          limit=200, epsabs=1e-9, epsrel=1e-9)
        except Exception:
            return np.nan

        return float(1.0 + 4.0 * val)

    # ------------------------------------------------------------------
    # Sampling — conditional inversion
    # ------------------------------------------------------------------

    def sample(self, n: int, seed: int | None = None, param=None, rng=None):
        """
        Sample (u, v) from BB6 via conditional inversion.

        Algorithm:
          1. Draw U ~ Uniform(0,1)
          2. Draw W ~ Uniform(0,1)  (target: h(v|u) = W)
          3. Solve  ∂C/∂u(u, v) = W  for v  via Brentq on each pair.

        The h-function is monotone increasing in v ∈ (0,1), so Brentq is safe.
        """
        if param is None:
            param = self.get_parameters()

        if rng is None:
            rng = np.random.default_rng(seed)

        u_samples = rng.random(n)
        w_samples = rng.random(n)

        eps = 1e-9
        v_samples = np.empty(n)

        for i in range(n):
            ui = float(np.clip(u_samples[i], eps, 1.0 - eps))
            wi = float(np.clip(w_samples[i], eps, 1.0 - eps))

            def objective(v_val):
                return float(self.partial_derivative_C_wrt_u(ui, v_val, param)) - wi

            try:
                # h is in [0,1]; bracket full unit interval
                v_lo, v_hi = eps, 1.0 - eps
                f_lo = objective(v_lo)
                f_hi = objective(v_hi)
                if f_lo * f_hi > 0:
                    # fallback: use w as crude estimate
                    v_samples[i] = wi
                else:
                    v_samples[i] = brentq(objective, v_lo, v_hi,
                                          xtol=1e-10, rtol=1e-10, maxiter=200)
            except Exception:
                v_samples[i] = wi  # graceful degradation

        u_out = np.clip(u_samples, eps, 1.0 - eps)
        v_out = np.clip(v_samples, eps, 1.0 - eps)
        return np.column_stack([u_out, v_out])

    # ------------------------------------------------------------------
    # Initialisation from data
    # ------------------------------------------------------------------

    def init_from_data(self, u, v):
        """
        Moment-based initialisation of (θ, δ) from pseudo-observations.

        Strategy
        --------
        1. Compute empirical Kendall's τ̂ and Blomqvist β̂.
        2. Search a 2D grid of (θ, δ) pairs and find the pair minimising
           |τ(θ,δ) − τ̂|.
        3. Refine by fixing θ and solving for δ via Brentq.
        4. Clip to parameter bounds.
        """
        u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)

        tau_emp, _ = kendalltau(u, v)
        tau_emp = float(np.clip(tau_emp, 0.02, 0.98))  # BB6 has τ > 0

        # 2D grid search
        theta_grid = [1.2, 1.5, 2.0, 3.0, 5.0, 8.0]
        delta_grid = [1.2, 1.5, 2.0, 3.0, 5.0, 8.0]

        best = [2.0, 2.0]
        best_err = np.inf

        for th in theta_grid:
            for de in delta_grid:
                try:
                    tau_try = self.kendall_tau([th, de])
                    if np.isnan(tau_try):
                        continue
                    err = abs(tau_try - tau_emp)
                    if err < best_err:
                        best_err = err
                        best = [th, de]
                except Exception:
                    continue

        # Refine: fix theta from grid, solve for delta via Brentq
        theta_fixed = best[0]
        try:
            def obj_delta(d):
                tau_try = self.kendall_tau([theta_fixed, d])
                return tau_try - tau_emp if not np.isnan(tau_try) else 1.0

            f_lo = obj_delta(1.01)
            f_hi = obj_delta(30.0)
            if np.isfinite(f_lo) and np.isfinite(f_hi) and f_lo * f_hi < 0:
                delta_opt = brentq(obj_delta, 1.01, 30.0,
                                   xtol=1e-5, rtol=1e-5, maxiter=100)
                best = [theta_fixed, delta_opt]
        except Exception:
            pass

        lo_t, hi_t = self.get_bounds()[0]
        lo_d, hi_d = self.get_bounds()[1]
        theta_final = float(np.clip(best[0], lo_t + 1e-6, min(hi_t, 35.0)))
        delta_final = float(np.clip(best[1], lo_d + 1e-6, min(hi_d, 35.0)))

        return np.array([theta_final, delta_final])

    # ------------------------------------------------------------------
    # Disabled statistics
    # ------------------------------------------------------------------

    def IAD(self, data):
        """IAD disabled for BB6."""
        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        """AD disabled for BB6."""
        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan