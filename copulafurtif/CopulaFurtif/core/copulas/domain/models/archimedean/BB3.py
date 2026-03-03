import math
from typing import Optional

import numpy as np
import scipy.stats as stx
from numpy.random import default_rng
from scipy.optimize import minimize

from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel, CopulaParameters
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


_MAX_LOG_EXP = 700.0  # safe exp bound (exp(700) ~ 1e304)


def _safe_exp(x: np.ndarray) -> np.ndarray:
    return np.exp(np.minimum(x, _MAX_LOG_EXP))


def _pow_pos(x: np.ndarray, p: float) -> np.ndarray:
    """Compute x**p for x>0 safely (log-domain)."""
    x = np.asarray(x, float)
    return np.exp(p * np.log(np.maximum(x, 1e-300)))


def _logS_and_weights(A: np.ndarray, B: np.ndarray):
    """
    Compute:
      logS = log(exp(A) + exp(B) - 1)
      w_u  = exp(A) / (exp(A)+exp(B)-1)
      w_v  = exp(B) / (exp(A)+exp(B)-1)
    stably, using the max-trick (no catastrophic cancellation).
    Assumes A,B >= 0 (true for BB3).
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)

    M = np.maximum(A, B)
    eA = np.exp(A - M)          # in (0,1]
    eB = np.exp(B - M)
    em = np.exp(-M)             # may underflow to 0 (fine)

    denom = eA + eB - em
    denom = np.maximum(denom, 1e-300)

    logS = M + np.log(denom)
    w_u = eA / denom
    w_v = eB / denom
    return logS, w_u, w_v


class BB3Copula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """
    BB3 copula (Joe & Hu, 1996) — "positive stable stopped-gamma LT".

    Book formula (4.59):
        C(u,v; θ, δ) = exp( - [ (1/δ) * log( exp(δ*(-log u)^θ) + exp(δ*(-log v)^θ) - 1 ) ]^{1/θ} )

    Parameters
    ----------
    θ (theta): power parameter, θ >= 1 in the book.
    δ (delta): scale parameter, δ > 0.

    In this architecture, bounds are exclusive, so we enforce:
      theta in (1, +inf), delta in (0, +inf) (implemented with finite caps).

    Notes on numerics
    -----------------
    BB3 is *lower-tail very sharp* when θ>1 (tail-comonotonic), which makes:
      - finite differences fragile for pdf checks,
      - naive Monte-Carlo integration of pdf high-variance.
    We therefore:
      - compute in log-domain where needed,
      - compute weights w_u, w_v with a stable normalizer,
      - clip u,v into (eps,1-eps) in bulk computations,
      - treat boundaries (0/1) explicitly in get_cdf and h-functions.
    """

    def __init__(self):
        super().__init__(use_jax=False)
        self.name = "BB3 Copula"
        self.type = "bb3"
        self.default_optim_method = "Powell"

        # Exclusive bounds (align with your framework)
        bounds = [(1.0, 50.0), (1e-6, 50.0)]
        self.init_parameters(CopulaParameters(np.array([2.0, 1.5]), bounds, ["theta", "delta"]))

    # ------------------------------------------------------------------
    # Core: CDF / h-functions / PDF
    # ------------------------------------------------------------------

    @staticmethod
    def _prep_uv(u, v):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        u, v = np.broadcast_arrays(u, v)
        return u, v

    def get_cdf(self, u, v, param=None):
        """
        Compute the BB3 copula CDF C(u,v).

        Boundary handling (exact):
          C(u,0)=0, C(0,v)=0, C(u,1)=u, C(1,v)=v.

        In the interior (0,1)^2 we clip to (eps,1-eps) for stability.
        """
        if param is None:
            theta, delta = map(float, self.get_parameters())
        else:
            theta, delta = map(float, param)

        u, v = self._prep_uv(u, v)
        out = np.empty_like(u, dtype=float)

        out[(u <= 0.0) | (v <= 0.0)] = 0.0
        out[u >= 1.0] = np.clip(v[u >= 1.0], 0.0, 1.0)
        out[v >= 1.0] = np.clip(u[v >= 1.0], 0.0, 1.0)

        mask = (u > 0.0) & (u < 1.0) & (v > 0.0) & (v < 1.0)
        if not np.any(mask):
            return float(out) if out.shape == () else out

        eps = 1e-15
        um = np.clip(u[mask], eps, 1.0 - eps)
        vm = np.clip(v[mask], eps, 1.0 - eps)

        au = -np.log(um)  # \tilde u
        av = -np.log(vm)

        # a^theta computed in log-domain for safety
        au_th = _safe_exp(theta * np.log(np.maximum(au, 1e-300)))
        av_th = _safe_exp(theta * np.log(np.maximum(av, 1e-300)))

        A = delta * au_th
        B = delta * av_th

        logS, _, _ = _logS_and_weights(A, B)
        W = logS / delta
        t = _pow_pos(W, 1.0 / theta)  # W^{1/theta}

        out[mask] = np.exp(-t)
        return float(out) if out.shape == () else out

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """
        h_{V|U=u}(v) = ∂C(u,v)/∂u.

        Derived from the book CDF:
          Let A = δ(-log u)^θ, B = δ(-log v)^θ,
              S = exp(A)+exp(B)-1, logS=log S, W=logS/δ, C=exp(-W^{1/θ})
              w_u = exp(A)/S.
          Then:
              ∂C/∂u = C * W^{1/θ - 1} * w_u * (-log u)^{θ-1} / u

        Boundary in v:
          v<=0 -> 0, v>=1 -> 1 (CDF property).
        """
        if param is None:
            theta, delta = map(float, self.get_parameters())
        else:
            theta, delta = map(float, param)

        u, v = self._prep_uv(u, v)
        out = np.empty_like(u, dtype=float)

        out[v <= 0.0] = 0.0
        out[v >= 1.0] = 1.0
        out[u <= 0.0] = 0.0
        out[u >= 1.0] = 1.0

        mask = (u > 0.0) & (u < 1.0) & (v > 0.0) & (v < 1.0)
        if not np.any(mask):
            return float(out) if out.shape == () else out

        eps = 1e-15
        um = np.clip(u[mask], eps, 1.0 - eps)
        vm = np.clip(v[mask], eps, 1.0 - eps)

        au = -np.log(um)
        av = -np.log(vm)

        au_th = _safe_exp(theta * np.log(np.maximum(au, 1e-300)))
        av_th = _safe_exp(theta * np.log(np.maximum(av, 1e-300)))

        A = delta * au_th
        B = delta * av_th

        logS, w_u, _ = _logS_and_weights(A, B)

        W = logS / delta
        t = _pow_pos(W, 1.0 / theta)
        C = np.exp(-t)

        W_pow = _pow_pos(W, 1.0 / theta - 1.0)  # W^{1/θ - 1}
        au_pow = _pow_pos(au, theta - 1.0)      # (-log u)^{θ-1}

        out[mask] = C * W_pow * w_u * au_pow / um
        return float(out) if out.shape == () else out

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        # symmetry
        return self.partial_derivative_C_wrt_u(v, u, param)

    def get_pdf(self, u, v, param=None):
        """
        Mixed derivative density c(u,v) = ∂²C/∂u∂v on (0,1)^2, 0 on boundary.

        Closed-form in terms of:
          W = logS/δ, t = W^{1/θ}, C = exp(-t),
          w_u = exp(A)/S, w_v = exp(B)/S,
          a = -log u, b = -log v.

        Formula:
          c = C * (a^{θ-1} b^{θ-1} / (u v)) * (w_u w_v) * W^{1/θ - 2}
              * [ W^{1/θ} + θ δ W + θ(θ-1) ].
        """
        if param is None:
            theta, delta = map(float, self.get_parameters())
        else:
            theta, delta = map(float, param)

        u, v = self._prep_uv(u, v)
        out = np.zeros_like(u, dtype=float)

        mask = (u > 0.0) & (u < 1.0) & (v > 0.0) & (v < 1.0)
        if not np.any(mask):
            return float(out) if out.shape == () else out

        eps = 1e-15
        um = np.clip(u[mask], eps, 1.0 - eps)
        vm = np.clip(v[mask], eps, 1.0 - eps)

        au = -np.log(um)
        av = -np.log(vm)

        au_th = _safe_exp(theta * np.log(np.maximum(au, 1e-300)))
        av_th = _safe_exp(theta * np.log(np.maximum(av, 1e-300)))

        A = delta * au_th
        B = delta * av_th

        logS, w_u, w_v = _logS_and_weights(A, B)
        w_u = np.maximum(w_u, 1e-300)
        w_v = np.maximum(w_v, 1e-300)

        W = logS / delta
        logW = np.log(np.maximum(W, 1e-300))

        t = np.exp((1.0 / theta) * logW)  # W^{1/θ}
        C = np.exp(-t)

        au_pow = _pow_pos(au, theta - 1.0)
        av_pow = _pow_pos(av, theta - 1.0)

        # W^{1/θ - 2}
        W_pow = np.exp((1.0 / theta - 2.0) * logW)

        # bracket = t + theta*delta*W + theta*(theta-1)
        term1 = t
        term2 = theta * delta * W
        term3 = (theta - 1.0)
        bracket = term1 + term2 + term3

        pdf = C * (au_pow * av_pow / (um * vm)) * (w_u * w_v) * W_pow * bracket
        pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)

        out[mask] = pdf
        return float(out) if out.shape == () else out

    # ------------------------------------------------------------------
    # Dependence measures
    # ------------------------------------------------------------------

    def blomqvist_beta(self, param=None) -> float:
        if param is None:
            param = self.get_parameters()
        return float(4.0 * self.get_cdf(0.5, 0.5, param=param) - 1.0)

    def UTDC(self, param=None) -> float:
        """
        Upper tail dependence (book property):
            λ_U = 2 - 2^{1/θ}  (independent of δ).
        """
        if param is None:
            theta, _ = map(float, self.get_parameters())
        else:
            theta, _ = map(float, param)
        return float(2.0 - 2.0 ** (1.0 / theta))

    def LTDC(self, param=None) -> float:
        """
        Lower tail dependence:
          - θ = 1  -> λ_L = 2^{-1/δ}
          - θ > 1  -> λ_L = 1
        In our architecture θ is strictly > 1, so λ_L is 1 in normal use.
        """
        if param is None:
            theta, delta = map(float, self.get_parameters())
        else:
            theta, delta = map(float, param)

        if theta <= 1.0:
            return float(2.0 ** (-1.0 / delta))
        return 1.0

    def kendall_tau(self, param=None, n_quad: int = 120) -> float:
        """
        Kendall's tau for Archimedean copulas can be computed from the inverse generator:
            τ = 1 + 4 ∫_0^1 [ ψ^{-1}(u) / (ψ^{-1})'(u) ] du.

        For BB3:
            ψ^{-1}(u) = exp( δ(-log u)^θ ) - 1.
        """
        if param is None:
            theta, delta = map(float, self.get_parameters())
        else:
            theta, delta = map(float, param)

        x, w = np.polynomial.legendre.leggauss(n_quad)
        u = 0.5 * (x + 1.0)
        w = 0.5 * w

        eps = 1e-15
        u = np.clip(u, eps, 1.0 - eps)

        a = -np.log(u)                            # >0
        a_th = _safe_exp(theta * np.log(np.maximum(a, 1e-300)))  # a^θ
        g = delta * a_th                          # δ a^θ

        emg = np.exp(-np.minimum(g, 745.0))        # exp(-g) stable
        one_minus = 1.0 - emg                      # 1 - exp(-g)

        # ratio = ψ^{-1}(u) / (ψ^{-1})'(u) = - u*(1-exp(-g)) / (δ θ a^{θ-1})
        denom = delta * theta * _pow_pos(a, theta - 1.0)
        ratio = -u * one_minus / np.maximum(denom, 1e-300)

        integral = float(np.sum(w * ratio))
        tau = 1.0 + 4.0 * integral
        return float(np.clip(tau, -1.0, 1.0))

    # ------------------------------------------------------------------
    # init_from_data (MPL) + sampling (conditional inversion)
    # ------------------------------------------------------------------

    def init_from_data(self, u, v):
        """
        Robust initializer via Maximum Pseudo-Likelihood (MPL):
            maximize mean(log c(u_i,v_i)) over a subsample.

        This is far more stable than matching (tau,beta) for BB3.
        """
        u = np.asarray(u, float).ravel()
        v = np.asarray(v, float).ravel()
        m = np.isfinite(u) & np.isfinite(v)
        u, v = u[m], v[m]
        if u.size < 200:
            return self.get_parameters()

        eps_fit = 1e-5
        u = np.clip(u, eps_fit, 1.0 - eps_fit)
        v = np.clip(v, eps_fit, 1.0 - eps_fit)

        rng = default_rng(0)
        n_sub = min(3000, u.size)
        idx = rng.choice(u.size, size=n_sub, replace=False)
        uu = u[idx]
        vv = v[idx]

        (lo_th, hi_th), (lo_de, hi_de) = self.get_bounds()

        # coarse grid ranges (cap for speed)
        th_min = max(lo_th + 1e-3, 1.05)
        de_min = max(lo_de + 1e-6, 0.1)
        th_max = min(hi_th - 1e-3, 10.0)
        de_max = min(hi_de - 1e-6, 10.0)

        thetas = np.geomspace(th_min, th_max, 10)
        deltas = np.geomspace(de_min, de_max, 10)

        def nll(th, de):
            lp = np.log(np.maximum(self.get_pdf(uu, vv, param=(th, de)), 1e-300))
            val = -float(np.mean(lp))
            if not np.isfinite(val):
                return 1e99
            return val

        best = None
        best_val = float("inf")
        for th0 in thetas:
            for de0 in deltas:
                val = nll(float(th0), float(de0))
                if val < best_val:
                    best_val = val
                    best = (float(th0), float(de0))

        if best is None:
            return self.get_parameters()

        x0 = np.log(np.array(best, dtype=float))

        def obj(x):
            th = float(np.exp(x[0]))
            de = float(np.exp(x[1]))
            if not (lo_th < th < hi_th) or not (lo_de < de < hi_de):
                return 1e50
            return nll(th, de)

        res = minimize(
            obj, x0, method="Nelder-Mead",
            options={"maxiter": 160, "xatol": 1e-4, "fatol": 1e-4, "disp": False}
        )

        th_hat = float(np.exp(res.x[0]))
        de_hat = float(np.exp(res.x[1]))

        th_hat = float(np.clip(th_hat, lo_th + 1e-6, hi_th - 1e-6))
        de_hat = float(np.clip(de_hat, lo_de + 1e-6, hi_de - 1e-6))

        self.set_parameters([th_hat, de_hat])
        return self.get_parameters()

    def sample(self, n: int, rng=None, param=None):
        """
        Sampling via conditional inversion:
          U ~ Unif(0,1),
          draw P ~ Unif(0,1),
          solve ∂C(U,v)/∂u = P for v by bisection.

        Bisection is robust for BB3 even in sharp tails.
        """
        if rng is None:
            rng = default_rng(0)

        if param is None:
            theta, delta = map(float, self.get_parameters())
        else:
            theta, delta = map(float, param)

        eps = 1e-12
        u = rng.uniform(eps, 1.0 - eps, size=n)
        p = rng.uniform(eps, 1.0 - eps, size=n)

        def F(vv):
            return self.partial_derivative_C_wrt_u(u, vv, param=(theta, delta))

        lo = np.full(n, eps)
        hi = np.full(n, 1.0 - eps)

        for _ in range(60):
            mid = 0.5 * (lo + hi)
            val = F(mid)
            left = val > p
            hi = np.where(left, mid, hi)
            lo = np.where(left, lo, mid)

        v = 0.5 * (lo + hi)
        return np.column_stack([u, v])

    # GOF placeholders
    def IAD(self, data):
        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan