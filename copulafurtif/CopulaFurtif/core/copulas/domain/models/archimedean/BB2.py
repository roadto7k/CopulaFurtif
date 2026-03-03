import math
from typing import Optional, Sequence

import numpy as np
from numpy.random import default_rng
from scipy.optimize import minimize
from scipy.stats import kendalltau

from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel, CopulaParameters
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


_MAX_EXP = 700.0
_LOG_SAFE = 30.0  # above this, "-1" in (e^A+e^B-1) is negligible


def _safe_exp(x: np.ndarray) -> np.ndarray:
    return np.exp(np.minimum(x, _MAX_EXP))

def _logS_and_weights(A: np.ndarray, B: np.ndarray):
    """
    Return:
      logS = log(exp(A) + exp(B) - 1)
      w_u = exp(A)/S
      w_v = exp(B)/S
    computed stably without subtracting huge numbers.
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)

    M = np.maximum(A, B)
    eA = np.exp(A - M)          # in (0,1]
    eB = np.exp(B - M)          # in (0,1]
    em = np.exp(-M)             # in (0,1], underflows to 0 when M is huge

    denom = eA + eB - em        # = (exp(A)+exp(B)-1)/exp(M)
    denom = np.maximum(denom, 1e-300)  # safety

    logS = M + np.log(denom)
    w_u = eA / denom
    w_v = eB / denom
    return logS, w_u, w_v


def _logsumexp_minus1(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute log( exp(A) + exp(B) - 1 ) stably, for A,B >= 0 (true here).
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    M = np.maximum(A, B)
    m = np.minimum(A, B)

    # When M is large, exp(M) dominates and "-1" is negligible:
    # log(exp(A)+exp(B)-1) ≈ M + log1p(exp(m-M))
    out = np.empty_like(M)

    big = M > _LOG_SAFE
    if np.any(big):
        out[big] = M[big] + np.log1p(np.exp(m[big] - M[big]))

    # When M is moderate, use log1p(expm1(A)+expm1(B)) exactly
    if np.any(~big):
        a = A[~big]
        b = B[~big]
        out[~big] = np.log1p(np.expm1(a) + np.expm1(b))

    return out


class BB2Copula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """
    BB2 copula (Joe & Hu, 1996), bivariate, two parameters theta>0, delta>0.

    Book formula (4.57):
        C(u,v) = [ 1 + (1/delta) * log( exp(delta*(u^{-theta}-1)) + exp(delta*(v^{-theta}-1)) - 1 ) ]^{-1/theta}

    Notes:
      - Lower tail dependence coefficient is 1 (lower tail comonotonic).
      - Upper tail dependence coefficient is 0 (upper tail order = 2).
    """

    def __init__(self):
        super().__init__(use_jax=False)
        self.name = "BB2 Copula"
        self.type = "bb2"
        self.default_optim_method = "Powell"

        # keep bounds exclusive in your framework
        bounds = [(1e-6, 50.0), (1e-6, 50.0)]
        self.init_parameters(CopulaParameters(np.array([2.0, 1.5]), bounds, ["theta", "delta"]))

    # ---------------------------------------------------------------------
    # Core primitives (CDF, h-functions, PDF) on (0,1)^2 + exact boundaries
    # ---------------------------------------------------------------------

    @staticmethod
    def _prep_uv(u, v):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        u, v = np.broadcast_arrays(u, v)
        return u, v

    def get_cdf(self, u, v, param=None):
        """
        Compute the BB2 copula CDF C(u,v).

        BB2 (Joe & Hu, 1996) — book formula (4.57):
            C(u,v) = [ 1 + (1/delta) * log( exp(delta*(u^{-theta}-1)) + exp(delta*(v^{-theta}-1)) - 1 ) ]^{-1/theta}

        Conventions:
          - Exact Fréchet boundaries are handled explicitly:
              C(u,0)=0, C(0,v)=0, C(u,1)=u, C(1,v)=v
          - In the interior (0,1)^2, inputs are clipped to (eps,1-eps) for numerical stability.
          - Vectorization is pairwise via broadcasting.
        """
        if param is None:
            theta, delta = map(float, self.get_parameters())
        else:
            theta, delta = map(float, param)

        u, v = self._prep_uv(u, v)
        out = np.empty_like(u, dtype=float)

        # exact Fréchet boundaries
        out[(u <= 0.0) | (v <= 0.0)] = 0.0
        out[u >= 1.0] = np.clip(v[u >= 1.0], 0.0, 1.0)
        out[v >= 1.0] = np.clip(u[v >= 1.0], 0.0, 1.0)

        mask = (u > 0.0) & (u < 1.0) & (v > 0.0) & (v < 1.0)
        if not np.any(mask):
            return float(out) if out.shape == () else out

        eps = 1e-15
        um = np.clip(u[mask], eps, 1.0 - eps)
        vm = np.clip(v[mask], eps, 1.0 - eps)

        # A = delta*(u^{-theta}-1), B analogous
        su = -theta * np.log(um)                 # >=0
        sv = -theta * np.log(vm)
        u_m_theta = _safe_exp(su)                # u^{-theta}
        v_m_theta = _safe_exp(sv)
        A = delta * (u_m_theta - 1.0)
        B = delta * (v_m_theta - 1.0)

        logS, _, _ = _logS_and_weights(A, B)
        T = 1.0 + logS / delta
        logT = np.log(T)

        out[mask] = np.exp(-(1.0 / theta) * logT)  # T^{-1/theta}

        return float(out) if out.shape == () else out

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """
        Compute h_{V|U=u}(v) = ∂C(u,v)/∂u.

        Interpretation:
          For continuous copulas, ∂C/∂u is the conditional CDF of V given U=u.
          We implement the closed-form derivative implied by the BB2 CDF.

        Boundary behavior:
          - For v<=0: 0 ; for v>=1: 1 (consistent with a conditional CDF)
        """
        if param is None:
            theta, delta = map(float, self.get_parameters())
        else:
            theta, delta = map(float, param)

        u, v = self._prep_uv(u, v)
        out = np.empty_like(u, dtype=float)

        # boundary in v: h(u,0)=0 ; h(u,1)=1
        out[v <= 0.0] = 0.0
        out[v >= 1.0] = 1.0

        # boundary in u: safe conventions (not usually queried in tests)
        out[u <= 0.0] = 0.0
        out[u >= 1.0] = 1.0

        mask = (u > 0.0) & (u < 1.0) & (v > 0.0) & (v < 1.0)
        if not np.any(mask):
            return float(out) if out.shape == () else out

        eps = 1e-15
        um = np.clip(u[mask], eps, 1.0 - eps)
        vm = np.clip(v[mask], eps, 1.0 - eps)

        su = -theta * np.log(um)
        sv = -theta * np.log(vm)
        u_m_theta = _safe_exp(su)
        v_m_theta = _safe_exp(sv)
        A = delta * (u_m_theta - 1.0)
        B = delta * (v_m_theta - 1.0)

        logS, w_u, _ = _logS_and_weights(A, B)

        T = 1.0 + logS / delta
        C = np.exp(-(1.0 / theta) * np.log(T))

        u_pow = um ** (-theta - 1.0)
        out[mask] = (C * w_u * u_pow) / T

        return float(out) if out.shape == () else out

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        # symmetry
        return self.partial_derivative_C_wrt_u(v, u, param)

    def get_pdf(self, u, v, param=None):
        """
        Compute the formal mixed derivative c(u,v) = ∂²C/∂u∂v.

        BB2 can generate extremely sharp lower-tail behavior; numerically,
        this 'density' can become very large near (0,0). This method returns
        the mixed derivative on the open square (0,1)^2 and returns 0 on the boundary.

        Note:
          In practice, tests based on Monte-Carlo integration of c(u,v) can be
          unstable unless the sampling scheme resolves the extreme lower tail.
        """
        if param is None:
            theta, delta = map(float, self.get_parameters())
        else:
            theta, delta = map(float, param)

        u, v = self._prep_uv(u, v)

        # density only meaningful in the open square; return 0 on boundary
        out = np.zeros_like(u, dtype=float)
        mask = (u > 0.0) & (u < 1.0) & (v > 0.0) & (v < 1.0)
        if not np.any(mask):
            return float(out) if out.shape == () else out

        eps = 1e-15
        um = np.clip(u[mask], eps, 1.0 - eps)
        vm = np.clip(v[mask], eps, 1.0 - eps)

        su = -theta * np.log(um)
        sv = -theta * np.log(vm)
        u_m_theta = _safe_exp(su)
        v_m_theta = _safe_exp(sv)

        A = delta * (u_m_theta - 1.0)
        B = delta * (v_m_theta - 1.0)

        logS, w_u, w_v = _logS_and_weights(A, B)
        # weights can be extremely close to 0 in extreme tails -> protect logs
        w_u = np.maximum(w_u, 1e-300)
        w_v = np.maximum(w_v, 1e-300)
        log_w = np.log(w_u) + np.log(w_v)

        T = 1.0 + logS / delta
        logT = np.log(T)

        # bracket = theta*delta*T + theta + 1, computed stably for both small/large base
        base = theta * delta * T
        log_bracket = np.where(
            base > 50.0,
            np.log(base) + np.log1p((theta + 1.0) / base),
            np.log(theta + 1.0) + np.log1p(base / (theta + 1.0)),
        )

        log_pow = (-theta - 1.0) * (np.log(um) + np.log(vm))
        logpdf = (-1.0 / theta - 2.0) * logT + log_pow + log_w + log_bracket

        pdf = np.exp(logpdf)
        pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)
        out[mask] = pdf
        return float(out) if out.shape == () else out

    # ---------------------------------------------------------------------
    # Dependence measures
    # ---------------------------------------------------------------------

    def blomqvist_beta(self, param=None) -> float:
        if param is None:
            param = self.get_parameters()
        return float(4.0 * self.get_cdf(0.5, 0.5, param=param) - 1.0)

    def LTDC(self, param=None) -> float:
        # From the book: lower tail dependence parameter is lambda_L = 1
        return 1.0

    def UTDC(self, param=None) -> float:
        # Upper tail order is 2 => coefficient lambda_U = 0
        return 0.0

    def kendall_tau(self, param=None, n_quad: int = 120) -> float:
        """
        Kendall's tau for an Archimedean copula can be computed via:
            tau = 1 + 4 * ∫_0^1 (phi(u)/phi'(u)) du

        For BB2, the Archimedean generator is:
            phi(u) = exp(delta*(u^{-theta}-1)) - 1

        We evaluate the 1D integral with Gauss–Legendre quadrature (stable + fast),
        and clip the final result to [-1,1] defensively.
        """
        if param is None:
            theta, delta = map(float, self.get_parameters())
        else:
            theta, delta = map(float, param)

        # Gauss–Legendre on (0,1)
        x, w = np.polynomial.legendre.leggauss(n_quad)
        u = 0.5 * (x + 1.0)
        w = 0.5 * w

        eps = 1e-15
        u = np.clip(u, eps, 1.0 - eps)

        # g(u) = delta*(u^{-theta}-1)
        g = delta * (np.exp(np.minimum(-theta * np.log(u), _MAX_EXP)) - 1.0)

        # 1 - exp(-g) stable (for big g, exp(-g)->0)
        emg = np.exp(-np.minimum(g, 745.0))
        one_minus_emg = 1.0 - emg

        # phi/phi' = - u^{theta+1} * (1 - exp(-g)) / (delta*theta)
        ratio = - (u ** (theta + 1.0)) * one_minus_emg / (delta * theta)

        integral = float(np.sum(w * ratio))
        tau = 1.0 + 4.0 * integral
        return float(np.clip(tau, -1.0, 1.0))

    # ---------------------------------------------------------------------
    # Fitting helpers
    # ---------------------------------------------------------------------

    def init_from_data(self, u, v):
        """
        Initialize (theta, delta) from data via Maximum Pseudo-Likelihood (MPL).

        Why MPL for BB2?
        ----------------
        For BB2, moment-matching on (tau, beta) can be ill-conditioned: different
        (theta, delta) pairs may yield similar (tau, beta) under sampling noise.
        MPL is much more stable: we maximize the average log-density log c(u,v).

        Practical approach
        ------------------
        1) Clip data into (0,1) to respect the open-interval convention.
        2) Use a deterministic subsample (for speed + robustness).
        3) Coarse grid search for a good starting point.
        4) Refine using Nelder–Mead on log-parameters (ensures positivity).
        """

        u = np.asarray(u, float).ravel()
        v = np.asarray(v, float).ravel()
        mask = np.isfinite(u) & np.isfinite(v)
        u, v = u[mask], v[mask]
        if u.size < 200:
            return self.get_parameters()

        eps = 1e-12
        u = np.clip(u, eps, 1.0 - eps)
        v = np.clip(v, eps, 1.0 - eps)

        # deterministic subsample for speed
        rng = default_rng(0)
        n_sub = min(3000, u.size)
        idx = rng.choice(u.size, size=n_sub, replace=False)
        uu = u[idx]
        vv = v[idx]

        (lo_th, hi_th), (lo_de, hi_de) = self.get_bounds()
        # fallback caps if bounds are infinite
        hi_th_cap = float(hi_th) if np.isfinite(hi_th) else 50.0
        hi_de_cap = float(hi_de) if np.isfinite(hi_de) else 50.0

        # ----- internal stable logpdf (pairwise) -----
        def _logpdf_pairwise(u_, v_, th, de):
            # mirrors get_pdf() but returns log-density directly
            u_ = np.asarray(u_, float)
            v_ = np.asarray(v_, float)
            u_, v_ = np.broadcast_arrays(u_, v_)

            su = -th * np.log(u_)
            sv = -th * np.log(v_)

            u_m_th = _safe_exp(su)
            v_m_th = _safe_exp(sv)

            A = de * (u_m_th - 1.0)
            B = de * (v_m_th - 1.0)

            logS, w_u, w_v = _logS_and_weights(A, B)
            w_u = np.maximum(w_u, 1e-300)
            w_v = np.maximum(w_v, 1e-300)

            T = 1.0 + logS / de
            logT = np.log(T)

            base = th * de * T
            log_bracket = np.where(
                base > 50.0,
                np.log(base) + np.log1p((th + 1.0) / base),
                np.log(th + 1.0) + np.log1p(base / (th + 1.0)),
            )

            log_pow = (-th - 1.0) * (np.log(u_) + np.log(v_))
            log_w = np.log(w_u) + np.log(w_v)

            return (-1.0 / th - 2.0) * logT + log_pow + log_w + log_bracket

        def _nll(th, de):
            # negative mean log-likelihood (robust)
            lp = _logpdf_pairwise(uu, vv, th, de)
            m = -float(np.mean(lp))
            if not np.isfinite(m):
                return 1e99
            return m

        # ----- coarse grid search for a good start -----
        th_min = max(float(lo_th) * 1.05, 0.2)
        de_min = max(float(lo_de) * 1.05, 0.2)
        th_max = min(hi_th_cap * 0.95, 10.0)
        de_max = min(hi_de_cap * 0.95, 10.0)

        thetas = np.geomspace(th_min, th_max, 10)
        deltas = np.geomspace(de_min, de_max, 10)

        best = None
        best_val = float("inf")
        for th0 in thetas:
            for de0 in deltas:
                val = _nll(float(th0), float(de0))
                if val < best_val:
                    best_val = val
                    best = (float(th0), float(de0))

        if best is None:
            return self.get_parameters()

        x0 = np.log(np.array(best, dtype=float))

        # ----- refine with Nelder–Mead in log-space -----
        def obj(x):
            th = float(np.exp(x[0]))
            de = float(np.exp(x[1]))
            # enforce exclusive bounds with a soft penalty
            if not (lo_th < th < hi_th_cap) or not (lo_de < de < hi_de_cap):
                return 1e50
            return _nll(th, de)

        res = minimize(
            obj, x0, method="Nelder-Mead",
            options={"maxiter": 160, "xatol": 1e-4, "fatol": 1e-4, "disp": False}
        )

        th_hat = float(np.exp(res.x[0]))
        de_hat = float(np.exp(res.x[1]))

        # clamp strictly inside bounds
        th_hat = float(np.clip(th_hat, lo_th + 1e-6, hi_th_cap - 1e-6))
        de_hat = float(np.clip(de_hat, lo_de + 1e-6, hi_de_cap - 1e-6))

        self.set_parameters([th_hat, de_hat])
        return self.get_parameters()

    # ---------------------------------------------------------------------
    # Sampling via conditional inversion (bisection)
    # ---------------------------------------------------------------------

    def sample(self, n: int, rng=None, param=None):
        if rng is None:
            rng = default_rng(0)

        if param is None:
            theta, delta = map(float, self.get_parameters())
        else:
            theta, delta = map(float, param)

        eps = 1e-12
        u = rng.uniform(eps, 1.0 - eps, size=n)
        p = rng.uniform(eps, 1.0 - eps, size=n)

        # For BB2 with this CDF, ∂C/∂u(u,1)=1, so F_{V|U=u}(v) = ∂C/∂u(u,v)
        def F(vv):
            return self.partial_derivative_C_wrt_u(u, vv, param=(theta, delta))

        lo = np.full(n, eps)
        hi = np.full(n, 1.0 - eps)

        for _ in range(60):
            mid = 0.5 * (lo + hi)
            val = F(mid)
            go_left = val > p
            hi = np.where(go_left, mid, hi)
            lo = np.where(go_left, lo, mid)

        v = 0.5 * (lo + hi)
        return np.column_stack([u, v])

    # ---------------------------------------------------------------------
    # GOF placeholders (as in your other models)
    # ---------------------------------------------------------------------

    def IAD(self, data):
        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan