"""
BB4 Copula (Joe and Hu, 1996) — two-parameter Archimedean copula.

CDF (Joe 2014, §4.20):
    C(u,v;θ,δ) = (u^{-θ} + v^{-θ} - 1
                  - [(u^{-θ}-1)^{-δ} + (v^{-θ}-1)^{-δ}]^{-1/δ})^{-1/θ}

Parameters
----------
θ > 0  (parent class enforces open interval)
δ > 0  (parent class enforces open interval)

Key properties
--------------
• Symmetric copula: C(u,v) = C(v,u).
• Both lower and upper tail dependence.
  λ_L = (2 - 2^{-1/δ})^{-1/θ}   λ_U = 2^{-1/δ}
• Kendall's τ = 1 - 2·B(1+1/θ, 1+1/δ)   (B = beta function)
• Blomqvist β = 4·C(½,½) - 1
  with C(½,½) = (2^{θ+1} - 1 - 2^{-1/δ}(2^θ-1))^{-1/θ}
• δ → ∞ → Gumbel limit;  θ → 0 → Galambos limit.
"""

import numpy as np
from numpy.random import default_rng
from scipy.optimize import brentq
from scipy.stats import kendalltau as sp_kendalltau


from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel, CopulaParameters
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


class BB4Copula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """
    BB4 Copula — Joe & Hu (1996) two-parameter family.

    Attributes
    ----------
    name : str
    type : str
    bounds_param : list[tuple]   [(0, ∞), (0, ∞)]  — parent enforces open bounds
    parameters : np.ndarray      [theta, delta]
    default_optim_method : str
    """

    def __init__(self):
        super().__init__()
        self.name = "BB4 Copula"
        self.type = "bb4"
        self.default_optim_method = "Powell"
        # Parent class excludes the endpoints → effectively (0, ∞) × (0, ∞)
        self.init_parameters(
            CopulaParameters(
                np.array([1.0, 1.0]),
                [(0, np.inf), (0, np.inf)],
                ["theta", "delta"],
            )
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _logsumexp(a, b):
        """Numerically stable log(exp(a) + exp(b))."""
        m = np.maximum(a, b)
        return m + np.log1p(np.exp(-np.abs(a - b)))

    @staticmethod
    def _core(u, v, theta, delta, eps=1e-14):
        """
        Return (x, y, a, b, S, T, Z) used by CDF/PDF/h-functions.
            x = u^{-θ},  y = v^{-θ}
            a = x-1,     b = y-1
            S = a^{-δ} + b^{-δ}
            T = S^{-1/δ}
            Z = x + y - 1 - T
        """
        u = np.clip(u, eps, 1.0 - eps)
        v = np.clip(v, eps, 1.0 - eps)
        x = u ** (-theta)
        y = v ** (-theta)
        a = x - 1.0
        b = y - 1.0
        # use logsumexp for numerical stability
        log_a = np.log(np.maximum(a, eps))
        log_b = np.log(np.maximum(b, eps))
        log_S = BB4Copula._logsumexp(-delta * log_a, -delta * log_b)
        S = np.exp(log_S)
        T = np.exp(-log_S / delta)
        Z = x + y - 1.0 - T
        return x, y, a, b, S, T, Z, log_a, log_b, log_S

    # ------------------------------------------------------------------
    # CDF
    # ------------------------------------------------------------------

    def get_cdf(self, u, v, param=None):
        """
        Copula CDF C(u,v;θ,δ).

        Parameters
        ----------
        u, v : float or array-like
        param : [theta, delta], optional

        Returns
        -------
        float or np.ndarray
        """
        if param is None:
            param = self.get_parameters()
        theta, delta = param
        *_, Z, _, _, _ = self._core(u, v, theta, delta)
        Z = np.maximum(Z, 1e-300)
        return np.round(Z ** (-1.0 / theta), 14)

    # ------------------------------------------------------------------
    # PDF
    # ------------------------------------------------------------------

    def get_pdf(self, u, v, param=None):
        """
        Copula density c(u,v;θ,δ) = ∂²C/∂u∂v, computed fully in log-space.

        Formula (Joe 2014, §4.20):
            a = u^{-θ}-1,  b = v^{-θ}-1
            S = a^{-δ}+b^{-δ},  T = S^{-1/δ},  Z = 1+a+b−T

            c = Z^{-1/θ-2} · a^{-δ-1} · b^{-δ-1} · (uv)^{-θ-1}
              · [(θ+1)(a^{δ+1}−T/S)(b^{δ+1}−T/S) + θ(1+δ)·Z·T/S²]

        Note: a^{δ+1} ≥ T/S always (since S ≥ a^{-δ}), so each bracket
        factor is non-negative and can be computed safely in log-space via
        log1p(-exp(log_ratio)) where log_ratio ≤ 0.

        Parameters
        ----------
        u, v : float or array-like
        param : [theta, delta], optional

        Returns
        -------
        float or np.ndarray
        """
        if param is None:
            param = self.get_parameters()
        theta, delta = float(param[0]), float(param[1])

        eps = 1e-14
        u = np.clip(np.asarray(u, float), eps, 1.0 - eps)
        v = np.clip(np.asarray(v, float), eps, 1.0 - eps)

        # --- log-space core quantities ---
        log_a = np.log(np.maximum(u ** (-theta) - 1.0, eps))
        log_b = np.log(np.maximum(v ** (-theta) - 1.0, eps))

        # log(S) = log(a^{-δ} + b^{-δ}) via logsumexp
        log_S = self._logsumexp(-delta * log_a, -delta * log_b)
        log_T = -log_S / delta                   # log(S^{-1/δ})

        # Z = 1 + a + b - T  (always > 0 for valid copula parameters)
        Z = np.maximum(1.0 + np.exp(log_a) + np.exp(log_b) - np.exp(log_T), eps)
        log_Z = np.log(Z)

        # --- leading log-factor ---
        # log( Z^{-1/θ-2} · a^{-δ-1} · b^{-δ-1} · u^{-θ-1} · v^{-θ-1} )
        log_lead = (
            (-1.0 / theta - 2.0) * log_Z
            + (-delta - 1.0) * (log_a + log_b)
            + (-theta - 1.0) * (np.log(u) + np.log(v))
        )

        # --- bracket term P1 = (θ+1)·f1·f2 ---
        # f1 = a^{δ+1} - T/S = a^{δ+1} · (1 - exp(log_ratio_a))
        # log_ratio_a = log(T/S) - log(a^{δ+1}) = -(1/δ+1)·log_S - (δ+1)·log_a ≤ 0
        log_ratio_a = -(1.0 / delta + 1.0) * log_S - (delta + 1.0) * log_a
        log_ratio_b = -(1.0 / delta + 1.0) * log_S - (delta + 1.0) * log_b
        # clip to strictly < 0 to avoid log1p(-1) = -inf
        log_ratio_a = np.minimum(log_ratio_a, -1e-12)
        log_ratio_b = np.minimum(log_ratio_b, -1e-12)

        log_f1 = (delta + 1.0) * log_a + np.log1p(-np.exp(log_ratio_a))
        log_f2 = (delta + 1.0) * log_b + np.log1p(-np.exp(log_ratio_b))
        log_P1 = np.log(theta + 1.0) + log_f1 + log_f2

        # --- bracket term P2 = θ·(1+δ)·Z·T/S² ---
        # log(T/S²) = log_T - 2·log_S = -(1/δ + 2)·log_S
        log_P2 = (
            np.log(theta * (1.0 + delta))
            + log_Z
            - (1.0 / delta + 2.0) * log_S
        )

        # --- combine and exponentiate ---
        log_bracket = np.logaddexp(log_P1, log_P2)
        log_pdf = log_lead + log_bracket
        return np.maximum(np.exp(log_pdf), 0.0)

    # ------------------------------------------------------------------
    # Partial derivatives (h-functions)
    # ------------------------------------------------------------------

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """
        ∂C(u,v)/∂u  — conditional CDF F_{V|U}(v|u).

        Parameters
        ----------
        u, v : float or array-like
        param : [theta, delta], optional

        Returns
        -------
        float or np.ndarray  in (0, 1)
        """
        if param is None:
            param = self.get_parameters()
        theta, delta = param

        x, y, a, b, S, T, Z, log_a, log_b, log_S = self._core(u, v, theta, delta)

        # ∂x/∂u = -θ u^{-θ-1}
        eps = 1e-14
        u_arr = np.clip(np.asarray(u, float), eps, 1.0 - eps)
        dxdu = -theta * u_arr ** (-theta - 1.0)

        # ∂T/∂u = a^{-δ-1} · S^{-1/δ-1} · dxdu
        log_dT = (-delta - 1.0) * log_a + (-1.0 / delta - 1.0) * log_S + np.log(np.abs(dxdu))
        dTdu = np.sign(dxdu) * np.exp(log_dT)

        dZdu = dxdu - dTdu
        Z_clipped = np.maximum(Z, 1e-300)
        return np.clip((-1.0 / theta) * Z_clipped ** (-1.0 / theta - 1.0) * dZdu, 0.0, 1.0)

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """
        ∂C(u,v)/∂v — by symmetry of C.

        Parameters
        ----------
        u, v : float or array-like
        param : [theta, delta], optional

        Returns
        -------
        float or np.ndarray  in (0, 1)
        """
        return self.partial_derivative_C_wrt_u(v, u, param)

    # ------------------------------------------------------------------
    # Tail dependence
    # ------------------------------------------------------------------

    def LTDC(self, param=None):
        """
        Lower tail dependence coefficient.
        λ_L = (2 - 2^{-1/δ})^{-1/θ}

        Parameters
        ----------
        param : [theta, delta], optional

        Returns
        -------
        float
        """
        if param is None:
            param = self.get_parameters()
        theta, delta = param
        return float((2.0 - 2.0 ** (-1.0 / delta)) ** (-1.0 / theta))

    def UTDC(self, param=None):
        """
        Upper tail dependence coefficient.
        λ_U = 2^{-1/δ}

        Parameters
        ----------
        param : [theta, delta], optional

        Returns
        -------
        float
        """
        if param is None:
            param = self.get_parameters()
        delta = float(param[1])
        return float(2.0 ** (-1.0 / delta))

    # ------------------------------------------------------------------
    # Kendall's tau
    # ------------------------------------------------------------------

    # Pre-computed 32-point Gauss-Legendre nodes/weights on [0.005, 0.995]
    # (computed once at import time, reused by every kendall_tau call)
    _GL_N = 32
    _xi_gl, _wi_gl = __import__('numpy.polynomial.legendre', fromlist=['leggauss']).leggauss(_GL_N)
    _GL_HALF = 0.5 * (0.995 - 0.005)
    _GL_MID  = 0.5 * (0.995 + 0.005)
    _GL_U    = _GL_HALF * _xi_gl + _GL_MID        # nodes on [0.005, 0.995]
    _GL_W    = _GL_HALF * _wi_gl                   # scaled weights
    # meshgrid (32×32 = 1024 points)
    _GL_UU, _GL_VV = np.meshgrid(_GL_U, _GL_U)    # shape (32, 32)
    _GL_WU, _GL_WV = np.meshgrid(_GL_W, _GL_W)

    def kendall_tau(self, param=None):
        """
        Kendall's τ via 2D Gauss-Legendre quadrature (32×32 grid).

        BB4 is NOT Archimedean so the standard φ/φ' formula does not apply.
        The formula τ = 1 − 2·B(1+1/θ, 1+1/δ) is INCORRECT for BB4.

        General concordance identity:
            τ = 4·∫₀¹∫₀¹ C(u,v)·c(u,v) du dv − 1

        Parameters
        ----------
        param : [theta, delta], optional

        Returns
        -------
        float  in (0, 1) for θ,δ > 0
        """
        if param is None:
            param = self.get_parameters()

        n = self._GL_N
        U = self._GL_UU.ravel()
        V = self._GL_VV.ravel()

        cdf_vals = self.get_cdf(U, V, param).reshape(n, n)
        pdf_vals = self.get_pdf(U, V, param).reshape(n, n)

        Q = float(np.sum(cdf_vals * pdf_vals * self._GL_WU * self._GL_WV))
        # clip to [0, 1) — exact 1.0 can occur numerically for extreme params
        return float(np.clip(4.0 * Q - 1.0, 0.0, 1.0 - 1e-15))

    # ------------------------------------------------------------------
    # Blomqvist beta
    # ------------------------------------------------------------------

    def blomqvist_beta(self, param=None):
        """
        Blomqvist's β = 4·C(½, ½) - 1.

        Closed form (from image eq. after (4.61)):
            C(½,½) = (2^{θ+1} - 1 - 2^{-1/δ}·(2^θ - 1))^{-1/θ}

        Parameters
        ----------
        param : [theta, delta], optional

        Returns
        -------
        float
        """
        if param is None:
            param = self.get_parameters()
        theta, delta = float(param[0]), float(param[1])
        lam_U = 2.0 ** (-1.0 / delta)
        inner = 2.0 ** (theta + 1.0) - 1.0 - lam_U * (2.0 ** theta - 1.0)
        c_half = max(inner, 1e-300) ** (-1.0 / theta)
        return float(4.0 * c_half - 1.0)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, n, param=None, rng=None, eps=1e-12, max_iter=48):
        """
        Draw n i.i.d. pairs (U, V) via conditional inversion (vectorised bisection).

        Algorithm
        ---------
        1. U ~ Unif(0,1)
        2. P ~ Unif(0,1)  (target quantile)
        3. For each i, solve ∂C/∂u(U_i, v) = P_i  in v by bisection.

        Parameters
        ----------
        n : int
        param : [theta, delta], optional
        rng : np.random.Generator, optional
        eps : float
            Hard clip for margins.
        max_iter : int
            Bisection depth (2^{-max_iter} absolute tolerance).

        Returns
        -------
        np.ndarray, shape (n, 2)
        """
        if rng is None:
            rng = default_rng()
        if param is None:
            param = self.get_parameters()

        theta, delta = float(param[0]), float(param[1])

        u = rng.random(n)
        p = rng.random(n)

        lo = np.full(n, eps)
        hi = np.full(n, 1.0 - eps)

        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            cdf_mid = self.partial_derivative_C_wrt_u(u, mid, param)
            above = cdf_mid > p
            hi[above] = mid[above]
            lo[~above] = mid[~above]

        v = 0.5 * (lo + hi)
        np.clip(u, eps, 1.0 - eps, out=u)
        np.clip(v, eps, 1.0 - eps, out=v)
        return np.column_stack((u, v))

    # ------------------------------------------------------------------
    # init_from_data
    # ------------------------------------------------------------------

    def init_from_data(self, u, v):
        """
        Initialise (θ, δ) from pseudo-observations (u, v).

        Strategy
        --------
        1. Estimate δ from empirical upper tail dependence:
               λ̂_U  →  δ = -1 / log₂(λ̂_U)
        2. Given δ, estimate θ by inverting the closed-form tau:
               τ = 1 - 2·B(1+1/θ, 1+1/δ)  →  solve for θ.
        3. Fallback: if step 1 fails, estimate δ from Blomqvist β̂,
           then repeat step 2.

        Parameters
        ----------
        u, v : array-like
            Pseudo-observations in (0, 1).

        Returns
        -------
        np.ndarray, shape (2,)
            Initial guess [theta₀, delta₀].
        """
        u = np.asarray(u, float)
        v = np.asarray(v, float)
        low_th, high_th = self.get_bounds()[0]
        low_de, high_de = self.get_bounds()[1]
        th_lo = max(low_th, 1e-4) if np.isfinite(low_th) else 1e-4
        th_hi = min(high_th, 300.0) if np.isfinite(high_th) else 300.0
        de_lo = max(low_de, 1e-4) if np.isfinite(low_de) else 1e-4
        de_hi = min(high_de, 300.0) if np.isfinite(high_de) else 300.0

        # ---- 1. Estimate δ from empirical λ_U ----------------------------
        delta0 = None
        qs = (0.90, 0.92, 0.94, 0.96, 0.98)
        lam_vals = []
        for q in qs:
            uq, vq = np.quantile(u, q), np.quantile(v, q)
            joint = np.mean((u > uq) & (v > vq))
            denom = max(1.0 - q, 1e-9)
            lam_vals.append(joint / denom)
        lam_U_emp = float(np.clip(np.median(lam_vals), 1e-6, 0.9999))

        if 0 < lam_U_emp < 1:
            d_try = -1.0 / np.log2(lam_U_emp)
            if np.isfinite(d_try) and de_lo < d_try < de_hi:
                delta0 = d_try

        # ---- 2. Fallback via Blomqvist β̂ ---------------------------------
        if delta0 is None:
            same_half = np.mean(((u > 0.5) & (v > 0.5)) | ((u <= 0.5) & (v <= 0.5)))
            beta_emp = float(np.clip(2.0 * same_half - 1.0, -0.99, 0.99))
            # β = 4·C(½,½)-1, C(½,½) = {2^{θ+1}-1 - λ_U(2^θ-1)}^{-1/θ}
            # For a rough estimate assume θ ≈ 1: solve 4·(2^2-1-λ_U)^{-1}-1 = β
            # → λ_U = 3 - (β+1)/4  = (12-β-1)/4
            lam_U_from_beta = np.clip((12.0 - beta_emp - 1.0) / 4.0, 1e-6, 0.9999)
            d_try = -1.0 / np.log2(float(lam_U_from_beta))
            delta0 = float(np.clip(d_try if np.isfinite(d_try) and d_try > 0 else 1.0,
                                   de_lo, de_hi))

        delta0 = float(np.clip(delta0, de_lo, de_hi))

        # ---- 3. Estimate θ by grid search + Brentq on numerical τ(θ,δ₀) ----
        tau_emp_val, _ = sp_kendalltau(u, v)
        tau_emp_val = float(np.clip(tau_emp_val, 0.01, 0.99))

        # Coarse grid to bracket the root (τ increases with θ)
        theta_grid = [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 4.0, 8.0, 20.0, 50.0]
        theta_grid = [t for t in theta_grid if th_lo < t < th_hi]

        tau_grid = []
        for th in theta_grid:
            try:
                tau_grid.append((th, self.kendall_tau([th, delta0])))
            except Exception:
                tau_grid.append((th, np.nan))

        # Find bracket where tau crosses tau_emp_val
        theta0 = theta_grid[0] if theta_grid else 1.0
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
                # track closest if no crossing found
                if abs(t_lo_br - tau_emp_val) < abs(self.kendall_tau([theta0, delta0]) - tau_emp_val):
                    theta0 = th_lo_br

        theta0 = float(np.clip(theta0, th_lo, th_hi))
        return np.array([theta0, delta0], dtype=float)

    # ------------------------------------------------------------------
    # Goodness-of-fit stubs (disabled)
    # ------------------------------------------------------------------

    def IAD(self, data):
        """Integrated Anderson-Darling — disabled for BB4."""
        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        """Anderson-Darling — disabled for BB4."""
        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan