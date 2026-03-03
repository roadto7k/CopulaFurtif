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
from scipy.special import beta as beta_fn
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
        Copula density c(u,v;θ,δ) = ∂²C/∂u∂v.

        Formula (Joe 2014, §4.20, equation for c(u,v;θ,δ)):

            Let x = (u^{-θ}-1)^{-δ},  y = (v^{-θ}-1)^{-δ}
                a = x^{-1/δ} = u^{-θ}-1,  b = y^{-1/δ} = v^{-θ}-1
                S = x+y = a^{-δ}+b^{-δ},  T = S^{-1/δ}
                Z = 1 + a + b - T

            c = Z^{-1/θ-2} · a^{-δ-1} · b^{-δ-1} · (uv)^{-θ-1}
              · [(θ+1)(a^{δ+1} - T/S)(b^{δ+1} - T/S)
                + θ(1+δ)·Z·T/S²]

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

        eps = 1e-14
        u = np.clip(np.asarray(u, float), eps, 1.0 - eps)
        v = np.clip(np.asarray(v, float), eps, 1.0 - eps)

        a = np.maximum(u ** (-theta) - 1.0, eps)
        b = np.maximum(v ** (-theta) - 1.0, eps)

        # S = a^{-δ} + b^{-δ}  (= x + y in image notation)
        S = a ** (-delta) + b ** (-delta)
        T = S ** (-1.0 / delta)                # = (x+y)^{-1/δ}
        Z = np.maximum(1.0 + a + b - T, eps)   # copula denominator

        # two bracket terms (Joe 2014 eq. after (4.61))
        term1 = (theta + 1.0) * (a ** (delta + 1.0) - T / S) * (b ** (delta + 1.0) - T / S)
        term2 = theta * (1.0 + delta) * Z * T / (S * S)

        pdf = (
            Z ** (-1.0 / theta - 2.0)
            * a ** (-delta - 1.0) * b ** (-delta - 1.0)
            * u ** (-theta - 1.0) * v ** (-theta - 1.0)
            * (term1 + term2)
        )
        return np.maximum(pdf, 0.0)

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

    def kendall_tau(self, param=None):
        """
        Theoretical Kendall's tau.
        τ = 1 - 2·B(1 + 1/θ, 1 + 1/δ)

        where B is the beta function.

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
        return float(1.0 - 2.0 * beta_fn(1.0 + 1.0 / theta, 1.0 + 1.0 / delta))

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

        # ---- 3. Estimate θ from empirical Kendall τ given δ ---------------
        tau_emp_val, _ = sp_kendalltau(u, v)
        tau_emp_val = float(np.clip(tau_emp_val, -0.99, 0.99))

        theta0 = 1.0  # safe default

        def tau_residual(th):
            return 1.0 - 2.0 * beta_fn(1.0 + 1.0 / th, 1.0 + 1.0 / delta0) - tau_emp_val

        try:
            f_lo = tau_residual(th_lo + 1e-4)
            f_hi = tau_residual(th_hi - 1e-4)
            if f_lo * f_hi < 0:
                theta0 = brentq(tau_residual, th_lo + 1e-4, th_hi - 1e-4,
                                xtol=1e-6, rtol=1e-6, maxiter=200)
        except (ValueError, RuntimeError):
            # monotonic search: larger tau → smaller theta
            for th_try in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
                if th_lo < th_try < th_hi:
                    theta0 = th_try
                    break

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