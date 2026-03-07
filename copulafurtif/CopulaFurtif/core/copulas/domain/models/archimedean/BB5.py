import numpy as np
from numpy import log as _np_log, exp as _np_exp

from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel, CopulaParameters
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence

_FLOAT_MIN = 1e-308  # strictly positive, smallest normal double
_FLOAT_MAX_LOG = 709.0  # np.exp(709) ~= 8.2e307, still in float64 range
_FLOAT_MIN_LOG = -745.0  # np.exp(-745) ~= 5e-324, smallest sub‑normal


def _safe_log(x):
    """log clipped away from 0.  Works on arrays or scalars."""
    return _np_log(np.clip(x, _FLOAT_MIN, None))


def _safe_exp(log_x):
    """exp with log‑input clipping so the result is finite."""
    return _np_exp(np.clip(log_x, _FLOAT_MIN_LOG, _FLOAT_MAX_LOG))


def _safe_pow(base, exponent):
    """Compute base**exponent robustly for base>0 via exp(exponent*log(base))."""
    return _safe_exp(exponent * _safe_log(base))


class BB5Copula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """
    BB5 Copula (Joe's two-parameter extreme-value copula).

    Attributes:
        name (str): Human-readable name of the copula.
        type (str): Identifier for the copula family.
        bounds_param (list of tuple): Bounds for copula parameters [theta, delta].
        parameters (np.ndarray): Current copula parameters.
        default_optim_method (str): Optimization method.
    """

    def __init__(self):
        """Initialize BB5 copula with default parameters theta=1.0, delta=1.0."""
        super().__init__()
        self.name = "BB5 Copula"
        self.type = "bb5"
        self.default_optim_method = "Powell"
        self.init_parameters(
            CopulaParameters(np.array([2.0, 2.0]), [(1.0, np.inf), (1e-6, np.inf)], ["theta", "delta"]))

    def get_cdf(self, u, v, param=None):
        """
        Evaluate the copula cumulative distribution function at (u, v).

        Args:
            u (float or array-like): First uniform margin in (0,1).
            v (float or array-like): Second uniform margin in (0,1).
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float or np.ndarray: CDF value C(u, v).
        """

        if param is None:
            param = self.get_parameters()
        theta, delta = param

        # Clip inputs away from 0 and 1 to avoid log(0)
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        x = -_safe_log(u)
        y = -_safe_log(v)

        # Pre‑compute powers in log‑space where possible
        log_xt = theta * _safe_log(x)
        log_yt = theta * _safe_log(y)

        log_xdt = -delta * theta * _safe_log(x)  # log(x^(-delta*theta))
        log_ydt = -delta * theta * _safe_log(y)

        # log‑sum‑exp for S = x^(-δθ)+y^(-δθ)
        log_Sdt = np.logaddexp(log_xdt, log_ydt)
        # (x^(-δθ)+y^(-δθ))^{-1/δ}
        log_xyp = -(1.0 / delta) * log_Sdt

        # Go back to real space where safe
        xt = _safe_exp(log_xt)
        yt = _safe_exp(log_yt)
        xyp = _safe_exp(log_xyp)

        w = xt + yt - xyp
        w = np.clip(w, 1e-300, np.inf)  # guarantee strictly positive

        g = _safe_pow(w, 1.0 / theta)
        log_C = -g
        C = _safe_exp(log_C)
        return C

    def get_pdf(self, u, v, param=None):
        """
        Evaluate the copula probability density function at (u, v).

        Args:
            u (float or array-like): First uniform margin in (0,1).
            v (float or array-like): Second uniform margin in (0,1).
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float or np.ndarray: PDF value c(u, v).
        """

        if param is None:
            param = self.get_parameters()
        theta, delta = param

        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        x = -_safe_log(u)
        y = -_safe_log(v)

        # Same log‑space machinery as in CDF
        log_xt = theta * _safe_log(x)
        log_yt = theta * _safe_log(y)
        log_xdt = -delta * theta * _safe_log(x)
        log_ydt = -delta * theta * _safe_log(y)
        log_Sdt = np.logaddexp(log_xdt, log_ydt)
        log_xyp = -(1.0 / delta) * log_Sdt

        xt = _safe_exp(log_xt)
        yt = _safe_exp(log_yt)
        xdt = _safe_exp(log_xdt)
        ydt = _safe_exp(log_ydt)
        xyp = _safe_exp(log_xyp)
        S = xdt + ydt  # in real space; finite by construction

        w = xt + yt - xyp
        w = np.clip(w, 1e-300, np.inf)

        g = _safe_pow(w, 1.0 / theta)
        C = _safe_exp(-g)

        # Derivative helpers (all finite by design)
        dx_du = -1.0 / u
        dy_dv = -1.0 / v

        dxt_du = theta * _safe_pow(x, theta - 1) * dx_du
        dyt_dv = theta * _safe_pow(y, theta - 1) * dy_dv

        dxdt_du = -delta * theta * _safe_pow(x, -delta * theta - 1) * dx_du
        dydt_dv = -delta * theta * _safe_pow(y, -delta * theta - 1) * dy_dv

        S_pow = _safe_pow(S, -1.0 / delta - 1)
        dw_du = dxt_du - (-(1.0 / delta) * S_pow * dxdt_du)
        dw_dv = dyt_dv - (-(1.0 / delta) * S_pow * dydt_dv)

        w_uv = -(
                (1.0 / delta) * (1.0 / delta + 1.0) * _safe_pow(S, -1.0 / delta - 2) * dxdt_du * dydt_dv
        )

        dg_dw = (1.0 / theta) * _safe_pow(w, 1.0 / theta - 1)
        d2g_dw2 = (1.0 / theta) * (1.0 / theta - 1.0) * _safe_pow(w, 1.0 / theta - 2)

        term1 = -C * dg_dw * w_uv
        term2 = -C * d2g_dw2 * dw_du * dw_dv
        term3 = C * (dg_dw ** 2) * dw_du * dw_dv

        pdf = term1 + term2 + term3
        return np.clip(pdf, 0.0, None)

    # ------------------------------------------------------------------
    # Cached Gauss-Legendre nodes/weights for kendall_tau (32×32 grid)
    # ------------------------------------------------------------------
    _GL_N = 32
    _xi_gl, _wi_gl = __import__(
        'numpy.polynomial.legendre', fromlist=['leggauss']
    ).leggauss(_GL_N)
    _GL_HALF = 0.5 * (0.995 - 0.005)
    _GL_MID = 0.5 * (0.995 + 0.005)
    _GL_U = _GL_HALF * _xi_gl + _GL_MID
    _GL_W = _GL_HALF * _wi_gl
    _GL_UU, _GL_VV = np.meshgrid(_GL_U, _GL_U)
    _GL_WU, _GL_WV = np.meshgrid(_GL_W, _GL_W)

    def kendall_tau(self, param=None):
        """
        Kendall's τ via 2-D Gauss-Legendre quadrature (32×32 grid).

        BB5 is an extreme-value copula (no closed-form τ expression).

        Uses the concordance identity:
            τ = 4 · ∫₀¹∫₀¹ C(u,v) · c(u,v) du dv − 1

        Parameters
        ----------
        param : [theta, delta], optional

        Returns
        -------
        float  in (0, 1)
        """
        if param is None:
            param = self.get_parameters()

        n = self._GL_N
        U = self._GL_UU.ravel()
        V = self._GL_VV.ravel()

        cdf_vals = self.get_cdf(U, V, param).reshape(n, n)
        pdf_vals = self.get_pdf(U, V, param).reshape(n, n)

        Q = float(np.sum(cdf_vals * pdf_vals * self._GL_WU * self._GL_WV))
        return float(np.clip(4.0 * Q - 1.0, 0.0, 1.0 - 1e-15))

    def sample(self, n, rng=None):
        """
        Draw n samples from the BB5 copula via conditional inversion.

        For each u ~ Uniform(0,1), we solve h(v|u) = W for v, where:

            h(v|u) = ∂C/∂u / (∂C/∂u evaluated at v→1) → simplified to
            h(v|u) = C_{2|1}(v|u) = u^{-1} · C(u,v) · (dg/dw) · (dw/du) · (−1)

        i.e. the already-implemented partial_derivative_C_wrt_u divided by
        the marginal density of u (which equals 1 for uniform margins).

        Parameters
        ----------
        n   : int – number of samples
        rng : np.random.Generator, optional

        Returns
        -------
        np.ndarray, shape (n, 2), columns ∈ (0,1)
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

            # At v→0, h→0; at v→1, h→1 — bracket always valid
            try:
                v_sol = brentq(obj, 1e-8, 1 - 1e-8, xtol=1e-10, maxiter=100)
            except ValueError:
                # fallback: closest boundary
                v_sol = 1e-6 if abs(obj(1e-8)) < abs(obj(1 - 1e-8)) else 1 - 1e-6
            V[i] = v_sol

        return np.column_stack([U, V])

    def blomqvist_beta(self, param=None):
        """
        Blomqvist's β = 4·C(½,½) − 1.

        For BB5, λ_U = 2 − (2 − 2^{−1/δ})^{1/θ} and
        β = 2^{λ_U} − 1  (Joe 2014, p.200).

        Parameters
        ----------
        param : [theta, delta], optional

        Returns
        -------
        float
        """
        if param is None:
            param = self.get_parameters()
        # Use the exact closed-form from Joe p.200
        lam_U = self.UTDC(param)
        return float(2.0 ** lam_U - 1.0)

    def init_from_data(self, u, v):
        """
        Moment-matching initialisation from pseudo-observations.

        Strategy
        --------
        1. Estimate δ from empirical λ_U using tail quantile ratios.
        2. For that δ₀, bracket and solve for θ via numerical kendall_tau.

        Parameters
        ----------
        u, v : array-like  – pseudo-observations in (0,1)

        Returns
        -------
        np.ndarray [theta0, delta0]
        """
        from scipy.optimize import brentq
        from scipy.stats import kendalltau as sp_kendalltau

        u = np.asarray(u, float)
        v = np.asarray(v, float)

        th_lo, de_lo = 1.0 + 1e-4, 1e-4

        # ---- 1. Estimate δ from empirical λ_U ---------------------------------
        qs = (0.90, 0.92, 0.94, 0.96, 0.98)
        lam_vals = []
        for q in qs:
            uq, vq = np.quantile(u, q), np.quantile(v, q)
            joint = np.mean((u > uq) & (v > vq))
            lam_vals.append(joint / max(1.0 - q, 1e-9))
        lam_U_emp = float(np.clip(np.median(lam_vals), 1e-6, 1.9999))

        # Invert λ_U = 2 − (2 − 2^{-1/δ})^{1/θ} w.r.t. δ at θ=2 (prior)
        # (2 − λ_U)^θ = 2 − 2^{-1/δ}  →  δ = −1/log₂(1 − (2−λ_U)^θ)
        theta_prior = 2.0
        rhs = float(np.clip((2.0 - lam_U_emp) ** theta_prior, 1e-9, 1 - 1e-9))
        inner = 1.0 - rhs  # = 2^{-1/δ}
        if inner > 0:
            d_try = -1.0 / np.log2(inner)
            delta0 = float(np.clip(d_try if np.isfinite(d_try) and d_try > 0 else 1.0,
                                   de_lo, 300.0))
        else:
            delta0 = 1.0

        # ---- 2. Estimate θ by grid search + Brentq on numerical τ(θ,δ₀) ------
        tau_emp_val, _ = sp_kendalltau(u, v)
        tau_emp_val = float(np.clip(tau_emp_val, 0.01, 0.99))

        theta_grid = [1.05, 1.3, 1.7, 2.5, 4.0, 7.0, 15.0, 40.0]
        theta_grid = [t for t in theta_grid if t > th_lo]

        tau_grid = []
        for th in theta_grid:
            try:
                tau_grid.append((th, self.kendall_tau([th, delta0])))
            except Exception:
                tau_grid.append((th, np.nan))

        theta0 = theta_grid[0] if theta_grid else 2.0
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

        theta0 = float(np.clip(theta0, th_lo, 200.0))
        delta0 = float(np.clip(delta0, de_lo, 300.0))
        return np.array([theta0, delta0], dtype=float)

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """
        Compute the partial derivative ∂C(u,v)/∂u of the copula CDF.

        Args:
            u (float or array-like): First margin in (0,1).
            v (float or array-like): Second margin in (0,1).
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float or np.ndarray: Value of ∂C/∂u at (u, v).
        """

        if param is None:
            param = self.get_parameters()
        theta, delta = param

        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        x = -_safe_log(u)
        y = -_safe_log(v)

        log_xdt = -delta * theta * _safe_log(x)
        log_ydt = -delta * theta * _safe_log(y)
        log_Sdt = np.logaddexp(log_xdt, log_ydt)
        S_pow = _safe_pow(_safe_exp(log_Sdt), -1.0 / delta - 1)

        w = _safe_pow(x, theta) + _safe_pow(y, theta) - _safe_pow(_safe_exp(log_Sdt), -1.0 / delta)
        w = np.clip(w, 1e-300, np.inf)

        C = _safe_exp(-_safe_pow(w, 1.0 / theta))
        dg_dw = (1.0 / theta) * _safe_pow(w, 1.0 / theta - 1)

        dxt_du = theta * _safe_pow(x, theta - 1) * (-1.0 / u)
        dxdt_du = -delta * theta * _safe_pow(x, -delta * theta - 1) * (-1.0 / u)
        dw_du = dxt_du - (-(1.0 / delta) * S_pow * dxdt_du)
        return -C * dg_dw * dw_du

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """
        Compute the partial derivative ∂C(u,v)/∂v of the copula CDF.

        Args:
            u (float or array-like): First margin in (0,1).
            v (float or array-like): Second margin in (0,1).
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float or np.ndarray: Value of ∂C/∂v at (u, v).
        """

        return self.partial_derivative_C_wrt_u(v, u, param)

    def LTDC(self, param=None):
        """
        Compute the lower tail dependence coefficient (LTDC) of the copula.

        Args:
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float: LTDC value (0.0 for this copula).
        """

        return 0.0

    def UTDC(self, param=None):
        """
        Compute the upper tail dependence coefficient (UTDC) of the copula.

        Args:
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float: UTDC value (2 − 2^(1/δ)).
        """

        if param is None:
            param = self.get_parameters()

        theta = param[0]
        delta = param[1]
        return 2.0 - (2.0 - 2.0 ** (-1.0 / delta)) ** (1.0 / theta)

    def IAD(self, data):
        """
        Return NaN for the Integrated Anderson-Darling (IAD) statistic.

        Args:
            data (Sequence[array-like, array-like]): Ignored pseudo-observations.

        Returns:
            float: Always returns numpy.nan.
        """

        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        """
        Return NaN for the Anderson-Darling (AD) statistic.

        Args:
            data (Sequence[array-like, array-like]): Ignored pseudo-observations.

        Returns:
            float: Always returns numpy.nan.
        """

        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan