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


class BB7Copula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """
    BB7 Copula (Joe-Clayton) Archimedean copula.

    Attributes:
        name (str): Name of the copula.
        type (str): Identifier for the copula family.
        bounds_param (list of tuple): Parameter bounds for optimization.
        parameters (np.ndarray): Copula parameters [theta, delta].
        default_optim_method (str): Optimization method.
    """

    def __init__(self):
        """Initialize the BB7 copula with default parameters."""
        super().__init__()
        self.name = "BB7 Copula"
        self.type = "bb7"
        self.default_optim_method = "Powell"
        self.init_parameters(CopulaParameters(np.array([2.0, 2.0]), [(1e-6, np.inf), (1e-6, np.inf)], ["theta", "delta"]))

    @staticmethod
    def _phi(t, theta, delta):
        """
        Compute the φ-transform for the BB7 copula.

        Args:
            t (float or array-like): Input variable in (0,1).
            theta (float): Copula parameter θ.
            delta (float): Copula parameter δ.

        Returns:
            float or numpy.ndarray: Value of φ(t) = (φJ⁻ᵟ − 1)/δ, where φJ = 1 − (1−t)ᵗʰᵉᵗᵃ.
        """

        t = np.clip(t, 1e-12, 1 - 1e-12)
        log_one_minus_t = _safe_log(1.0 - t)  # ≤0
        # log((1-t)^θ) = θ*log(1-t)
        log_pow = theta * log_one_minus_t
        pow_term = _safe_exp(log_pow)  # (1-t)^θ, in (0,1)
        phiJ = 1.0 - pow_term  # ∈(0,1)
        log_phiJ = _safe_log(phiJ)
        phiJ_neg_delta = _safe_exp(-delta * log_phiJ)
        return (phiJ_neg_delta - 1.0) / delta

    @staticmethod
    def _phi_prime(t, theta, delta):
        """First derivative φ'(t)."""
        t = np.clip(t, 1e-12, 1 - 1e-12)
        log_one_minus_t = _safe_log(1.0 - t)
        log_pow = theta * log_one_minus_t  # log((1-t)^θ)
        pow_term = _safe_exp(log_pow)  # (1-t)^θ
        phiJ = 1.0 - pow_term
        log_phiJ = _safe_log(phiJ)

        # g'(t) for g(t)=phiJ
        g_prime = theta * _safe_exp((theta - 1.0) * log_one_minus_t) * (-1.0)
        # φ'(t) = -phiJ^{-δ-1} * g'(t)
        phi_prime = -_safe_exp((-delta - 1.0) * log_phiJ) * g_prime
        # ensure positivity (numerically phi_prime>0)
        return np.abs(phi_prime)

    @staticmethod
    def _phi_inv(s, theta, delta):
        """
        Compute the inverse φ-transform for the BB7 copula.

        Args:
            s (float or array-like): Transformed variable ≥ 0.
            theta (float): Copula parameter θ.
            delta (float): Copula parameter δ.

        Returns:
            float or numpy.ndarray: Value of φ⁻¹(s).
        """

        s = np.maximum(s, 0.0)
        temp = _safe_pow(1.0 + delta * s, -1.0 / delta)  # (1+δs)^{-1/δ}
        one_minus_temp = 1.0 - temp
        # Return 1 - (1 - temp)^{1/θ}
        return 1.0 - _safe_pow(one_minus_temp, 1.0 / theta)

    @staticmethod
    def _psi(s, theta, delta):
        return BB7Copula._phi_inv(s, theta, delta)

    @staticmethod
    def _psi_prime(s, theta, delta):
        s = np.maximum(s, 0.0)
        # reuse intermediate values for efficiency
        temp = _safe_pow(1.0 + delta * s, -1.0 / delta)
        one_minus_temp = 1.0 - temp
        dtemp_ds = -(1.0 + delta * s) ** (-1.0 / delta - 1.0)
        return (1.0 / theta) * _safe_pow(one_minus_temp, 1.0 / theta - 1.0) * (-dtemp_ds)

    @staticmethod
    def _psi_second(s, theta, delta):
        """Second derivative ψ'' needed for joint density."""
        s = np.maximum(s, 0.0)
        temp = _safe_pow(1.0 + delta * s, -1.0 / delta)
        one_minus_temp = 1.0 - temp
        dtemp_ds = -(1.0 + delta * s) ** (-1.0 / delta - 1.0)
        d2temp_ds2 = (1.0 / delta + 1.0) * delta * (1.0 + delta * s) ** (-1.0 / delta - 2.0)

        term1 = (1.0 / theta) * (1.0 / theta - 1.0) * _safe_pow(one_minus_temp, 1.0 / theta - 2.0) * (dtemp_ds ** 2)
        term2 = (1.0 / theta) * _safe_pow(one_minus_temp, 1.0 / theta - 1.0) * (-d2temp_ds2)
        return term1 + term2

    def get_cdf(self, u, v, param=None):
        """
        Evaluate the BB7 copula cumulative distribution function at (u, v).

        Args:
            u (float or array-like): First uniform margin in (0,1).
            v (float or array-like): Second uniform margin in (0,1).
            param (Sequence[float], optional): Copula parameters (θ, δ). Defaults to self.get_parameters().

        Returns:
            float or numpy.ndarray: CDF value C(u, v).
        """

        if param is None:
            param = self.get_parameters()
        theta, delta = param
        s = self._phi(u, theta, delta) + self._phi(v, theta, delta)
        return self._psi(s, theta, delta)

    def get_pdf(self, u, v, param=None):
        """
        Approximate the BB7 copula probability density function at (u, v) via finite differences.

        Args:
            u (float or array-like): First uniform margin in (0,1).
            v (float or array-like): Second uniform margin in (0,1).
            param (Sequence[float], optional): Copula parameters (θ, δ). Defaults to self.get_parameters().

        Returns:
            float or numpy.ndarray: Approximate PDF c(u, v).
        """

        if param is None:
            param = self.get_parameters()
        theta, delta = param

        # compute φ, φ', ψ''
        phi_u = self._phi(u, theta, delta)
        phi_v = self._phi(v, theta, delta)
        s = phi_u + phi_v
        log_phi_u_prime = _safe_log(self._phi_prime(u, theta, delta))
        log_phi_v_prime = _safe_log(self._phi_prime(v, theta, delta))
        log_psi_second = _safe_log(np.abs(self._psi_second(s, theta, delta)))

        log_pdf = log_psi_second + log_phi_u_prime + log_phi_v_prime
        pdf = _safe_exp(log_pdf)
        return np.maximum(pdf, 0.0)

    # ------------------------------------------------------------------
    # Cached Gauss-Legendre nodes for the Archimedean τ integral
    # ------------------------------------------------------------------
    _GL_N_TAU = 64
    _xi_tau, _wi_tau = __import__(
        'numpy.polynomial.legendre', fromlist=['leggauss']
    ).leggauss(_GL_N_TAU)
    # Map to [1e-6, 1-1e-6]
    _GL_HALF_TAU = 0.5 * (1 - 2e-6)
    _GL_MID_TAU  = 0.5 * (1 + 2e-6 - 1e-6)
    _GL_T_TAU    = _GL_HALF_TAU * _xi_tau + _GL_MID_TAU   # shape (64,)
    _GL_W_TAU    = _GL_HALF_TAU * _wi_tau

    def kendall_tau(self, param=None):
        """
        Kendall's τ via the Archimedean concordance formula.

        BB7 IS Archimedean (generator φ(t) = [(1−(1−t)^θ)^{−δ} − 1]/δ),
        so the exact formula holds:

            τ = 1 + 4·∫₀¹ φ(t)/φ'(t) dt

        where φ'(t) < 0 (decreasing generator).

        Evaluated on a vectorised 64-point Gauss-Legendre grid (~0.5ms).
        For 1 < θ < 2 this also equals the closed form:
            τ = 1 − 2/(δ(2−θ)) + 4/(δθ²)·B(δ+2, 2/θ−1).

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

        T = self._GL_T_TAU  # shape (64,)
        W = self._GL_W_TAU

        phiJ = np.clip(1.0 - (1.0 - T) ** theta, 1e-15, None)
        phi_t     = (phiJ ** (-delta) - 1.0) / delta
        phi_prime = -phiJ ** (-delta - 1) * theta * (1.0 - T) ** (theta - 1)
        # φ'(t) < 0 → integrand < 0 → result < 0 → τ < 1
        integrand = phi_t / phi_prime

        result = float(np.dot(W, integrand))
        return float(np.clip(1.0 + 4.0 * result, 0.0, 1.0 - 1e-15))

    def blomqvist_beta(self, param=None):
        """
        Blomqvist's β = 4·C(½,½) − 1.

        For BB7:  β = 4β* − 1,  where
            β* = 1 − (1 − [2(1−2^{−θ})^{−δ} − 1]^{−1/δ})^{1/θ} = C(½,½).

        (Joe 2014, p.203.)

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
        Draw n samples from the BB7 copula via conditional inversion.

        For each u_i ~ U(0,1) draw w_i ~ U(0,1) and solve
            h(v|u_i) = ∂C/∂u(u_i, v) = w_i  for v ∈ (0,1)
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

    def init_from_data(self, u, v):
        """
        Moment-matching initialisation from pseudo-observations.

        Strategy
        --------
        1. Estimate δ from empirical λ_L = 2^{-1/δ}  → δ = log2 / (−log λ_L)
        2. Estimate θ from empirical λ_U = 2 − 2^{1/θ} → θ = log2 / log(2 − λ_U)
           (gives a λ-based starting point; then refine θ via Kendall τ).

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

        # ---- 1. Estimate δ from empirical λ_L ---------------------------------
        qs = (0.02, 0.04, 0.06, 0.08, 0.10)
        lam_vals = []
        for q in qs:
            uq, vq = np.quantile(u, q), np.quantile(v, q)
            joint = np.mean((u < uq) & (v < vq))
            lam_vals.append(joint / max(q, 1e-9))
        lam_L_emp = float(np.clip(np.median(lam_vals), 1e-6, 0.9999))

        # λ_L = 2^{-1/δ}  →  δ = log2 / (−log λ_L)
        d_try = np.log(2) / (-np.log(lam_L_emp))
        delta0 = float(np.clip(d_try if np.isfinite(d_try) and d_try > 0 else 1.0,
                               1e-3, 300.0))

        # ---- 2. Estimate θ from empirical τ -----------------------------------
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

        theta0 = float(np.clip(theta0, 1e-4, 200.0))
        delta0 = float(np.clip(delta0, 1e-4, 300.0))
        return np.array([theta0, delta0], dtype=float)

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """
        The partial derivative ∂C(u,v)/∂u for the BB7 copula.

        Args:
            u (float or array-like): First uniform margin in (0,1).
            v (float or array-like): Second uniform margin in (0,1).
            param (Sequence[float], optional): Copula parameters (θ, δ). Defaults to self.get_parameters().

        Returns:
            float or numpy.ndarray: Approximate ∂C/∂u at (u, v).
        """

        if param is None:
            param = self.get_parameters()
        theta, delta = param
        # φ(u), φ(v)
        phi_u = self._phi(u, theta, delta)
        phi_v = self._phi(v, theta, delta)
        s = phi_u + phi_v
        # ψ(s)
        psi_val = self._psi(s, theta, delta)
        # φ'(u) and φ'(ψ(s))
        phi_u_prime = self._phi_prime(u, theta, delta)
        phi_psi_prime = self._phi_prime(psi_val, theta, delta)
        return phi_u_prime / phi_psi_prime

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """
        Approximate the partial derivative ∂C(u,v)/∂v for the BB7 copula.

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
        Compute the lower tail dependence coefficient (LTDC) for the BB7 copula.

        Args:
            param (Sequence[float], optional): Copula parameters (θ, δ). Defaults to self.get_parameters().

        Returns:
            float: LTDC value.
        """

        if param is None:
            param = self.get_parameters()
        delta = param[1]
        # 2^(-1/δ) en safe
        return _safe_pow(2.0, -1.0 / delta)

    def UTDC(self, param=None):
        """
        Compute the upper tail dependence coefficient (UTDC) for the BB7 copula.

        Args:
            param (Sequence[float], optional): Copula parameters (θ, δ). Defaults to self.get_parameters().

        Returns:
            float: UTDC value = 2 − lim₍u→1₎ [1 − 2u + C(u,u)]/(1−u).
        """

        if param is None:
            param = self.get_parameters()
        theta = param[0]
        # 2^(1/θ) en safe, puis 2 - …
        two_pow = _safe_pow(2.0, 1.0 / theta)
        return 2.0 - two_pow


    def IAD(self, data):
        """
        Return NaN for the Integrated Anderson-Darling (IAD) statistic for BB7.

        Args:
            data (Sequence[array-like, array-like]): Ignored pseudo-observations.

        Returns:
            float: Always returns numpy.nan.
        """

        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        """
        Return NaN for the Anderson-Darling (AD) statistic for BB7.

        Args:
            data (Sequence[array-like, array-like]): Ignored pseudo-observations.

        Returns:
            float: Always returns numpy.nan.
        """

        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan