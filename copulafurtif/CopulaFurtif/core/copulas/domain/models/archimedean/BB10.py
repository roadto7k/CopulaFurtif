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


class BB10Copula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """
    BB10 Copula (Crowder) Archimedean copula.

    Attributes:
        name (str): Name of the copula.
        type (str): Identifier for the copula family.
        bounds_param (list of tuple): Parameter bounds for optimization.
        parameters (np.ndarray): Copula parameters [theta, delta].
        default_optim_method (str): Optimization method.
    """

    def __init__(self):
        """Initialize the BB10 copula with default parameters."""
        super().__init__()
        self.name = "BB10 Copula"
        self.type = "bb10"
        self.default_optim_method = "Powell"
        self.init_parameters(CopulaParameters(np.array([2.0, 0.5]), [(1e-6, np.inf), (1e-6, 1.0)], ["theta", "pi"]))


    def get_cdf(self, u, v, param=None):
        """Return :math:`C(u,v)=uv\\,[1-\\pi(1-u^{\\theta})(1-v^{\\theta})]^{-1/\\theta}`.

        Args:
            u (float | np.ndarray): First uniform margin *(0, 1)*.
            v (float | np.ndarray): Second uniform margin *(0, 1)*.
            param (Sequence[float], optional):
                Copula parameters ``[theta, pi]``.
                Defaults to current model parameters.

        Returns:
            float | np.ndarray: Copula CDF :math:`C(u,v)`.
        """
        if param is None:
            param = self.get_parameters()
        theta, pi = param

        u = np.clip(u, 1e-12, 1 - 1e-12)
        v = np.clip(v, 1e-12, 1 - 1e-12)

        u_theta = _safe_pow(u, theta)           # u^{θ}
        v_theta = _safe_pow(v, theta)           # v^{θ}
        T = 1.0 - pi * (1.0 - u_theta) * (1.0 - v_theta)
        T = np.clip(T, 1e-12, None)             # positivity guard

        return u * v * _safe_pow(T, -1.0 / theta)

    def get_pdf(self, u, v, param=None):
        r"""Copula density (Joe 2014, §4.26.1):

        .. math::

            c(u,v) = T^{-1/\theta-1}\bigl[1-\pi+\pi(1+\theta)v^{\theta}\bigr]
                     - (1+\theta)\pi v^{\theta}(1-u^{\theta})(1-\pi+\pi v^{\theta})
                       \cdot T^{-1/\theta-2}

        where :math:`T = 1-\pi(1-u^{\theta})(1-v^{\theta})`.

        Equivalently, this equals :math:`\partial^2 C/\partial u\,\partial v`.

        Args:
            u, v (float | np.ndarray): Uniform margins.
            param (Sequence[float], optional): ``[theta, pi]``.

        Returns:
            float | np.ndarray: Joint density :math:`c(u,v)\ge0`.
        """
        if param is None:
            param = self.get_parameters()
        theta, pi = float(param[0]), float(param[1])

        u = np.clip(u, 1e-12, 1 - 1e-12)
        v = np.clip(v, 1e-12, 1 - 1e-12)

        ut = _safe_pow(u, theta)
        vt = _safe_pow(v, theta)
        T  = np.clip(1.0 - pi * (1.0 - ut) * (1.0 - vt), 1e-12, None)

        # ∂/∂v [v(1-π+πv^θ)T^{-1/θ-1}]
        term1 = T ** (-1.0 / theta - 1.0) * (1.0 - pi + pi * (1.0 + theta) * vt)
        term2 = -(1.0 + theta) * pi * vt * (1.0 - ut) * (1.0 - pi + pi * vt) \
                * T ** (-1.0 / theta - 2.0)
        return np.maximum(term1 + term2, 0.0)

    # ------------------------------------------------------------------
    # Cached GL-64 nodes for kendall_tau (class-level, computed at import)
    # ------------------------------------------------------------------
    _GL_N_TAU = 64
    _xi_tau, _wi_tau = __import__(
        'numpy.polynomial.legendre', fromlist=['leggauss']
    ).leggauss(_GL_N_TAU)
    _GL_HALF_TAU = 0.5 * (1.0 - 2e-6)
    _GL_MID_TAU  = 0.5 * (1.0 + 2e-6 - 1e-6)
    _GL_T_TAU    = _GL_HALF_TAU * _xi_tau + _GL_MID_TAU
    _GL_W_TAU    = _GL_HALF_TAU * _wi_tau

    def kendall_tau(self, param=None):
        """
        Kendall's τ via the Archimedean concordance formula.

        BB10 IS Archimedean (Joe 2014, §4.26.1) with inverse generator:
            φ⁻¹(t; θ,π) = log[(1−π)t^{−θ} + π]   (maps (0,1]→[0,∞), = 0 at t=1)
            d/dt φ⁻¹ = −θ(1−π)t^{−θ−1} / [(1−π)t^{−θ}+π]   (< 0)

        τ = 1 + 4·∫₀¹ φ⁻¹(t)/[d/dt φ⁻¹(t)] dt

        Degenerate cases: π=0 → C⊥ (τ=0); π=1 → handled via limit.

        Evaluated on a vectorised 64-point Gauss-Legendre grid (~0.02ms).

        Note: for fixed π, τ is NOT necessarily monotone in θ (Joe 2014, p.207).
        For fixed θ, τ increases with π.

        Parameters
        ----------
        param : [theta, pi], optional

        Returns
        -------
        float ∈ [0, 1)
        """
        if param is None:
            param = self.get_parameters()
        theta, pi = float(param[0]), float(param[1])

        if pi <= 0.0:
            return 0.0
        if pi >= 1.0 - 1e-10:
            pi = 1.0 - 1e-10  # avoid 0/0 in generator

        T  = self._GL_T_TAU
        W  = self._GL_W_TAU

        denom  = np.clip((1.0 - pi) * T ** (-theta) + pi, 1e-300, None)
        phi_t  = _safe_log(denom)                          # ≥ 0 on (0,1)
        phi_p  = (1.0 - pi) * (-theta) * T ** (-theta - 1.0) / denom  # < 0

        result = float(np.dot(W, phi_t / phi_p))
        return float(np.clip(1.0 + 4.0 * result, 0.0, 1.0 - 1e-15))

    def blomqvist_beta(self, param=None):
        """
        Blomqvist's β = 4·C(½,½) − 1.

        Parameters
        ----------
        param : [theta, pi], optional

        Returns
        -------
        float
        """
        if param is None:
            param = self.get_parameters()
        return float(4.0 * self.get_cdf(0.5, 0.5, param) - 1.0)

    def sample(self, n, rng=None):
        """
        Draw n samples via conditional inversion.

        For each u_i ~ U(0,1), draw w_i ~ U(0,1) and solve
            ∂C/∂u(u_i, v) = w_i  for v  using Brentq.

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
        W_uni = rng.uniform(1e-6, 1 - 1e-6, n)
        V = np.empty(n)

        for i in range(n):
            u_i = float(U[i])
            w_i = float(W_uni[i])

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
        Since τ(θ,π) is monotone increasing in π for fixed θ, we can for each
        candidate θ find the unique π(θ) satisfying τ(θ,π)=τ̂ via Brentq.
        Among all (θ, π(θ)) pairs, we select the one that best matches the
        empirical Blomqvist β̂ (which carries additional information about
        the joint distribution).

        Note: BB10 is under-identified by τ alone (non-monotone in θ), so
        β̂ is used as a discriminating second moment.

        Parameters
        ----------
        u, v : array-like, pseudo-observations in (0, 1)

        Returns
        -------
        np.ndarray [theta0, pi0]
        """
        from scipy.optimize import brentq
        from scipy.stats import kendalltau as sp_kendalltau

        u = np.asarray(u, float)
        v = np.asarray(v, float)

        tau_emp_val, _ = sp_kendalltau(u, v)
        tau_emp_val = float(np.clip(tau_emp_val, 0.005, 0.98))

        beta_emp = float(np.clip(
            4.0 * np.mean((u > 0.5) == (v > 0.5)) - 1.0, 0.0, 0.99
        ))

        theta_grid = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
        best_err = np.inf
        theta0, pi0 = 2.0, 0.5

        for th_try in theta_grid:
            # Solve τ(th_try, π) = τ_emp — monotone in π, use brentq
            try:
                f_lo = self.kendall_tau([th_try, 0.01]) - tau_emp_val
                f_hi = self.kendall_tau([th_try, 0.99]) - tau_emp_val
                if f_lo * f_hi > 0:
                    continue  # τ_emp unreachable for this θ
                pi_try = brentq(
                    lambda pi: self.kendall_tau([th_try, pi]) - tau_emp_val,
                    0.01, 0.99, xtol=1e-5, maxiter=30,
                )
            except Exception:
                continue

            # Score by β̂ discrepancy
            try:
                beta_try = 4.0 * float(self.get_cdf(0.5, 0.5, [th_try, pi_try])) - 1.0
                err = abs(beta_try - beta_emp)
            except Exception:
                continue

            if err < best_err:
                best_err = err; theta0 = th_try; pi0 = pi_try

        theta0 = float(np.clip(theta0, 1e-4, 200.0))
        pi0    = float(np.clip(pi0, 1e-4, 1.0 - 1e-6))
        return np.array([theta0, pi0], dtype=float)

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """Compute :math:`\\partial C/\\partial u`.

        Closed form (Joe 2014, §4.26.1):

        .. math::

            \\frac{\\partial C}{\\partial u}
            = v\\,(1-\\pi+\\pi v^{\\theta})\\,
              [1-\\pi(1-u^{\\theta})(1-v^{\\theta})]^{-1/\\theta-1}

        Returns:
            float | np.ndarray: :math:`\\partial C/\\partial u \\in [0,1]`.
        """
        if param is None:
            param = self.get_parameters()
        theta, pi = float(param[0]), float(param[1])

        u = np.clip(u, 1e-12, 1 - 1e-12)
        v = np.clip(v, 1e-12, 1 - 1e-12)

        vt = _safe_pow(v, theta)
        ut = _safe_pow(u, theta)
        T  = np.clip(1.0 - pi * (1.0 - ut) * (1.0 - vt), 1e-12, None)

        return v * (1.0 - pi + pi * vt) * _safe_pow(T, -1.0 / theta - 1.0)

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """
        Approximate the partial derivative ∂C(u,v)/∂v for the BB10 copula.

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
        Compute the lower tail dependence coefficient (LTDC) for the BB10 copula.

        Args:
            param (Sequence[float], optional): Copula parameters (θ, δ). Defaults to self.get_parameters().

        Returns:
            float: LTDC value.
        """

        return 0.0

    def UTDC(self, param=None):
        """
        Compute the upper tail dependence coefficient (UTDC) for the BB10 copula.

        Args:
            param (Sequence[float], optional): Copula parameters (θ, δ). Defaults to self.get_parameters().

        Returns:
            float: UTDC value = 2 − lim₍u→1₎ [1 − 2u + C(u,u)]/(1−u).
        """

        return 0.0


    def IAD(self, data):
        """
        Return NaN for the Integrated Anderson-Darling (IAD) statistic for BB10.

        Args:
            data (Sequence[array-like, array-like]): Ignored pseudo-observations.

        Returns:
            float: Always returns numpy.nan.
        """

        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        """
        Return NaN for the Anderson-Darling (AD) statistic for BB10.

        Args:
            data (Sequence[array-like, array-like]): Ignored pseudo-observations.

        Returns:
            float: Always returns numpy.nan.
        """

        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan