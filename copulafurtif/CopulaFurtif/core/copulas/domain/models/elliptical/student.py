"""
Student Copula implementation for bivariate dependence modeling.

The Student copula extends the Gaussian copula by adding degrees of freedom, 
allowing for heavy tails and more flexible dependence structure, especially in the tails.

Key Features:
- Parametrized by correlation (rho) and degrees of freedom (nu)
- Supports sampling, CDF/PDF evaluation, tail dependence, and conditional distributions
- Suitable for heavy-tailed joint modeling

Attributes:
    name (str): Human-readable name.
    type (str): Identifier for the copula type.
    bounds_param (list of tuple): Bounds for rho and nu.
    parameters (np.ndarray): Copula parameters [rho, nu].
    default_optim_method (str): Default method used for fitting.
    n_nodes (int): Number of nodes for numerical integration (CDF approximation).
"""

import numpy as np
from scipy.stats import kendalltau, t as student_t
from scipy.optimize import brentq
from scipy.stats import t, multivariate_t, kendalltau, multivariate_normal
from scipy.special import gammaln, gamma, roots_genlaguerre
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel, CopulaParameters
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


class StudentCopula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """Student (t) Copula model."""

    def __init__(self):
        """Initialize the Student copula with default parameters and bounds."""
        super().__init__()
        self.name = "Student Copula"
        self.type = "student"
        # self.bounds_param = [(-0.999, 0.999), (2.01, 30.0)] 
        # self.param_names = ["rho", "nu"]
        # self.get_parameters() = [0.5, 4.0]
        self.default_optim_method = "SLSQP"
        self.n_nodes = 64
        self.init_parameters(CopulaParameters(np.array([0.5, 4.0]),[(-1, 1), (2.01, 30.0)], ["rho", "nu"] ))

    def get_cdf(self, u, v, param=None):
        """
        Student-t copula CDF on the open square (0,1)^2:
            C(u,v) = t_{ν,ρ}( t_ν^{-1}(u), t_ν^{-1}(v) )

        Convention:
          - Always clip u,v to (eps, 1-eps) (open interval), no explicit boundary handling.
          - Pairwise vectorization only: u and v must have the same shape.
        """

        if param is None:
            param = self.get_parameters()
        rho, nu = float(param[0]), float(param[1])

        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)

        # pairwise vectorization contract
        if u.ndim == 0 and v.ndim == 0:
            is_scalar = True
        else:
            is_scalar = False
            if u.shape != v.shape:
                raise ValueError("Vectorized evaluation is pairwise: u and v must have the same shape.")

        eps = 1e-12
        u = np.clip(u, eps, 1.0 - eps)
        v = np.clip(v, eps, 1.0 - eps)

        if abs(rho) < 1e-12:
            out = u * v
            return float(out) if is_scalar else out

        x = t.ppf(u, df=nu)
        y = t.ppf(v, df=nu)

        pts = np.column_stack([x.ravel(), y.ravel()])  # (n,2)
        shape = np.array([[1.0, rho], [rho, 1.0]], dtype=float)

        cdf_vals = multivariate_t.cdf(pts, loc=np.zeros(2), shape=shape, df=nu)

        if is_scalar:
            return float(cdf_vals)
        return np.asarray(cdf_vals, dtype=float).reshape(u.shape)

    def get_pdf(self, u, v, param=None):
        """Compute the joint PDF c(u,v) of the Student copula.

        Args:
            u (float or np.ndarray): Pseudo-observations.
            v (float or np.ndarray): Pseudo-observations.
            param (np.ndarray, optional): [rho, nu]

        Returns:
            float or np.ndarray: PDF values at (u, v)
        """
        if param is None:
            param = self.get_parameters()
        rho, nu = param

        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        if abs(rho) < 1e-12:
            u = np.asarray(u, float);
            v = np.asarray(v, float)
            u, v = np.broadcast_arrays(u, v)
            out = np.ones_like(u, dtype=float)
            return float(out) if out.shape == () else out

        u_q = t.ppf(u, df=nu)
        v_q = t.ppf(v, df=nu)
        det = 1 - rho**2
        quad_form = (u_q**2 - 2 * rho * u_q * v_q + v_q**2) / det
        log_num = gammaln((nu + 2) / 2) + gammaln(nu / 2)
        log_den = 2 * gammaln((nu + 1) / 2)
        log_det = 0.5 * np.log(det)
        log_prod = ((nu + 1) / 2) * (np.log1p(u_q**2 / nu) + np.log1p(v_q**2 / nu))
        log_dent = ((nu + 2) / 2) * np.log1p(quad_form / nu)
        log_c = log_num - log_den - log_det + log_prod - log_dent
        return np.exp(log_c)

    def sample(self, n, param=None, random_state=None, rng=None):
        """Generate n samples from the Student copula.

        Returns:
            np.ndarray: shape (n, 2) with pseudo-observations (u,v) in (0,1).
        """
        if param is None:
            param = self.get_parameters()
        rho, nu = param

        rho = float(np.clip(rho, -0.999999, 0.999999))
        nu = float(nu)

        if rng is None:
            rng = np.random.default_rng()

        cov = np.array([[1.0, rho], [rho, 1.0]], dtype=float)
        L = np.linalg.cholesky(cov)

        z = rng.standard_normal((n, 2))
        chi2 = rng.chisquare(df=nu, size=n)
        scaled = (z @ L.T) / np.sqrt((chi2 / nu)[:, None])

        u = t.cdf(scaled[:, 0], df=nu)
        v = t.cdf(scaled[:, 1], df=nu)

        eps = 1e-12
        u = np.clip(u, eps, 1.0 - eps)
        v = np.clip(v, eps, 1.0 - eps)
        return np.column_stack((u, v))

    def kendall_tau(self, param=None, **_kwargs):
        """Exact Kendall's tau for elliptical copulas (Gaussian / Student).

        tau = 2/pi * arcsin(rho)
        """
        if param is None:
            param = self.get_parameters()
        rho, _nu = param
        rho = float(np.clip(rho, -0.999999, 0.999999))
        return float((2.0 / np.pi) * np.arcsin(rho))

    def LTDC(self, param=None):
        """Lower Tail Dependence Coefficient.

        Args:
            param (np.ndarray, optional): [rho, nu]

        Returns:
            float: LTDC value
        """
        if param is None:
            param = self.get_parameters()
        rho, nu = param
        return 2 * t.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), df=nu + 1)

    def UTDC(self, param=None):
        """Upper Tail Dependence Coefficient (same as LTDC for Student copula).

        Args:
            param (np.ndarray, optional): [rho, nu]

        Returns:
            float: UTDC value
        """
        return self.LTDC(param)

    def IAD(self, data):
        """Integrated Absolute Deviation (disabled for Student copula).

        Args:
            data (array-like): Ignored.

        Returns:
            float: np.nan
        """
        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        """Anderson–Darling statistic (disabled for Student copula).

        Args:
            data (array-like): Ignored.

        Returns:
            float: np.nan
        """
        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """∂C(u,v)/∂v = P(U ≤ u | V = v) (must be a CDF in [0,1])."""
        return self.conditional_cdf_u_given_v(u, v, param)

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """∂C(u,v)/∂u = P(V ≤ v | U = u) (must be a CDF in [0,1])."""
        return self.conditional_cdf_v_given_u(u, v, param)

    def conditional_cdf_u_given_v(self, u, v, param=None):
        """P(U ≤ u | V = v) for the t-copula.

        Uses the known closed-form conditional for multivariate t:
        T1 | T2=ty ~ t_{nu+1}(loc=rho*ty, scale=sqrt((nu+ty^2)*(1-rho^2)/(nu+1))).
        """
        if param is None:
            param = self.get_parameters()
        rho, nu = param
        rho = float(np.clip(rho, -0.999999, 0.999999))
        nu = float(nu)

        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)

        eps = 1e-12
        u = np.clip(u, eps, 1.0 - eps)
        v = np.clip(v, eps, 1.0 - eps)

        if abs(rho) < 1e-12:
            u = np.asarray(u, float)
            v = np.asarray(v, float)
            u, v = np.broadcast_arrays(u, v)
            out = u
            return float(out) if out.shape == () else out

        tx = t.ppf(u, df=nu)
        ty = t.ppf(v, df=nu)

        df_c = nu + 1.0
        loc_c = rho * ty
        scale_c = np.sqrt((nu + ty ** 2) * (1.0 - rho ** 2) / df_c)

        z = (tx - loc_c) / scale_c
        out = t.cdf(z, df=df_c)
        return np.clip(out, 1e-12, 1.0 - 1e-12)

    def conditional_cdf_v_given_u(self, u, v, param=None):
        """P(V ≤ v | U = u). Symmetry holds for the bivariate t-copula."""
        return self.conditional_cdf_u_given_v(v, u, param)

    def blomqvist_beta(self, param=None) -> float:
        """
        Fast closed-form Blomqvist beta for elliptical copulas (Gaussian/Student):
            β = (2/π) * arcsin(rho)
        Independent of nu.
        """
        if param is None:
            param = self.get_parameters()
        rho, _nu = param
        rho = float(np.clip(rho, -0.999999, 0.999999))
        return float((2.0 / np.pi) * np.arcsin(rho))

    def init_from_data(self, u, v):
        """
        Robust initialization of Student-t copula parameters [rho, nu].

        Same strategy as before, but faster:
        - empirical tail dependence uses thresholds q directly (u,v are pseudo-obs ~ U(0,1))
        - refinement uses log-density directly (no exp->log roundtrip)
        """
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)

        # 1) rho via Kendall tau
        tau_hat, _ = kendalltau(u, v)
        tau_hat = float(np.clip(tau_hat, -0.999, 0.999))
        rho0 = float(np.sin(0.5 * np.pi * tau_hat))

        (rho_lo, rho_hi), (nu_lo, nu_hi) = self.get_bounds()
        rho0 = float(np.clip(rho0, rho_lo + 1e-6, rho_hi - 1e-6))

        # 2) empirical upper-tail dependence (fast)
        qs = np.array([0.90, 0.92, 0.94, 0.96, 0.98], dtype=float)
        # since u,v are pseudo-observations, use q directly (no quantile calls)
        joint = ((u[:, None] > qs) & (v[:, None] > qs)).mean(axis=0)
        lam_vals = joint / np.maximum(1e-9, 1.0 - qs)
        lam_vals = lam_vals[np.isfinite(lam_vals)]
        lam_emp = float(np.clip(np.median(lam_vals) if lam_vals.size else 0.0, 0.0, 0.999))

        if abs(rho0) < 0.05 or lam_emp < 1e-3:
            nu_guess = 20.0
        else:
            def lambda_of_nu(nu):
                den = max(1e-8, 1.0 + rho0)
                arg = -np.sqrt((nu + 1.0) * (1.0 - rho0) / den)
                return 2.0 * student_t.cdf(arg, df=nu + 1.0)

            a = max(nu_lo + 1e-6, 2.01)
            b = nu_hi - 1e-6

            nu_guess = None
            try:
                fa = lambda_of_nu(a) - lam_emp
                fb = lambda_of_nu(b) - lam_emp
                if fa * fb < 0:
                    nu_guess = brentq(lambda x: lambda_of_nu(x) - lam_emp, a, b, maxiter=200)
            except Exception:
                nu_guess = None

            if nu_guess is None:
                grid = np.array([2.1, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 25.0, 30.0], dtype=float)
                grid = grid[(grid >= a) & (grid <= b)]
                vals = np.array([lambda_of_nu(g) for g in grid], dtype=float)
                nu_guess = float(grid[int(np.argmin(np.abs(vals - lam_emp)))]) if grid.size else 10.0

        # 3) local refinement around nu_guess using log-density directly
        def log_copula_density(u_, v_, rho_, nu_):
            eps = 1e-12
            u_ = np.clip(u_, eps, 1.0 - eps)
            v_ = np.clip(v_, eps, 1.0 - eps)

            u_q = t.ppf(u_, df=nu_)
            v_q = t.ppf(v_, df=nu_)

            det = 1.0 - rho_ * rho_
            det = max(det, 1e-12)

            quad = (u_q * u_q - 2.0 * rho_ * u_q * v_q + v_q * v_q) / det

            log_num = gammaln((nu_ + 2.0) / 2.0) + gammaln(nu_ / 2.0)
            log_den = 2.0 * gammaln((nu_ + 1.0) / 2.0)
            log_det = 0.5 * np.log(det)

            log_prod = ((nu_ + 1.0) / 2.0) * (np.log1p(u_q * u_q / nu_) + np.log1p(v_q * v_q / nu_))
            log_dent = ((nu_ + 2.0) / 2.0) * np.log1p(quad / nu_)

            return (log_num - log_den - log_det + log_prod - log_dent)

        def pseudo_loglik_nu(nu_):
            try:
                return float(np.sum(log_copula_density(u, v, rho0, float(nu_))))
            except Exception:
                return -np.inf

        cand = np.unique(np.clip(nu_guess * np.array([0.67, 0.8, 1.0, 1.25, 1.5]),
                                 nu_lo + 1e-6, nu_hi - 1e-6))
        anchors = np.array([4.0, 8.0, 12.0, 20.0], dtype=float)
        cand = np.unique(np.concatenate([cand, anchors]))

        scores = np.array([pseudo_loglik_nu(x) for x in cand], dtype=float)
        nu0 = float(cand[int(np.argmax(scores))]) if cand.size else float(nu_guess)
        nu0 = float(np.clip(nu0, nu_lo + 1e-6, nu_hi - 1e-6))

        return np.array([rho0, nu0])

