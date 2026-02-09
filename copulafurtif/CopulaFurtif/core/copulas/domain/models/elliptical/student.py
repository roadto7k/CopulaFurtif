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
        self.init_parameters(CopulaParameters(np.array([0.5, 4.0]),[(-0.999, 0.999), (2.01, 30.0)], ["rho", "nu"] ))

    def get_cdf(self, u, v, param=None):
        """Compute copula CDF C(u,v) = P(U≤u, V≤v) for the t-copula.

        Vectorized when possible via scipy.stats.multivariate_t.cdf.
        Falls back to the Gauss–Laguerre mixture-of-normals representation if needed.
        """
        if param is None:
            param = self.get_parameters()
        rho, nu = param
        rho = float(np.clip(rho, -0.999999, 0.999999))
        nu = float(nu)

        u_arr = np.asarray(u, dtype=float)
        v_arr = np.asarray(v, dtype=float)
        scalar = (u_arr.ndim == 0 and v_arr.ndim == 0)

        # Boundary logic (vectorized)
        out = np.zeros(np.broadcast(u_arr, v_arr).shape, dtype=float)
        uu = np.broadcast_to(u_arr, out.shape).copy()
        vv = np.broadcast_to(v_arr, out.shape).copy()

        # Exact boundaries
        mask0 = (uu <= 0.0) | (vv <= 0.0)
        out[mask0] = 0.0

        mask_u1 = (uu >= 1.0) & (~mask0)
        out[mask_u1] = np.clip(vv[mask_u1], 0.0, 1.0)

        mask_v1 = (vv >= 1.0) & (~mask0) & (~mask_u1)
        out[mask_v1] = np.clip(uu[mask_v1], 0.0, 1.0)

        mask_main = (~mask0) & (~mask_u1) & (~mask_v1)
        if np.any(mask_main):
            eps = 1e-12
            uc = np.clip(uu[mask_main], eps, 1.0 - eps)
            vc = np.clip(vv[mask_main], eps, 1.0 - eps)

            x = t.ppf(uc, df=nu)
            y = t.ppf(vc, df=nu)
            pts = np.column_stack([x, y])

            try:
                shape = np.array([[1.0, rho], [rho, 1.0]], dtype=float)
                vals = multivariate_t.cdf(pts, loc=np.zeros(2), shape=shape, df=nu)
                out[mask_main] = vals
            except Exception:
                # Fallback: your Gauss–Laguerre mixture-of-normals quadrature (scalar)
                # Use the same internal logic as before, but per-point.
                k = nu / 2.0
                alpha = k - 1.0
                z_nodes, w_weights = roots_genlaguerre(self.n_nodes, alpha)
                cov = [[1.0, rho], [rho, 1.0]]
                mvn = multivariate_normal(mean=[0.0, 0.0], cov=cov)

                vals = []
                for xi, yi in pts:
                    total = 0.0
                    for zi, wi in zip(z_nodes, w_weights):
                        sx = xi * np.sqrt(2.0 * zi / nu)
                        sy = yi * np.sqrt(2.0 * zi / nu)
                        total += wi * mvn.cdf([sx, sy])
                    vals.append(total / gamma(k))
                out[mask_main] = np.array(vals, dtype=float)

        out = np.clip(out, 0.0, 1.0)
        return float(out) if scalar else out

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

    def sample(self, n, param=None, random_state=None):
        """Generate n samples from the Student copula.

        Returns:
            np.ndarray: shape (n, 2) with pseudo-observations (u,v) in (0,1).
        """
        if param is None:
            param = self.get_parameters()
        rho, nu = param

        rho = float(np.clip(rho, -0.999999, 0.999999))
        nu = float(nu)

        rng = np.random.RandomState(random_state) if random_state is not None else np.random

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

    def init_from_data(self, u, v):
        """
        Robust initialization of Student-t copula parameters [rho, nu].

        Strategy
        --------
        1) rho0 from empirical Kendall's tau:
               tau_hat = kendalltau(u, v)
               rho0 = sin(pi/2 * tau_hat)
        2) nu0 from empirical upper-tail dependence:
               lambda_U_hat = median over q in {0.90,0.92,0.94,0.96,0.98}
                               of  P(U>q, V>q) / (1-q)
           Invert the theoretical t-copula formula for nu with a brentq solver.
           If it fails or tail is ~0, fall back to a small grid over nu.
        3) Local refinement: among a few nu candidates around the current guess,
           pick the one maximizing a fast pseudo log-likelihood with rho=rho0.

        Args
        ----
        u, v : array-like
            Pseudo-observations in (0,1).

        Returns
        -------
        list
            [rho0, nu0] suitable as starting values for MLE/IFM.
        """

        u = np.asarray(u); v = np.asarray(v)

        # -------------------- 1) rho via Kendall tau (closed form) --------------------
        tau_hat, _ = kendalltau(u, v)
        tau_hat = float(np.clip(tau_hat, -0.999, 0.999))
        rho0 = float(np.sin(0.5 * np.pi * tau_hat))

        # bounds
        (rho_lo, rho_hi), (nu_lo, nu_hi) = self.get_bounds()
        rho0 = float(np.clip(rho0, rho_lo + 1e-6, rho_hi - 1e-6))

        # -------------------- 2) empirical tail dependence (upper) --------------------
        def empirical_lambda_u(u, v, qs=(0.90, 0.92, 0.94, 0.96, 0.98)):
            vals = []
            for q in qs:
                qu = np.quantile(u, q)
                qv = np.quantile(v, q)
                joint = np.mean((u > qu) & (v > qv))
                denom = max(1e-9, 1.0 - q)
                vals.append(joint / denom)
            vals = np.array(vals, dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                return 0.0
            vals.sort()
            # light trimming if enough points
            if vals.size >= 5:
                k = max(1, vals.size // 10)
                vals = vals[k:-k] if vals.size - 2 * k >= 1 else vals
            return float(np.median(vals))

        lam_emp = empirical_lambda_u(u, v)
        lam_emp = float(np.clip(lam_emp, 0.0, 0.999))

        # If dependence is weak or rho0 ~ 0, lambda is near 0 anyway -> use large nu
        if abs(rho0) < 0.05 or lam_emp < 1e-3:
            nu_guess = 20.0  # conservative heavy-tail but not extreme
        else:
            # invert lambda(nu; rho0) = lam_emp
            def lambda_of_nu(nu):
                # guard denom 1+rho
                den = max(1e-8, 1.0 + rho0)
                arg = -np.sqrt((nu + 1.0) * (1.0 - rho0) / den)
                return 2.0 * student_t.cdf(arg, df=nu + 1.0)

            # robust bracketing
            a, b = max(nu_lo + 1e-6, 2.01), min(nu_hi - 1e-6, 80.0)
            # try brentq; if it doesn't straddle, we'll grid-search
            nu_guess = None
            try:
                fa = lambda_of_nu(a) - lam_emp
                fb = lambda_of_nu(b) - lam_emp
                if fa * fb < 0:
                    nu_guess = brentq(lambda x: lambda_of_nu(x) - lam_emp, a, b, maxiter=200)
            except Exception:
                nu_guess = None

            if nu_guess is None:
                # coarse grid fallback on |lambda - lam_emp|
                grid = np.array([2.1, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0,
                                 20.0, 30.0, 40.0, 60.0, 80.0], dtype=float)
                grid = grid[(grid >= a) & (grid <= b)]
                diffs = [abs(lambda_of_nu(g) - lam_emp) for g in grid]
                nu_guess = float(grid[int(np.argmin(diffs))]) if len(diffs) else 10.0

        # -------------------- 3) local pseudo-LL refinement around nu_guess ----------
        def pseudo_loglik_nu(nu):
            # fast sum log c(u,v) at fixed rho0, varying nu (no gradients)
            try:
                c = self.get_pdf(u, v, param=[rho0, nu])
                # avoid log(0)
                c = np.maximum(c, 1e-300)
                return float(np.sum(np.log(c)))
            except Exception:
                return -np.inf

        # candidates around nu_guess (log-spaced +/- ~50%)
        cand = np.unique(
            np.clip(
                nu_guess * np.array([0.67, 0.8, 1.0, 1.25, 1.5]),
                nu_lo + 1e-6, nu_hi - 1e-6
            )
        )
        # always include a few safe anchors
        anchors = np.array([4.0, 8.0, 12.0, 20.0], dtype=float)
        cand = np.unique(np.concatenate([cand, anchors]))
        scores = [pseudo_loglik_nu(x) for x in cand]
        nu0 = float(cand[int(np.argmax(scores))]) if len(scores) else float(nu_guess)

        # final clip
        nu0 = float(np.clip(nu0, nu_lo + 1e-6, nu_hi - 1e-6))
        return np.array([rho0, nu0])

