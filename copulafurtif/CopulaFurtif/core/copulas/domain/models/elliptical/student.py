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
from math import sqrt
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
        self.bounds_param = [(-0.999, 0.999), (2.01, 30.0)] 
        self.param_names = ["rho", "nu"]
        self.parameters = [0.5, 4.0] 
        self.default_optim_method = "SLSQP"
        self.n_nodes = 64

    def get_cdf(self, u, v, param=None):
        """Numerically compute the CDF C(u,v) using Gauss-Laguerre quadrature.

        Args:
            u (float): Pseudo-observation in (0,1).
            v (float): Pseudo-observation in (0,1).
            param (np.ndarray, optional): [rho, nu]

        Returns:
            float: Value of the CDF at (u, v).
        """
        if param is None:
            param = self.parameters
        rho, nu = param
        if u <= 0 or v <= 0:
            return 0.0
        if u >= 1 and v >= 1:
            return 1.0
        if u >= 1:
            return v
        if v >= 1:
            return u

        x = t.ppf(u, df=nu)
        y = t.ppf(v, df=nu)
        k = nu / 2.0
        alpha = k - 1.0
        z_nodes, w_weights = roots_genlaguerre(self.n_nodes, alpha)
        cov = [[1.0, rho], [rho, 1.0]]
        mvn = multivariate_normal(mean=[0.0, 0.0], cov=cov)
        total = sum(wi * mvn.cdf([x * sqrt(2.0 * zi / nu), y * sqrt(2.0 * zi / nu)])
                    for zi, wi in zip(z_nodes, w_weights))
        return total / gamma(k)

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
            param = self.parameters
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

    def sample(self, n, param=None):
        """Generate n samples from the Student copula.

        Args:
            n (int): Number of samples.
            param (np.ndarray, optional): [rho, nu]

        Returns:
            np.ndarray: Samples of shape (n, 2).
        """
        if param is None:
            param = self.parameters
        rho, nu = param
        cov = np.array([[1.0, rho], [rho, 1.0]])
        L = np.linalg.cholesky(cov)
        z = np.random.standard_normal((n, 2))
        chi2 = np.random.chisquare(df=nu, size=n)
        scaled = (z @ L.T) / np.sqrt((chi2 / nu)[:, None])
        u = t.cdf(scaled[:, 0], df=nu)
        v = t.cdf(scaled[:, 1], df=nu)
        return np.column_stack((u, v))

    def kendall_tau(self, param=None, n_samples=10000, random_state=None):
        """Estimate Kendall's tau via sampling.

        Args:
            param (np.ndarray, optional): [rho, nu]
            n_samples (int): Number of Monte Carlo samples.
            random_state (int, optional): Seed for reproducibility.

        Returns:
            float: Estimated Kendall's tau.
        """
        if param is None:
            param = self.parameters
        rho, nu = param
        rng = np.random.RandomState(random_state) if random_state is not None else np.random
        cov = np.array([[1.0, rho], [rho, 1.0]])
        L = np.linalg.cholesky(cov)
        z = rng.standard_normal((n_samples, 2))
        chi2 = rng.chisquare(df=nu, size=n_samples)
        scaled = (z @ L.T) / np.sqrt((chi2 / nu)[:, None])
        u = t.cdf(scaled[:, 0], df=nu)
        v = t.cdf(scaled[:, 1], df=nu)
        tau, _ = kendalltau(u, v)
        return tau

    def LTDC(self, param=None):
        """Lower Tail Dependence Coefficient.

        Args:
            param (np.ndarray, optional): [rho, nu]

        Returns:
            float: LTDC value
        """
        if param is None:
            param = self.parameters
        rho, nu = param
        return 2 * t.cdf(-sqrt((nu + 1) * (1 - rho) / (1 + rho)), df=nu + 1)

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
        """Compute ∂C(u,v)/∂v = P(U ≤ u | V = v).

        Args:
            u (float): Pseudo-observation.
            v (float): Pseudo-observation.
            param (np.ndarray, optional): [rho, nu]

        Returns:
            float: Conditional CDF value
        """
        if param is None:
            rho, nu = self.parameters
        else:
            rho, nu = param

        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        tx = t.ppf(u, df=nu)
        ty = t.ppf(v, df=nu)
        df_c = nu + 1
        scale = sqrt((1 - rho**2) * (nu + ty**2) / df_c)
        loc = rho * ty
        z = (tx - loc) / scale
        return t.pdf(z, df=df_c) / scale

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """Compute ∂C(u,v)/∂u using symmetry.

        Args:
            u (float): Pseudo-observation.
            v (float): Pseudo-observation.
            param (np.ndarray, optional): [rho, nu]

        Returns:
            float: Conditional CDF value
        """
        return self.partial_derivative_C_wrt_v(v, u, param)

    def conditional_cdf_u_given_v(self, u, v, param=None):
        """Compute conditional CDF P(U ≤ u | V = v).

        Args:
            u (float): Pseudo-observation.
            v (float): Pseudo-observation.
            param (np.ndarray, optional): [rho, nu]

        Returns:
            float: Conditional CDF value
        """
        if param is None:
            rho, nu = self.parameters
        else:
            rho, nu = param

        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        tx = t.ppf(u, df=nu)
        ty = t.ppf(v, df=nu)

        df_c = nu + 1
        loc_c = rho * ty
        scale_c = sqrt((nu + ty**2) * (1 - rho**2) / df_c)

        return t.cdf((tx - loc_c) / scale_c, df=df_c)

    def conditional_cdf_v_given_u(self, u, v, param=None):
        """Compute conditional CDF P(V ≤ v | U = u).

        Args:
            u (float): Pseudo-observation.
            v (float): Pseudo-observation.
            param (np.ndarray, optional): [rho, nu]

        Returns:
            float: Conditional CDF value
        """
        return self.conditional_cdf_u_given_v(v, u, param)
