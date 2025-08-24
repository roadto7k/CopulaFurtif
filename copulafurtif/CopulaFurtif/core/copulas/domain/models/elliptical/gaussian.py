"""
Gaussian Copula implementation following the project coding standard.

Norms:
 1. Use private attribute `_parameters` with public `@property parameters` and validation in the setter.
 2. All methods accept `param: np.ndarray = None` defaulting to `self.parameters`.
 3. Docstrings must include **Args** and **Returns** sections with types.
 4. Parameter bounds are defined in `bounds_param`; setter enforces them.
 5. Consistent boundary handling with `eps = 1e-12` and `np.clip`.

This module implements the Gaussian copula, supporting evaluation of CDF, PDF,
sampling, Kendall's tau, and conditional distributions. It also supports
tail dependence structure (always zero for Gaussian copula), and integrates
with the CopulaFurtif model selection and evaluation pipeline.
"""

import numpy as np
from scipy.special import erfinv
from scipy.stats import norm, multivariate_normal, kendalltau

from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel, CopulaParameters
from CopulaFurtif.core.copulas.domain.models.mixins import SupportsTailDependence, ModelSelectionMixin

class GaussianCopula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """
    Gaussian Copula model.

    Attributes:
        type (str): Copula type identifier.
        name (str): Human-readable copula name.
        bounds_param (list of tuple): Bounds for the copula parameter rho in (-1, 1).
        _parameters (CopulaParameters): Internal parameter [rho].
        default_optim_method (str): Default optimization method to use during fitting.
    """

    def __init__(self):
        """Initialize the Gaussian Copula with default parameters and bounds."""
        super().__init__()
        self.name = "Gaussian Copula"
        self.type = "gaussian"
        self.default_optim_method = "Powell"
        self.init_parameters(CopulaParameters([0.8], [(-0.99, 0.99)], ["rho"]))


    def get_cdf(self, u, v, param: np.ndarray = None):
        """
        Compute the Gaussian copula CDF C(u, v).

        Args:
            u (float or array-like): Pseudo-observations in (0, 1).
            v (float or array-like): Pseudo-observations in (0, 1).
            param (np.ndarray, optional): Copula parameter [rho]. Defaults to self.parameters.

        Returns:
            float or np.ndarray: CDF values at (u, v).
        """
        if param is None:
            param = self.get_parameters()
        rho = param[0]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        y1 = norm.ppf(u)
        y2 = norm.ppf(v)
        cov = [[1.0, rho], [rho, 1.0]]

        if np.isscalar(y1) and np.isscalar(y2):
            return multivariate_normal.cdf([y1, y2], mean=[0.0, 0.0], cov=cov)
        else:
            return np.array([
                multivariate_normal.cdf([a, b], mean=[0.0, 0.0], cov=cov)
                for a, b in zip(y1, y2)
            ])

    def get_pdf(self, u, v, param: np.ndarray = None):
        """
        Compute the Gaussian copula PDF c(u, v).

        Args:
            u (float or array-like): Pseudo-observations in (0, 1).
            v (float or array-like): Pseudo-observations in (0, 1).
            param (np.ndarray, optional): Copula parameter [rho]. Defaults to self.parameters.

        Returns:
            float or np.ndarray: PDF values at (u, v).
        """
        if param is None:
            param = self.get_parameters()
        rho = param[0]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        x = np.sqrt(2) * erfinv(2 * u - 1)
        y = np.sqrt(2) * erfinv(2 * v - 1)
        det = 1 - rho**2
        exponent = -((x**2 + y**2) * rho**2 - 2 * x * y * rho) / (2 * det)
        return (1.0 / np.sqrt(det)) * np.exp(exponent)

    def kendall_tau(self, param: np.ndarray = None) -> float:
        """
        Compute Kendall's tau = (2/π) * arcsin(rho).

        Args:
            param (np.ndarray, optional): Copula parameter [rho]. Defaults to self.parameters.

        Returns:
            float: Kendall's tau.
        """
        if param is None:
            param = self.get_parameters()
        rho = param[0]
        return (2.0 / np.pi) * np.arcsin(rho)

    def sample(self, n: int, param: np.ndarray = None) -> np.ndarray:
        """
        Generate random samples from the Gaussian copula.

        Args:
            n (int): Number of samples to generate.
            param (np.ndarray, optional): Copula parameter [rho]. Defaults to self.parameters.

        Returns:
            np.ndarray: Array of shape (n, 2) of samples (u, v).
        """
        if param is None:
            param = self.get_parameters()
        rho = param[0]
        cov = np.array([[1.0, rho], [rho, 1.0]])
        L = np.linalg.cholesky(cov)

        z = np.random.randn(n, 2)
        corr = z @ L.T
        u = norm.cdf(corr[:, 0])
        v = norm.cdf(corr[:, 1])
        return np.column_stack((u, v))

    def LTDC(self, param: np.ndarray = None) -> float:
        """
        Compute the lower tail dependence coefficient (always 0 for Gaussian copula).

        Args:
            param (np.ndarray, optional): Copula parameter. Unused here.

        Returns:
            float: 0.0
        """
        return 0.0

    def UTDC(self, param: np.ndarray = None) -> float:
        """
        Compute the upper tail dependence coefficient (always 0 for Gaussian copula).

        Args:
            param (np.ndarray, optional): Copula parameter. Unused here.

        Returns:
            float: 0.0
        """
        return 0.0

    def IAD(self, data) -> float:
        """
        Integrated Absolute Deviation (disabled for Gaussian copula).

        Args:
            data (array-like): Input data (ignored).

        Returns:
            float: NaN, as metric is not implemented.
        """
        print(f"[INFO] IAD is disabled for {self.name} due to performance limitations.")
        return np.nan

    def AD(self, data) -> float:
        """
        Anderson–Darling statistic (disabled for Gaussian copula).

        Args:
            data (array-like): Input data (ignored).

        Returns:
            float: NaN, as metric is not implemented.
        """
        print(f"[INFO] AD is disabled for {self.name} due to performance limitations.")
        return np.nan

    def partial_derivative_C_wrt_u(self, u, v, param: np.ndarray = None):
        """
        Compute ∂C(u, v)/∂u = P(V ≤ v | U = u).

        Args:
            u (float or array-like): Value(s) for U.
            v (float or array-like): Value(s) for V.
            param (np.ndarray, optional): Copula parameter [rho]. Defaults to self.parameters.

        Returns:
            float or np.ndarray: Conditional CDF values.
        """
        if param is None:
            param = self.get_parameters()
        u = np.clip(u, 1e-12, 1 - 1e-12)
        v = np.clip(v, 1e-12, 1 - 1e-12)
        x, y = norm.ppf(u), norm.ppf(v)
        return norm.cdf((y - param[0] * x) / np.sqrt(1 - param[0]**2))

    def partial_derivative_C_wrt_v(self, u, v, param: np.ndarray = None):
        """
        Compute ∂C(u, v)/∂v = P(U ≤ u | V = v) via symmetry.

        Args:
            u (float or array-like): Value(s) for U.
            v (float or array-like): Value(s) for V.
            param (np.ndarray, optional): Copula parameter [rho]. Defaults to self.parameters.

        Returns:
            float or np.ndarray: Conditional CDF values.
        """
        return self.partial_derivative_C_wrt_u(v, u, param)

    def init_from_data(self, u, v):
        """
        Initialize Gaussian copula parameter (rho) from pseudo-observations.

        Strategy
        --------
        - Compute empirical Kendall's tau.
        - Convert tau -> rho via closed-form inversion:
              rho = sin(pi/2 * tau_emp)

        Args:
            u (array-like): First margin pseudo-observations in (0,1).
            v (array-like): Second margin pseudo-observations in (0,1).

        Returns:
            float: Initial guess for rho.
        """

        u, v = np.asarray(u), np.asarray(v)

        tau_emp, _ = kendalltau(u, v)
        tau_emp = np.clip(tau_emp, -0.999, 0.999)

        rho0 = np.sin(0.5 * np.pi * tau_emp)

        # clip to parameter bounds
        low, high = self.get_bounds()[0]
        rho0 = np.clip(rho0, low + 1e-6, high - 1e-6)

        return np.array([rho0])