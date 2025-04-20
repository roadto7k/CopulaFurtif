"""
Gaussian Copula implementation following the project coding standard:

Norms:
 1. Use private attribute `_parameters` with public `@property parameters` and validation in the setter.
 2. All methods accept `param: np.ndarray = None` defaulting to `self.parameters`.
 3. Docstrings must include **Parameters** and **Returns** sections with types.
 4. Parameter bounds are defined in `bounds_param`; setter enforces them.
 5. Consistent boundary handling with `eps = 1e-12` and `np.clip`.
"""
import numpy as np
from scipy.special import erfinv
from scipy.stats import norm, multivariate_normal

from Service.Copulas.base import BaseCopula


class GaussianCopula(BaseCopula):
    """
    Gaussian Copula class.

    Attributes
    ----------
    family : str
        Identifier for the copula family, "gaussian".
    name : str
        Human-readable name for output/logging.
    bounds_param : list of tuple
        Bounds for the copula parameter rho, in (-1,1).
    parameters : np.ndarray
        Copula parameter [rho].
    """

    def __init__(self):
        super().__init__()
        self.type = "gaussian"
        self.name = "Gaussian Copula"
        self.bounds_param = [(-0.999, 0.999)]
        self._parameters = np.array([0.0])
        self.default_optim_method = "SLSQP"

    @property
    def parameters(self) -> np.ndarray:
        """
        Get the copula parameters.

        Returns
        -------
        np.ndarray
            Current copula parameter [rho].
        """
        return self._parameters

    @parameters.setter
    def parameters(self, param: np.ndarray):
        """
        Set and validate copula parameters against bounds_param.

        Parameters
        ----------
        param : array-like
            New copula parameters [rho].

        Raises
        ------
        ValueError
            If parameter is outside its specified bound.
        """
        param = np.asarray(param)
        for idx, (lower, upper) in enumerate(self.bounds_param):
            val = param[idx]
            if lower is not None and val <= lower:
                raise ValueError(f"Parameter at index {idx} must be > {lower}, got {val}")
            if upper is not None and val >= upper:
                raise ValueError(f"Parameter at index {idx} must be < {upper}, got {val}")
        self._parameters = param

    def get_cdf(self, u, v, param: np.ndarray = None):
        """
        Compute the Gaussian copula CDF C(u,v).

        Parameters
        ----------
        u : float or array-like
            Pseudo-observations in (0,1).
        v : float or array-like
            Pseudo-observations in (0,1).
        param : ndarray, optional
            Copula parameter [rho]. If None, uses self.parameters.

        Returns
        -------
        float or np.ndarray
            CDF value(s) at (u, v).
        """
        if param is None:
            param = self.parameters
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
        Compute the Gaussian copula PDF c(u,v).

        Parameters
        ----------
        u : float or array-like
            Pseudo-observations in (0,1).
        v : float or array-like
            Pseudo-observations in (0,1).
        param : ndarray, optional
            Copula parameter [rho]. If None, uses self.parameters.

        Returns
        -------
        float or np.ndarray
            PDF value(s) at (u, v).
        """
        if param is None:
            param = self.parameters
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
        Compute Kendall's tau = (2/π) · arcsin(rho).

        Parameters
        ----------
        param : ndarray, optional
            Copula parameter [rho]. If None, uses self.parameters.

        Returns
        -------
        float
            Kendall's tau.
        """
        if param is None:
            param = self.parameters
        rho = param[0]
        return (2.0 / np.pi) * np.arcsin(rho)

    def sample(self, n: int, param: np.ndarray = None) -> np.ndarray:
        """
        Generate n samples from the Gaussian copula.

        Parameters
        ----------
        n : int
            Number of samples.
        param : ndarray, optional
            Copula parameter [rho]. If None, uses self.parameters.

        Returns
        -------
        np.ndarray
            Shape (n,2) array of pseudo-observations (u, v).
        """
        if param is None:
            param = self.parameters
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
        Lower tail dependence coefficient (always 0 for Gaussian).

        Returns
        -------
        float
            0.0
        """
        return 0.0

    def UTDC(self, param: np.ndarray = None) -> float:
        """
        Upper tail dependence coefficient (always 0 for Gaussian).

        Returns
        -------
        float
            0.0
        """
        return 0.0

    def IAD(self, data) -> float:
        """
        Integrated absolute deviation (disabled for elliptical copulas).

        Returns
        -------
        float
            np.nan
        """
        print(f"[INFO] IAD is disabled for {self.name} due to performance limitations.")
        return np.nan

    def AD(self, data) -> float:
        """
        Anderson–Darling test statistic (disabled for elliptical copulas).

        Returns
        -------
        float
            np.nan
        """
        print(f"[INFO] AD is disabled for {self.name} due to performance limitations.")
        return np.nan

    def partial_derivative_C_wrt_u(self, u, v, param: np.ndarray = None):
        """
        Compute ∂C(u,v)/∂u for conditional CDF.

        Parameters
        ----------
        u : float or array-like
            Values in (0,1).
        v : float or array-like
            Values in (0,1).
        param : ndarray, optional
            Copula parameter [rho].

        Returns
        -------
        float or np.ndarray
            ∂C/∂u.
        """
        if param is None:
            param = self.parameters
        rho = param[0]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        x = norm.ppf(u)
        y = norm.ppf(v)
        return norm.cdf((y - rho * x) / np.sqrt(1 - rho**2))

    def partial_derivative_C_wrt_v(self, u, v, param: np.ndarray = None):
        """
        Compute ∂C(u,v)/∂v for conditional CDF.

        Parameters
        ----------
        u : float or array-like
            Values in (0,1).
        v : float or array-like
            Values in (0,1).
        param : ndarray, optional
            Copula parameter [rho].

        Returns
        -------
        float or np.ndarray
            ∂C/∂v.
        """
        if param is None:
            param = self.parameters
        rho = param[0]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        x = norm.ppf(u)
        y = norm.ppf(v)
        return norm.cdf((x - rho * y) / np.sqrt(1 - rho**2))

    def conditional_cdf_u_given_v(self, u, v, param: np.ndarray = None):
        """
        Compute P(U ≤ u | V = v).

        Parameters
        ----------
        u : float or array-like
            Threshold for U.
        v : float or array-like
            Given value for V.
        param : ndarray, optional
            Copula parameter [rho].

        Returns
        -------
        float or np.ndarray
            Conditional CDF P(U ≤ u | V = v).
        """
        if param is None:
            param = self.parameters
        partial = self.partial_derivative_C_wrt_v(u, v, param)
        eps = 1e-12
        v = np.clip(v, eps, 1 - eps)
        y = norm.ppf(v)
        return partial / norm.pdf(y)

    def conditional_cdf_v_given_u(self, u, v, param: np.ndarray = None):
        """
        Compute P(V ≤ v | U = u).

        Parameters
        ----------
        u : float or array-like
            Given value for U.
        v : float or array-like
            Threshold for V.
        param : ndarray, optional
            Copula parameter [rho].

        Returns
        -------
        float or np.ndarray
            Conditional CDF P(V ≤ v | U = u).
        """
        if param is None:
            param = self.parameters
        partial = self.partial_derivative_C_wrt_u(u, v, param)
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        x = norm.ppf(u)
        return partial / norm.pdf(x)
