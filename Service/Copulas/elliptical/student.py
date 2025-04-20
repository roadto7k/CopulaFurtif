"""
Student-t Copula implementation following the project coding standard:

Norms:
 1. Use private attribute `_parameters` with public `@property parameters` and validation in the setter.
 2. All methods accept `param: np.ndarray = None` defaulting to `self.parameters`.
 3. Docstrings must include **Parameters** and **Returns** sections with types.
 4. Parameter bounds are defined in `bounds_param`; setter enforces them.
 5. Consistent boundary handling with `eps=1e-12` and `np.clip`.
"""
import numpy as np
from math import sqrt
from scipy.integrate import quad
from scipy.special import gammaln, roots_genlaguerre, gamma
from scipy.stats import multivariate_normal, t, kendalltau

from Service.Copulas.base import BaseCopula


class StudentCopula(BaseCopula):
    """
    Student-t Copula class.

    Attributes
    ----------
    family : str
        Identifier for the copula family, "student".
    name : str
        Human-readable name for output/logging.
    bounds_param : list of tuple
        Bounds for parameters [rho, nu], with rho in (-1,1), nu > 0.
    parameters : np.ndarray
        Copula parameters [rho, nu].
    n_nodes : int
        Number of Gauss–Laguerre nodes for CDF computation.
    """

    def __init__(self):
        super().__init__()
        self.type = "student"
        self.name = "Student-t Copula"
        self.bounds_param = [(-0.999, 0.999), (1e-6, None)]
        self._parameters = np.array([0.5, 4.0])
        self.default_optim_method = "Powell"
        self.n_nodes = 64

    @property
    def parameters(self) -> np.ndarray:
        """
        Get the copula parameters.

        Returns
        -------
        np.ndarray
            Current parameters [rho, nu].
        """
        return self._parameters

    @parameters.setter
    def parameters(self, param: np.ndarray):
        """
        Set and validate copula parameters against bounds_param.

        Parameters
        ----------
        param : array-like
            New parameters [rho, nu].

        Raises
        ------
        ValueError
            If any value is outside its specified bound.
        """
        param = np.asarray(param)
        for idx, (lower, upper) in enumerate(self.bounds_param):
            val = param[idx]
            if lower is not None and val <= lower:
                raise ValueError(f"Parameter at index {idx} must be > {lower}, got {val}")
            if upper is not None and val >= upper:
                raise ValueError(f"Parameter at index {idx} must be < {upper}, got {val}")
        self._parameters = param

    def get_cdf(self, u: float, v: float, param: np.ndarray = None) -> float:
        """
        Compute C(u,v) via Gauss–Laguerre quadrature.

        Parameters
        ----------
        u : float
            First pseudo-observation in (0,1).
        v : float
            Second pseudo-observation in (0,1).
        param : ndarray, optional
            Copula parameters [rho, nu].

        Returns
        -------
        float
            Copula CDF at (u, v).
        """
        if param is None:
            param = self.parameters
        rho, nu = param
        # boundary checks
        if u <= 0 or v <= 0:
            return 0.0
        if u >= 1 and v >= 1:
            return 1.0
        if u >= 1:
            return v
        if v >= 1:
            return u
        # quantiles
        x = t.ppf(u, df=nu)
        y = t.ppf(v, df=nu)
        # quadrature
        k = nu / 2.0
        alpha = k - 1.0
        z_nodes, w_weights = roots_genlaguerre(self.n_nodes, alpha)
        cov = [[1.0, rho], [rho, 1.0]]
        mvn = multivariate_normal(mean=[0.0,0.0], cov=cov)
        total = 0.0
        for zi, wi in zip(z_nodes, w_weights):
            scale = sqrt(2.0 * zi / nu)
            total += wi * mvn.cdf([x * scale, y * scale])
        return total / gamma(k)

    def get_pdf(self, u: float, v: float, param: np.ndarray = None):
        """
        Compute PDF c(u,v) using log-domain formulation.

        Parameters
        ----------
        u : float or array-like
            Pseudo-observations in (0,1).
        v : float or array-like
            Pseudo-observations in (0,1).
        param : ndarray, optional
            Copula parameters [rho, nu].

        Returns
        -------
        float or ndarray
            Copula PDF value(s).
        """
        if param is None:
            param = self.parameters
        rho, nu = param
        eps = 1e-12
        u = np.clip(u, eps, 1-eps)
        v = np.clip(v, eps, 1-eps)
        u_q = t.ppf(u, df=nu)
        v_q = t.ppf(v, df=nu)
        det = 1 - rho**2
        quad_form = (u_q**2 - 2*rho*u_q*v_q + v_q**2) / det
        # log-domain
        log_num = gammaln((nu+2)/2) + gammaln(nu/2)
        log_den = 2 * gammaln((nu+1)/2)
        log_det = 0.5 * np.log(det)
        log_prod = ((nu+1)/2) * (np.log1p((u_q**2)/nu) + np.log1p((v_q**2)/nu))
        log_dent = ((nu+2)/2) * np.log1p(quad_form/nu)
        log_c = log_num - log_den - log_det + log_prod - log_dent
        return np.exp(log_c)

    def sample(self, n: int, param: np.ndarray = None) -> np.ndarray:
        """
        Generate samples (u,v) from the Student-t copula.

        Parameters
        ----------
        n : int
            Number of samples.
        param : ndarray, optional
            Copula parameters [rho, nu].

        Returns
        -------
        np.ndarray
            Shape (n,2) array of pseudo-observations.
        """
        if param is None:
            param = self.parameters
        rho, nu = param
        cov = np.array([[1.0, rho],[rho,1.0]])
        L = np.linalg.cholesky(cov)
        z = np.random.standard_normal((n,2))
        chi2 = np.random.chisquare(df=nu, size=n)
        scaled = (z @ L.T) / np.sqrt((chi2/nu)[:,None])
        u = t.cdf(scaled[:,0], df=nu)
        v = t.cdf(scaled[:,1], df=nu)
        return np.column_stack((u, v))

    def kendall_tau(self, param: np.ndarray = None, n_samples: int=10000, random_state=None) -> float:
        """
        Estimate Kendall's tau via Monte Carlo when no closed form.

        Parameters
        ----------
        param : ndarray, optional
            Copula parameters [rho, nu].
        n_samples : int
            Number of MC draws.
        random_state : int or RandomState, optional
            Seed or rng.

        Returns
        -------
        float
            Estimated Kendall's tau.
        """
        if param is None:
            param = self.parameters
        rho, nu = param
        rng = np.random.RandomState(random_state) if random_state is not None else np.random
        cov = np.array([[1.0, rho],[rho,1.0]])
        L = np.linalg.cholesky(cov)
        z = rng.standard_normal((n_samples,2))
        chi2 = rng.chisquare(df=nu, size=n_samples)
        scaled = (z @ L.T) / np.sqrt((chi2/nu)[:,None])
        u = t.cdf(scaled[:,0], df=nu)
        v = t.cdf(scaled[:,1], df=nu)
        tau, _ = kendalltau(u, v)
        return tau

    def LTDC(self, param: np.ndarray = None) -> float:
        """
        Lower-tail dependence λ_L = 2*T_{ν+1}(-√((ν+1)*(1-ρ)/(1+ρ))).

        Parameters
        ----------
        param : ndarray, optional
            Copula parameters [rho, nu].

        Returns
        -------
        float
            Lower-tail dependence.
        """
        if param is None:
            param = self.parameters
        rho, nu = param
        return 2 * t.cdf(-sqrt((nu+1)*(1-rho)/(1+rho)), df=nu+1)

    def UTDC(self, param: np.ndarray = None) -> float:
        """
        Upper-tail dependence = same as lower for Student-t.

        Parameters
        ----------
        param : ndarray, optional
            Copula parameters [rho, nu].

        Returns
        -------
        float
            Upper-tail dependence.
        """
        if param is None:
            param = self.parameters
        return self.LTDC(param)

    def IAD(self, data) -> float:
        """
        Integrated absolute deviation disabled for elliptical.

        Returns
        -------
        float
            np.nan
        """
        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data) -> float:
        """
        Anderson–Darling disabled for elliptical.

        Returns
        -------
        float
            np.nan
        """
        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan

    def partial_derivative_C_wrt_v(self, u: float, v: float, param: np.ndarray=None):
        """
        Compute ∂C/∂v for Student-t copula.

        Parameters
        ----------
        u : float
            Pseudo-observation.
        v : float
            Pseudo-observation.
        param : ndarray, optional
            Copula parameters [rho, nu].

        Returns
        -------
        float
            Partial derivative ∂C/∂v.
        """
        if param is None:
            param = self.parameters
        rho, nu = param
        eps = 1e-12
        u = np.clip(u, eps, 1-eps)
        v = np.clip(v, eps, 1-eps)
        x = t.ppf(u, df=nu)
        y = t.ppf(v, df=nu)
        scale = sqrt((1-rho**2)*(nu+y**2)/(nu+1))
        z = (x - rho*y)/scale
        pdf_std = t.pdf(z, df=nu+1)
        return pdf_std/scale

    def partial_derivative_C_wrt_u(self, u: float, v: float, param: np.ndarray=None):
        """
        Compute ∂C/∂u for Student-t copula.

        Parameters
        ----------
        u : float
            Pseudo-observation.
        v : float
            Pseudo-observation.
        param : ndarray, optional
            Copula parameters [rho, nu].

        Returns
        -------
        float
            Partial derivative ∂C/∂u.
        """
        if param is None:
            param = self.parameters
        rho, nu = param
        eps = 1e-12
        u = np.clip(u, eps, 1-eps)
        v = np.clip(v, eps, 1-eps)
        x = t.ppf(u, df=nu)
        y = t.ppf(v, df=nu)
        scale = sqrt((1-rho**2)*(nu+x**2)/(nu+1))
        z = (y - rho*x)/scale
        pdf_std = t.pdf(z, df=nu+1)
        return pdf_std/scale

    def conditional_cdf_u_given_v(self, u: float, v: float, param: np.ndarray=None) -> float:
        """
        Compute P(U ≤ u|V=v) for Student-t.

        Parameters
        ----------
        u : float
            Threshold for U.
        v : float
            Given V.
        param : ndarray, optional
            Copula parameters [rho, nu].

        Returns
        -------
        float
            Conditional CDF.
        """
        if param is None:
            param = self.parameters
        num = self.partial_derivative_C_wrt_v(u, v, param)
        den = self.partial_derivative_C_wrt_v(1.0, v, param)
        return num / max(den, 1e-14)

    def conditional_cdf_v_given_u(self, u: float, v: float, param: np.ndarray=None) -> float:
        """
        Compute P(V ≤ v|U=u) for Student-t.

        Parameters
        ----------
        u : float
            Given U.
        v : float
            Threshold for V.
        param : ndarray, optional
            Copula parameters [rho, nu].

        Returns
        -------
        float
            Conditional CDF.
        """
        if param is None:
            param = self.parameters
        num = self.partial_derivative_C_wrt_u(u, v, param)
        den = self.partial_derivative_C_wrt_u(u, 1.0, param)
        return num / max(den, 1e-14)
