"""
Clayton Copula implementation following the project coding standard:

Norms:
 1. Use private `_parameters` with public `@property parameters` and validation in setter.
 2. All methods accept `param: np.ndarray = None` defaulting to `self.parameters`.
 3. Docstrings include **Parameters** and **Returns** with types.
 4. Parameter bounds in `bounds_param`; setter enforces them.
 5. Uniform boundary clipping with `eps=1e-12`.
"""
import numpy as np
from scipy.stats import uniform
from Service.Copulas.base import BaseCopula


class ClaytonCopula(BaseCopula):
    """
    Clayton Copula (Archimedean) class.

    Parameters
    ----------
    theta : float
        Dependence parameter (θ > 0) controlling lower-tail dependence.
    """
    def __init__(self):
        super().__init__()
        self.type = 'clayton'
        self.name = 'Clayton Copula'
        # theta > 0
        self.bounds_param = [(1e-6, None)]
        self._parameters = np.array([0.5])  # [theta]
        self.default_optim_method = 'SLSQP'

    @property
    def parameters(self) -> np.ndarray:
        """
        Get copula parameters [theta].

        Returns
        -------
        np.ndarray
            Current parameter array.
        """
        return self._parameters

    @parameters.setter
    def parameters(self, param: np.ndarray):
        """
        Set and validate copula parameter θ.

        Parameters
        ----------
        param : array-like
            New parameter array [theta].

        Raises
        ------
        ValueError
            If theta is outside its bound.
        """
        param = np.asarray(param)
        theta = param[0]
        lower, upper = self.bounds_param[0]
        if lower is not None and theta < lower:
            raise ValueError(f"Parameter 'theta' must be >= {lower}, got {theta}")
        if upper is not None and theta > upper:
            raise ValueError(f"Parameter 'theta' must be <= {upper}, got {theta}")
        self._parameters = param

    def get_cdf(self, u, v, param: np.ndarray = None):
        """
        Compute the Clayton copula CDF: C(u,v) = (u^-θ + v^-θ - 1)^(-1/θ).

        Parameters
        ----------
        u : float or ndarray
            First pseudo-observation in (0,1).
        v : float or ndarray
            Second pseudo-observation in (0,1).
        param : ndarray, optional
            Copula parameters [theta].

        Returns
        -------
        float or ndarray
            Copula CDF value(s).
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        return (u**(-theta) + v**(-theta) - 1.0)**(-1.0/theta)

    def get_pdf(self, u, v, param: np.ndarray = None):
        """
        Compute the Clayton copula PDF.

        Parameters
        ----------
        u : float or ndarray
        v : float or ndarray
        param : ndarray, optional
            Copula parameters [theta].

        Returns
        -------
        float or ndarray
            Copula PDF value(s).
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        term1 = (theta + 1.0) * (u * v)**(-theta - 1.0)
        term2 = (u**(-theta) + v**(-theta) - 1.0)**(-2.0 - 1.0/theta)
        return term1 * term2

    def kendall_tau(self, param: np.ndarray = None) -> float:
        """
        Compute Kendall's tau: τ = θ / (θ + 2).

        Parameters
        ----------
        param : ndarray, optional
            Copula parameters [theta].

        Returns
        -------
        float
            Kendall's tau.
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        return theta / (theta + 2.0)

    def sample(self, n: int, param: np.ndarray = None) -> np.ndarray:
        """
        Generate samples via inverse-transform method.

        Parameters
        ----------
        n : int
            Number of samples.
        param : ndarray, optional
            Copula parameters [theta].

        Returns
        -------
        ndarray of shape (n,2)
            Samples in [0,1]^2.
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        if theta <= 0:
            raise ValueError("Clayton copula requires theta > 0")
        u = uniform.rvs(size=n)
        w = uniform.rvs(size=n)
        v = (w**(-theta/(1.0+theta)) * (u**(-theta) - 1.0) + 1.0)**(-1.0/theta)
        return np.column_stack((u, v))

    def LTDC(self, param: np.ndarray = None) -> float:
        """
        Lower tail dependence: λ_L = 2^(-1/θ).

        Parameters
        ----------
        param : ndarray, optional
            Copula parameters [theta].

        Returns
        -------
        float
            Lower tail dependence.
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        return 2.0**(-1.0/theta)

    def UTDC(self, param: np.ndarray = None) -> float:
        """
        Upper tail dependence (always 0 for Clayton).

        Returns
        -------
        float
            Zero.
        """
        return 0.0

    def partial_derivative_C_wrt_v(self, u, v, param: np.ndarray = None):
        """
        ∂C/∂v = v^(-θ-1) * A^(-1/θ-1), where A = u^-θ + v^-θ - 1.

        Parameters
        ----------
        u : float or ndarray
        v : float or ndarray
        param : ndarray, optional
            Copula parameters [theta].

        Returns
        -------
        float or ndarray
            Partial derivative w.r.t. v.
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        A = u**(-theta) + v**(-theta) - 1.0
        return v**(-theta - 1.0) * A**(-1.0/theta - 1.0)

    def partial_derivative_C_wrt_u(self, u, v, param: np.ndarray = None):
        """
        ∂C/∂u = u^(-θ-1) * A^(-1/θ-1), same A as above.

        Returns
        -------
        float or ndarray
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        A = u**(-theta) + v**(-theta) - 1.0
        return u**(-theta - 1.0) * A**(-1.0/theta - 1.0)

    def conditional_cdf_u_given_v(self, u, v, param: np.ndarray = None):
        """
        P(U ≤ u | V = v) = ∂C/∂v.
        """
        return self.partial_derivative_C_wrt_v(u, v, param)

    def conditional_cdf_v_given_u(self, u, v, param: np.ndarray = None):
        """
        P(V ≤ v | U = u) = ∂C/∂u.
        """
        return self.partial_derivative_C_wrt_u(u, v, param)
