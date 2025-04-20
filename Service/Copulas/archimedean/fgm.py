"""
FGM Copula implementation following the project coding standard:

Norms:
 1. Use private `_parameters` with public `@property parameters` and bounds-checked setter.
 2. Methods accept `param: np.ndarray = None` defaulting to `self.parameters`.
 3. Docstrings include **Parameters** and **Returns** with types.
 4. Uniform clipping with `eps=1e-12`.
 5. Parameter bounds in `bounds_param`.
"""
import numpy as np
from scipy.stats import uniform
from Service.Copulas.base import BaseCopula


class FGMCopula(BaseCopula):
    """
    Farlie–Gumbel–Morgenstern (FGM) Copula class.

    Parameters
    ----------
    theta : float
        Dependence parameter θ ∈ [-1, 1].
    """
    def __init__(self):
        super().__init__()
        self.type = 'fgm'
        self.name = 'FGM Copula'
        # theta ∈ [-1,1]
        self.bounds_param = [(-1.0, 1.0)]
        self._parameters = np.array([0.0])  # [theta]
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
            If θ outside [-1,1].
        """
        param = np.asarray(param)
        theta = param[0]
        lower, upper = self.bounds_param[0]
        if theta < lower or theta > upper:
            raise ValueError(f"Parameter 'theta' must be in [{lower}, {upper}], got {theta}")
        self._parameters = param

    def get_cdf(self, u, v, param: np.ndarray = None):
        """
        Compute the FGM copula CDF:
            C(u,v) = u·v·[1 + θ·(1-u)·(1-v)].

        Parameters
        ----------
        u : float or ndarray
            First pseudo-observation(s) in (0,1).
        v : float or ndarray
            Second pseudo-observation(s) in (0,1).
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
        u = np.clip(u, eps, 1.0 - eps)
        v = np.clip(v, eps, 1.0 - eps)
        return u * v * (1.0 + theta * (1.0 - u) * (1.0 - v))

    def get_pdf(self, u, v, param: np.ndarray = None):
        """
        Compute the FGM copula PDF:
            c(u,v) = 1 + θ·(1 - 2u)·(1 - 2v).

        Parameters
        ----------
        u : float or ndarray
            First pseudo-observation(s) in (0,1).
        v : float or ndarray
            Second pseudo-observation(s) in (0,1).
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
        u = np.clip(u, eps, 1.0 - eps)
        v = np.clip(v, eps, 1.0 - eps)
        return 1.0 + theta * (1.0 - 2.0*u) * (1.0 - 2.0*v)

    def kendall_tau(self, param: np.ndarray = None) -> float:
        """
        Compute Kendall's tau: τ = 2θ/9.

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
        return 2.0 * theta / 9.0

    def sample(self, n: int, param: np.ndarray = None) -> np.ndarray:
        """
        Generate samples via conditional inversion.

        Parameters
        ----------
        n : int
            Number of samples.
        param : ndarray, optional
            Copula parameters [theta].

        Returns
        -------
        ndarray of shape (n,2)
            Pseudo-observations in [0,1]^2.
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        eps = 1e-12
        # generate uniforms
        u = uniform.rvs(size=n)
        w = uniform.rvs(size=n)
        A = theta * (1.0 - 2.0*u)
        v = np.copy(w)
        mask = np.abs(A) > eps
        a = A[mask]
        b = -(1.0 + A[mask])
        c = w[mask]
        disc = b**2 - 4.0*a*c
        sqrtD = np.sqrt(np.maximum(disc, 0.0))
        v1 = (-b - sqrtD) / (2.0*a)
        v2 = (-b + sqrtD) / (2.0*a)
        valid = (v1>=0)&(v1<=1)
        v[mask] = np.where(valid, v1, np.where((v2>=0)&(v2<=1), v2, w[mask]))
        return np.column_stack((u, v))

    def LTDC(self, param: np.ndarray = None) -> float:
        """
        Lower tail dependence (0 for FGM).

        Returns
        -------
        float
        """
        return 0.0

    def UTDC(self, param: np.ndarray = None) -> float:
        """
        Upper tail dependence (0 for FGM).

        Returns
        -------
        float
        """
        return 0.0

    def partial_derivative_C_wrt_v(self, u, v, param: np.ndarray = None):
        """
        ∂C/∂v = u + θ·u·(1-u)·(1-2v).

        Parameters
        ----------
        u : float or ndarray
        v : float or ndarray
        param : ndarray, optional
            Copula parameters [theta].

        Returns
        -------
        float or ndarray
            Partial derivative.
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        u = np.asarray(u)
        v = np.asarray(v)
        return u + theta * u * (1.0 - u) * (1.0 - 2.0*v)

    def partial_derivative_C_wrt_u(self, u, v, param: np.ndarray = None):
        """
        ∂C/∂u = v + θ·v·(1-v)·(1-2u).
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        u = np.asarray(u)
        v = np.asarray(v)
        return v + theta * v * (1.0 - v) * (1.0 - 2.0*u)

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
