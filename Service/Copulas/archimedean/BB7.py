"""
BB7 Copula (Joe–Clayton) implementation following project coding standard:

Norms:
 1. Use private `_parameters` with public `@property parameters` and validation in setter.
 2. All methods accept `param: np.ndarray = None` defaulting to `self.parameters`.
 3. Docstrings include **Parameters** and **Returns** with types.
 4. Parameter bounds in `bounds_param`; setter enforces them.
 5. Uniform clipping with `eps=1e-12`.
 6. Document parameter names (`theta`, `delta`) in `__init__`.
"""
import numpy as np
from scipy.optimize import brentq

from Service.Copulas.base import BaseCopula


class BB7Copula(BaseCopula):
    """
    BB7 Copula (Joe–Clayton) Archimedean generator φ(t) = φ_C(φ_J(t)).

    Parameters
    ----------
    theta : float
        Joe parameter (θ > 0) controlling upper tail.
    delta : float
        Clayton parameter (δ > 0) controlling lower tail.
    """
    def __init__(self):
        super().__init__()
        self.type = "bb7"
        self.name = "BB7 Copula"
        # theta > 0, delta > 0
        self.bounds_param = [(1e-6, None), (1e-6, None)]
        self._parameters = np.array([1.0, 1.0])  # [theta, delta]
        self.default_optim_method = "Powell"

    @property
    def parameters(self) -> np.ndarray:
        """
        Get copula parameters [theta, delta].

        Returns
        -------
        np.ndarray
            Current parameters.
        """
        return self._parameters

    @parameters.setter
    def parameters(self, param: np.ndarray):
        """
        Set and validate copula parameters.

        Parameters
        ----------
        param : array-like
            New parameters [theta, delta].

        Raises
        ------
        ValueError
            If any parameter outside its bound.
        """
        param = np.asarray(param)
        names = ['theta', 'delta']
        for i, (lower, upper) in enumerate(self.bounds_param):
            val = param[i]
            name = names[i]
            if lower is not None and val < lower:
                raise ValueError(f"Parameter '{name}' must be >= {lower}, got {val}")
        self._parameters = param

    def _phi(self, t: np.ndarray, theta: float, delta: float) -> np.ndarray:
        """
        Archimedean generator φ(t) = (φ_J(t)^(-delta) - 1)/delta,
        φ_J(t) = 1 - (1 - t)**theta.

        Parameters
        ----------
        t : float or ndarray
        theta : float
        delta : float

        Returns
        -------
        float or ndarray
            Generator value(s).
        """
        eps = 1e-12
        t = np.clip(t, eps, 1 - eps)
        phiJ = 1.0 - (1.0 - t)**theta
        return (phiJ**(-delta) - 1.0) / delta

    def _phi_inv(self, s: np.ndarray, theta: float, delta: float) -> np.ndarray:
        """
        Inverse generator φ^{-1}(s) = 1 - (1 - (1 + delta s)^{-1/delta})^{1/theta}.

        Parameters
        ----------
        s : float or ndarray
        theta : float
        delta : float

        Returns
        -------
        float or ndarray
            Inverse generator value(s).
        """
        s = np.maximum(s, 0.0)
        phiC_inv = (1.0 + delta * s)**(-1.0 / delta)
        return 1.0 - (1.0 - phiC_inv)**(1.0 / theta)

    def get_cdf(self, u, v, param: np.ndarray = None):
        """
        Copula CDF: C(u,v) = φ^{-1}(φ(u) + φ(v)).

        Parameters
        ----------
        u : float or ndarray
        v : float or ndarray
        param : ndarray, optional
            Copula parameters [theta, delta].

        Returns
        -------
        float or ndarray
            Copula CDF value(s).
        """
        if param is None:
            param = self.parameters
        theta, delta = param
        phi_u = self._phi(u, theta, delta)
        phi_v = self._phi(v, theta, delta)
        return self._phi_inv(phi_u + phi_v, theta, delta)

    def get_pdf(self, u, v, param: np.ndarray = None):
        """
        Approximate PDF by central finite-difference ∂²C/∂u∂v.

        Parameters
        ----------
        u : float or ndarray
        v : float or ndarray
        param : ndarray, optional
            Copula parameters [theta, delta].

        Returns
        -------
        float or ndarray
            PDF approximation.
        """
        if param is None:
            param = self.parameters
        eps = 1e-6
        c = self.get_cdf
        return (
            c(u+eps, v+eps, param)
            - c(u+eps, v-eps, param)
            - c(u-eps, v+eps, param)
            + c(u-eps, v-eps, param)
        ) / (4.0 * eps**2)

    def partial_derivative_C_wrt_u(self, u, v, param: np.ndarray = None):
        """
        Approximate ∂C/∂u by central difference.

        Parameters
        ----------
        u : float or ndarray
        v : float or ndarray
        param : ndarray, optional
            Copula parameters.

        Returns
        -------
        float or ndarray
        """
        if param is None:
            param = self.parameters
        eps = 1e-6
        c = self.get_cdf
        return (c(u+eps, v, param) - c(u-eps, v, param)) / (2.0 * eps)

    def partial_derivative_C_wrt_v(self, u, v, param: np.ndarray = None):
        """
        Approximate ∂C/∂v by central difference.

        Returns
        -------
        float or ndarray
        """
        if param is None:
            param = self.parameters
        eps = 1e-6
        c = self.get_cdf
        return (c(u, v+eps, param) - c(u, v-eps, param)) / (2.0 * eps)

    def conditional_cdf_u_given_v(self, u, v, param: np.ndarray = None):
        """
        P(U ≤ u | V = v) = ∂C/∂v(u,v) / ∂C/∂v(1,v).

        Returns
        -------
        float or ndarray
        """
        if param is None:
            param = self.parameters
        num = self.partial_derivative_C_wrt_v(u, v, param)
        den = self.partial_derivative_C_wrt_v(1.0, v, param)
        return num / den

    def conditional_cdf_v_given_u(self, u, v, param: np.ndarray = None):
        """
        P(V ≤ v | U = u) = ∂C/∂u(u,v) / ∂C/∂u(u,1).

        Returns
        -------
        float or ndarray
        """
        if param is None:
            param = self.parameters
        num = self.partial_derivative_C_wrt_u(u, v, param)
        den = self.partial_derivative_C_wrt_u(u, 1.0, param)
        return num / den

    def sample(self, n: int, param: np.ndarray = None) -> np.ndarray:
        """
        Generate samples via conditional inversion.

        Parameters
        ----------
        n : int
            Number of samples.
        param : ndarray, optional
            Copula parameters.

        Returns
        -------
        np.ndarray
            Shape (n,2) samples.
        """
        if param is None:
            param = self.parameters
        samples = np.empty((n, 2))
        eps = 1e-6
        for i in range(n):
            u = np.random.rand()
            p = np.random.rand()
            root = brentq(
                lambda vv: self.conditional_cdf_v_given_u(u, vv, param) - p,
                eps, 1.0 - eps
            )
            samples[i] = [u, root]
        return samples

    def LTDC(self, param: np.ndarray = None) -> float:
        """
        Approximate lower tail dependence λ_L = lim_{u→0} C(u,u)/u.

        Returns
        -------
        float
        """
        if param is None:
            param = self.parameters
        eps = 1e-6
        return self.get_cdf(eps, eps, param) / eps

    def UTDC(self, param: np.ndarray = None) -> float:
        """
        Approximate upper tail dependence λ_U = 2 - lim_{u→1} (1-2u+C(u,u))/(1-u).

        Returns
        -------
        float
        """
        if param is None:
            param = self.parameters
        eps = 1e-6
        u = 1.0 - eps
        return 2.0 - (1.0 - 2*u + self.get_cdf(u, u, param)) / eps

    def kendall_tau(self, param: np.ndarray = None):
        """
        Not implemented for BB7.
        """
        raise NotImplementedError("Kendall's tau not implemented for BB7.")
