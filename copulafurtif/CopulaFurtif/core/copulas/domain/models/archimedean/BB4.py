"""
BB4 Copula implementation following the project coding standard:

Norms:
 1. Use private `_parameters` with public `@property parameters` and validation in setter.
 2. All methods accept `param: np.ndarray = None` defaulting to `self.parameters`.
 3. Docstrings include **Parameters** and **Returns** with types.
 4. Parameter bounds in `bounds_param`; setter enforces them.
 5. Uniform boundary clipping with `eps=1e-12` and `np.clip`.
 6. Document parameter names (`mu`, `delta`) in `__init__` docstring.
"""
import numpy as np
from scipy.optimize import brentq

from Service.Copulas.base import BaseCopula


class BB4Copula(BaseCopula):
    """
    BB4 Copula (Two-parameter Archimedean copula).

    Parameters
    ----------
    mu : float
        Copula parameter (mu > 0) controlling marginal power.
    delta : float
        Copula parameter (delta > 0) controlling tail behavior.
    """
    def __init__(self):
        super().__init__()
        self.type = "bb4"
        self.name = "BB4 Copula"
        # mu > 0, delta > 0
        self.bounds_param = [(1e-6, None), (1e-6, None)]
        self._parameters = np.array([1.0, 1.0])  # [mu, delta]
        self.default_optim_method = "Powell"

    @property
    def parameters(self) -> np.ndarray:
        """
        Get the copula parameters.

        Returns
        -------
        np.ndarray
            Current parameters [mu, delta].
        """
        return self._parameters

    @parameters.setter
    def parameters(self, param: np.ndarray):
        """
        Set and validate copula parameters against bounds_param.

        Parameters
        ----------
        param : array-like
            New parameters [mu, delta].

        Raises
        ------
        ValueError
            If any value is outside its specified bound.
        """
        param = np.asarray(param)
        names = ['mu', 'delta']
        for idx, (lower, upper) in enumerate(self.bounds_param):
            val = param[idx]
            name = names[idx]
            if lower is not None and val <= lower:
                raise ValueError(f"Parameter '{name}' must be > {lower}, got {val}")
            if upper is not None and val <= 0 and False:
                pass
        self._parameters = param

    def get_cdf(self, u, v, param: np.ndarray = None):
        """
        Compute the BB4 copula CDF: C(u,v) = [x + y - 1 - A]^{-1/mu},
        where x = u^{-mu}, y = v^{-mu}, A = T^{-1/delta}, T = (x-1)^{-delta} + (y-1)^{-delta}.

        Parameters
        ----------
        u : float or np.ndarray
            Pseudo-observations in (0,1).
        v : float or np.ndarray
            Pseudo-observations in (0,1).
        param : ndarray, optional
            Copula parameters [mu, delta].

        Returns
        -------
        float or np.ndarray
            Copula CDF value(s).
        """
        if param is None:
            param = self.parameters
        mu, delta = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        x = u ** (-mu)
        y = v ** (-mu)
        T = (x - 1) ** (-delta) + (y - 1) ** (-delta)
        A = T ** (-1.0 / delta)
        z = x + y - 1.0 - A
        return z ** (-1.0 / mu)

    def get_pdf(self, u, v, param: np.ndarray = None):
        """
        Compute the BB4 copula PDF via analytic derivatives.

        Parameters
        ----------
        u : float or np.ndarray
            Pseudo-observations in (0,1).
        v : float or np.ndarray
            Pseudo-observations in (0,1).
        param : ndarray, optional
            Copula parameters [mu, delta].

        Returns
        -------
        float or np.ndarray
            Copula PDF value(s).
        """
        if param is None:
            param = self.parameters
        mu, delta = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        x = u ** (-mu)
        y = v ** (-mu)
        T = (x - 1) ** (-delta) + (y - 1) ** (-delta)
        A = T ** (-1.0 / delta)
        z = x + y - 1.0 - A
        dzdu = -mu * u ** (-mu - 1) * (1.0 - (x - 1) ** (-delta - 1) * T ** (-1.0 / delta - 1))
        dzdv = -mu * v ** (-mu - 1) * (1.0 - (y - 1) ** (-delta - 1) * T ** (-1.0 / delta - 1))
        d2zdudv = -mu**2 * (delta + 1) * u**(-mu - 1) * v**(-mu - 1) * (x - 1)**(-delta - 1) * (y - 1)**(-delta - 1) * T**(-1.0 / delta - 2)
        dCdz = -1.0 / mu * z ** (-1.0 / mu - 1)
        d2Cdz2 = (1.0 / mu) * (1.0 / mu + 1.0) * z ** (-1.0 / mu - 2)
        return d2Cdz2 * dzdu * dzdv + dCdz * d2zdudv

    def kendall_tau(self, param: np.ndarray = None, n: int = 201) -> float:
        """
        Estimate Kendall's tau by numeric double integration:

            τ = 4 ∫∫ C(u,v) du dv - 1.

        Parameters
        ----------
        param : ndarray, optional
            Copula parameters [mu, delta].
        n : int
            Number of grid points per axis.

        Returns
        -------
        float
            Estimated Kendall's tau.
        """
        if param is None:
            param = self.parameters
        eps = 1e-6
        u = np.linspace(eps, 1 - eps, n)
        U, V = np.meshgrid(u, u)
        Z = self.get_cdf(U, V, param)
        integral = np.trapz(np.trapz(Z, u, axis=1), u)
        return 4.0 * integral - 1.0

    def sample(self, n: int, param: np.ndarray = None) -> np.ndarray:
        """
        Generate samples via conditional inversion:

        Parameters
        ----------
        n : int
            Number of samples.
        param : ndarray, optional
            Copula parameters [mu, delta].

        Returns
        -------
        np.ndarray
            Shape (n,2) array of pseudo-observations.
        """
        if param is None:
            param = self.parameters
        samples = np.empty((n, 2))
        for i in range(n):
            u = np.random.rand()
            p = np.random.rand()
            root = brentq(
                lambda vv: self.conditional_cdf_v_given_u(u, vv, param) - p,
                1e-6,
                1 - 1e-6
            )
            samples[i] = [u, root]
        return samples

    def LTDC(self, param: np.ndarray = None) -> float:
        """
        Lower-tail dependence λ_L = 0 for BB4.

        Returns
        -------
        float
            0.0
        """
        return 0.0

    def UTDC(self, param: np.ndarray = None) -> float:
        """
        Upper-tail dependence λ_U = 2 - 2^(1/delta).

        Parameters
        ----------
        param : ndarray, optional
            Copula parameters [mu, delta].

        Returns
        -------
        float
            Upper-tail dependence.
        """
        if param is None:
            param = self.parameters
        delta = param[1]
        return 2.0 - 2.0 ** (1.0 / delta)

    def partial_derivative_C_wrt_v(self, u, v, param: np.ndarray = None):
        """
        Compute partial derivative ∂C/∂v.

        Parameters
        ----------
        u, v : float or ndarray
            Pseudo-observations in (0,1).
        param : ndarray, optional
            Copula parameters [mu, delta].

        Returns
        -------
        float or ndarray
        """
        if param is None:
            param = self.parameters
        mu, delta = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        x = u ** (-mu)
        y = v ** (-mu)
        T = (x - 1) ** (-delta) + (y - 1) ** (-delta)
        h1 = - (1.0/mu) * ((x + y - 1.0 - T**(-1.0/delta))**(-1.0/mu -1))
        phi_inv_v_prime = -mu * (v ** (-mu -1)) * (1 - (y -1)**(-delta-1) * T**(-1.0/delta-1))
        return h1 * phi_inv_v_prime

    def partial_derivative_C_wrt_u(self, u, v, param: np.ndarray = None):
        """
        Compute partial derivative ∂C/∂u.

        Parameters
        ----------
        u, v : float or ndarray
            Pseudo-observations in (0,1).
        param : ndarray, optional
            Copula parameters [mu, delta].

        Returns
        -------
        float or ndarray
        """
        if param is None:
            param = self.parameters
        mu, delta = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        x = u ** (-mu)
        y = v ** (-mu)
        T = (x - 1) ** (-delta) + (y - 1) ** (-delta)
        h1 = - (1.0/mu) * ((x + y - 1.0 - T**(-1.0/delta))**(-1.0/mu -1))
        phi_inv_u_prime = -mu * (u ** (-mu -1)) * (1 - (x -1)**(-delta-1) * T**(-1.0/delta-1))
        return h1 * phi_inv_u_prime

    def conditional_cdf_u_given_v(self, u, v, param: np.ndarray = None):
        """
        Conditional CDF P(U ≤ u | V = v).

        Parameters
        ----------
        u, v : float or ndarray
        param : ndarray, optional

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
        Conditional CDF P(V ≤ v | U = u).

        Parameters
        ----------
        u, v : float or ndarray
        param : ndarray, optional

        Returns
        -------
        float or ndarray
        """
        if param is None:
            param = self.parameters
        num = self.partial_derivative_C_wrt_u(u, v, param)
        den = self.partial_derivative_C_wrt_u(u, 1.0, param)
        return num / den
