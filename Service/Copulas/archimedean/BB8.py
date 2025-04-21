"""
BB8 Copula implementation following the project coding standard:

Norms:
 1. Use private `_parameters` with public `@property parameters` and validation in setter.
 2. All methods accept `param: np.ndarray = None` defaulting to `self.parameters`.
 3. Docstrings include **Parameters** and **Returns** with types.
 4. Parameter bounds in `bounds_param`; setter enforces them.
 5. Finite-difference approximations use `eps=1e-6`.
 6. Uniform boundary clipping with `eps=1e-12` for CDF.
"""
import numpy as np
from scipy.optimize import root_scalar

from Service.Copulas.base import BaseCopula


class BB8Copula(BaseCopula):
    """
    BB8 Copula (Durante et al.):
      C(u,v) = [1 - (1-A)*(1-B)]^(1/theta)
      A = [1 - (1-u)^theta]^delta, B = [1 - (1-v)^theta]^delta

    Parameters
    ----------
    theta : float
        Tail dependence parameter (θ >= 1).
    delta : float
        Asymmetry parameter (0 < δ <= 1).
    """
    def __init__(self):
        super().__init__()
        self.type = "bb8"
        self.name = "BB8 Copula (Durante)"
        # theta >= 1, delta in (0,1]
        self.bounds_param = [(1.0, None), (0.0, 1.0)]
        self._parameters = np.array([2.0, 0.7])  # [theta, delta]
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
        Set and validate copula parameters against bounds_param.

        Parameters
        ----------
        param : array-like
            New parameters [theta, delta].

        Raises
        ------
        ValueError
            If any value is outside its specified bound.
        """
        param = np.asarray(param)
        names = ['theta', 'delta']
        for i, (lower, upper) in enumerate(self.bounds_param):
            val = param[i]
            name = names[i]
            if lower is not None and val < lower:
                raise ValueError(f"Parameter '{name}' must be >= {lower}, got {val}")
            if upper is not None and val > upper:
                raise ValueError(f"Parameter '{name}' must be <= {upper}, got {val}")
        self._parameters = param

    def get_cdf(self, u, v, param: np.ndarray = None):
        """
        Compute BB8 copula CDF: C(u,v) = [1 - (1-A)*(1-B)]^(1/theta).

        Parameters
        ----------
        u : float or ndarray
            First pseudo-observation in (0,1).
        v : float or ndarray
            Second pseudo-observation in (0,1).
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
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        A = (1.0 - (1.0 - u)**theta)**delta
        B = (1.0 - (1.0 - v)**theta)**delta
        inner = 1.0 - (1.0 - A)*(1.0 - B)
        return inner**(1.0/theta)

    def get_pdf(self, u, v, param: np.ndarray = None):
        """
        Approximate BB8 copula PDF via finite differences.

        Parameters
        ----------
        u : float or ndarray
        v : float or ndarray
        param : ndarray, optional
            Copula parameters [theta, delta].

        Returns
        -------
        float or ndarray
            Approximate PDF c(u,v).
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
        Approximate ∂C/∂u using forward finite difference.

        Parameters
        ----------
        u : float or ndarray
        v : float or ndarray
        param : ndarray, optional
            Copula parameters.

        Returns
        -------
        float or ndarray
            Approximate ∂C/∂u.
        """
        if param is None:
            param = self.parameters
        eps = 1e-6
        c = self.get_cdf
        return (c(u+eps, v, param) - c(u, v, param)) / eps

    def partial_derivative_C_wrt_v(self, u, v, param: np.ndarray = None):
        """
        Approximate ∂C/∂v using forward finite difference.

        Returns
        -------
        float or ndarray
            Approximate ∂C/∂v.
        """
        if param is None:
            param = self.parameters
        eps = 1e-6
        c = self.get_cdf
        return (c(u, v+eps, param) - c(u, v, param)) / eps

    def conditional_cdf_v_given_u(self, u, v, param: np.ndarray = None):
        """
        Compute P(V ≤ v | U = u) = ∂C/∂u(u,v).

        Returns
        -------
        float or ndarray
        """
        return self.partial_derivative_C_wrt_u(u, v, param)

    def conditional_cdf_u_given_v(self, u, v, param: np.ndarray = None):
        """
        Compute P(U ≤ u | V = v) = ∂C/∂v(u,v).

        Returns
        -------
        float or ndarray
        """
        return self.partial_derivative_C_wrt_v(u, v, param)

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
        np.ndarray of shape (n,2)
            Sampled pseudo-observations.
        """
        if param is None:
            param = self.parameters
        samples = np.empty((n,2))
        eps = 1e-6
        for i in range(n):
            u = np.random.rand()
            p = np.random.rand()
            root = root_scalar(
                lambda vv: self.partial_derivative_C_wrt_u(u, vv, param) - p,
                bracket=[eps, 1-eps], method='bisect', xtol=1e-6
            )
            samples[i] = [u, root.root]
        return samples

    def LTDC(self, param: np.ndarray = None) -> float:
        """
        Approximate lower-tail dependence λ_L = lim_{u→0} C(u,u)/u.

        Returns
        -------
        float
        """
        if param is None:
            param = self.parameters
        u = 1e-6
        return self.get_cdf(u,u,param) / u

    def UTDC(self, param: np.ndarray = None) -> float:
        """
        Approximate upper-tail dependence λ_U = lim_{u→1} (1-2u+C(u,u))/(1-u).

        Returns
        -------
        float
        """
        if param is None:
            param = self.parameters
        u = 1 - 1e-6
        return (1 - 2*u + self.get_cdf(u,u,param)) / (1-u)

    def kendall_tau(self, param: np.ndarray = None) -> float:
        """
        Not implemented for BB8.
        """
        raise NotImplementedError("Kendall's tau not implemented for BB8.")
