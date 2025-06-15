"""
BB3 Copula implementation following the project coding standard:

Norms:
 1. Use private `_parameters` with public `@property parameters` and validation in setter.
 2. All methods accept `param: np.ndarray = None` defaulting to `self.parameters`.
 3. Docstrings include **Parameters** and **Returns** with types.
 4. Parameter bounds in `bounds_param`; setter enforces them.
 5. Uniform boundary clipping with `eps=1e-12` and `np.clip` where needed.
 6. Document parameter names (`d`, `q`) in `__init__` docstring.
"""
import numpy as np
from scipy.optimize import brentq

from SaucissonPerime.Copulas.base import BaseCopula


class BB3Copula(BaseCopula):
    """
    BB3 Copula (Two-parameter Archimedean copula).

    Parameters
    ----------
    d : float
        Parameter controlling generator scale (d > 0).
    q : float
        Parameter controlling generator shape (q ≥ 1).
    """
    def __init__(self):
        super().__init__()
        self.type = "bb3"
        self.name = "BB3 Copula"
        # d > 0, q >= 1
        self.bounds_param = [(1e-6, None), (1.0, None)]
        self._parameters = np.array([1.0, 1.0])  # [d, q]
        self.default_optim_method = "Powell"

    @property
    def parameters(self) -> np.ndarray:
        """
        Get the copula parameters.

        Returns
        -------
        np.ndarray
            Current parameters [d, q].
        """
        return self._parameters

    @parameters.setter
    def parameters(self, param: np.ndarray):
        """
        Set and validate copula parameters against bounds_param.

        Parameters
        ----------
        param : array-like
            New parameters [d, q].

        Raises
        ------
        ValueError
            If any value is outside its specified bound.
        """
        param = np.asarray(param)
        for idx, (lower, upper) in enumerate(self.bounds_param):
            val = param[idx]
            name = ['d', 'q'][idx]
            if lower is not None and val <= lower:
                raise ValueError(f"Parameter '{name}' must be > {lower}, got {val}")
            if upper is not None and val < upper and False:
                pass
        self._parameters = param

    def _h(self, s, param: np.ndarray = None):
        """
        Generator h(s) = [(log(1+s)/d)]^(1/q).

        Parameters
        ----------
        s : float or ndarray
            Generator argument.
        param : ndarray, optional
            Copula parameters [d, q].

        Returns
        -------
        float or ndarray
            Generator value h(s).
        """
        if param is None:
            param = self.parameters
        d, q = param
        return (np.log1p(s) / d) ** (1.0 / q)

    def _h_prime(self, s, param: np.ndarray = None):
        """
        Derivative h'(s) of generator.

        Parameters
        ----------
        s : float or ndarray
        param : ndarray, optional

        Returns
        -------
        float or ndarray
        """
        if param is None:
            param = self.parameters
        d, q = param
        g = np.log1p(s) / d
        return (1.0 / (q * d * (1.0 + s))) * g ** (1.0 / q - 1.0)

    def _h_double(self, s, param: np.ndarray = None):
        """
        Second derivative h''(s) of generator.

        Parameters
        ----------
        s : float or ndarray
        param : ndarray, optional

        Returns
        -------
        float or ndarray
        """
        if param is None:
            param = self.parameters
        d, q = param
        g = np.log1p(s) / d
        A = 1.0 / (q * d * (1.0 + s) ** 2)
        term1 = -g ** (1.0 / q - 1.0)
        term2 = (1.0 / q - 1.0) * g ** (1.0 / q - 2.0) / d
        return A * (term1 + term2)

    def get_cdf(self, u, v, param: np.ndarray = None):
        """
        Compute copula CDF C(u,v) = exp(-h(s)), s = phi^{-1}(u)+phi^{-1}(v).

        Parameters
        ----------
        u : float or ndarray
            Pseudo-observation(s) in (0,1).
        v : float or ndarray
            Pseudo-observation(s) in (0,1).
        param : ndarray, optional
            Copula parameters [d, q].

        Returns
        -------
        float or ndarray
            Copula CDF value(s).
        """
        if param is None:
            param = self.parameters
        d, q = param
        # domain clipping
        eps_dom = 1e-12
        u = np.clip(u, eps_dom, 1 - eps_dom)
        v = np.clip(v, eps_dom, 1 - eps_dom)
        # threshold clipping uses a larger epsilon to bound exp argument
        eps_max = 1e-3
        max_exp = d * (-np.log(eps_max)) ** q
        # compute generator inputs
        lu = -np.log(u)
        lv = -np.log(v)
        arg_u = np.minimum(d * lu ** q, max_exp)
        arg_v = np.minimum(d * lv ** q, max_exp)
        s_u = np.expm1(arg_u)
        s_v = np.expm1(arg_v)
        s = s_u + s_v
        return np.exp(-self._h(s, param))

    def get_pdf(self, u, v, param: np.ndarray = None):
        """
        Compute copula PDF c(u,v) = phi''(s)*phi_inv_u'*phi_inv_v'.

        Parameters
        ----------
        u : float or ndarray
            Pseudo-observation(s) in (0,1).
        v : float or ndarray
            Pseudo-observation(s) in (0,1).
        param : ndarray, optional
            Copula parameters [d, q].

        Returns
        -------
        float or ndarray
            Copula PDF value(s).
        """
        if param is None:
            param = self.parameters
        d, q = param
        # domain clipping
        eps_dom = 1e-12
        u = np.clip(u, eps_dom, 1 - eps_dom)
        v = np.clip(v, eps_dom, 1 - eps_dom)
        # threshold clipping uses larger epsilon
        eps_max = 1e-3
        max_exp = d * (-np.log(eps_max)) ** q
        # logs
        lu = -np.log(u)
        lv = -np.log(v)
        arg_u = np.minimum(d * lu ** q, max_exp)
        arg_v = np.minimum(d * lv ** q, max_exp)
        s_u = np.expm1(arg_u)
        s_v = np.expm1(arg_v)
        s = s_u + s_v
        # generator derivatives
        h = self._h(s, param)
        h1 = self._h_prime(s, param)
        h2 = self._h_double(s, param)
        phi_dd = np.exp(-h) * (h1 ** 2 - h2)
        # inverse generator derivatives
        phi_inv_u_prime = -d * q * np.exp(np.minimum(d * lu ** q, max_exp)) * (lu ** (q - 1) / u)
        phi_inv_v_prime = -d * q * np.exp(np.minimum(d * lv ** q, max_exp)) * (lv ** (q - 1) / v)
        return phi_dd * phi_inv_u_prime * phi_inv_v_prime

    def kendall_tau(self, param: np.ndarray = None, n: int = 201) -> float:
        """
        Estimate Kendall's tau by double integral:

            τ = 4 ∫_0^1 ∫_0^1 C(u,v) du dv - 1.

        Parameters
        ----------
        param : ndarray, optional
            Copula parameters [d, q].
        n : int
            Grid points per dimension (default=201).

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
        Generate samples (u,v) via conditional inversion.

        Parameters
        ----------
        n : int
            Number of samples.
        param : ndarray, optional
            Copula parameters [d, q].

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
                lambda v_val: self.conditional_cdf_v_given_u(u, v_val, param) - p,
                1e-6, 1 - 1e-6
            )
            samples[i] = [u, root]
        return samples

    def LTDC(self, param: np.ndarray = None) -> float:
        """
        Lower-tail dependence λ_L = 0 for BB3.

        Returns
        -------
        float
            0.0
        """
        return 0.0

    def UTDC(self, param: np.ndarray = None) -> float:
        """
        Upper-tail dependence λ_U = 2 - 2^(1/q).

        Parameters
        ----------
        param : ndarray, optional
            Copula parameters [d, q].

        Returns
        -------
        float
            Upper-tail dependence.
        """
        if param is None:
            param = self.parameters
        q = param[1]
        return 2.0 - 2.0 ** (1.0 / q)

    def partial_derivative_C_wrt_v(self, u, v, param: np.ndarray = None):
        """
        Compute partial ∂C/∂v.

        Parameters
        ----------
        u, v : float or ndarray
            Pseudo-observations in (0,1).
        param : ndarray, optional
            Copula parameters [d, q].

        Returns
        -------
        float or ndarray
            ∂C/∂v.
        """
        if param is None:
            param = self.parameters
        d, q = param
        # domain clipping
        eps_dom = 1e-12
        u = np.clip(u, eps_dom, 1 - eps_dom)
        v = np.clip(v, eps_dom, 1 - eps_dom)
        # threshold clipping
        eps_max = 1e-3
        max_exp = d * (-np.log(eps_max)) ** q
        # logs
        lu = -np.log(u)
        lv = -np.log(v)
        arg_u = np.minimum(d * lu ** q, max_exp)
        arg_v = np.minimum(d * lv ** q, max_exp)
        s_u = np.expm1(arg_u)
        s_v = np.expm1(arg_v)
        s = s_u + s_v
        # derivative
        h1 = self._h_prime(s, param)
        phi_p = -h1 * np.exp(-self._h(s, param))
        phi_inv_v_prime = -d * q * np.exp(arg_v) * (lv ** (q - 1) / v)
        return phi_p * phi_inv_v_prime

    def partial_derivative_C_wrt_u(self, u, v, param: np.ndarray = None):
        """
        Compute partial ∂C/∂u.

        Parameters
        ----------
        u, v : float or ndarray
            Pseudo-observations in (0,1).
        param : ndarray, optional
            Copula parameters [d, q].

        Returns
        -------
        float or ndarray
            ∂C/∂u.
        """
        if param is None:
            param = self.parameters
        d, q = param
        # domain clipping
        eps_dom = 1e-12
        u = np.clip(u, eps_dom, 1 - eps_dom)
        v = np.clip(v, eps_dom, 1 - eps_dom)
        # threshold clipping
        eps_max = 1e-3
        max_exp = d * (-np.log(eps_max)) ** q
        # logs
        lu = -np.log(u)
        lv = -np.log(v)
        arg_u = np.minimum(d * lu ** q, max_exp)
        arg_v = np.minimum(d * lv ** q, max_exp)
        s_u = np.expm1(arg_u)
        s_v = np.expm1(arg_v)
        s = s_u + s_v
        # derivative
        h1 = self._h_prime(s, param)
        phi_p = -h1 * np.exp(-self._h(s, param))
        phi_inv_u_prime = -d * q * np.exp(arg_u) * (lu ** (q - 1) / u)
        return phi_p * phi_inv_u_prime

    def conditional_cdf_u_given_v(self, u, v, param: np.ndarray = None):
        """
        Compute P(U ≤ u | V = v).

        Parameters
        ----------
        u, v : float or ndarray
            Pseudo-observations in (0,1).
        param : ndarray, optional
            Copula parameters [d, q].

        Returns
        -------
        float or ndarray
            Conditional CDF value.
        """
        if param is None:
            param = self.parameters
        num = self.partial_derivative_C_wrt_v(u, v, param)
        den = self.partial_derivative_C_wrt_v(1.0, v, param)
        return num / den

    def conditional_cdf_v_given_u(self, u, v, param: np.ndarray = None):
        """
        Compute P(V ≤ v | U = u).

        Parameters
        ----------
        u, v : float or ndarray
            Pseudo-observations in (0,1).
        param : ndarray, optional
            Copula parameters [d, q].

        Returns
        -------
        float or ndarray
            Conditional CDF value.
        """
        if param is None:
            param = self.parameters
        num = self.partial_derivative_C_wrt_u(u, v, param)
        den = self.partial_derivative_C_wrt_u(u, 1.0, param)
        return num / den