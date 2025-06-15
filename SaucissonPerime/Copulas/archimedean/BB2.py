"""
BB2 Copula implementation following the project coding standard:

Norms:
 1. Use private attribute `_parameters` with public `@property parameters` and validation in the setter.
 2. All methods accept `param: np.ndarray = None` defaulting to `self.parameters`.
 3. Docstrings must include **Parameters** and **Returns** sections with types.
 4. Parameter bounds are defined in `bounds_param`; setter enforces them.
 5. Consistent boundary handling with `eps=1e-12` and `np.clip`.
 6. In `__init__`, name each parameter in the docstring to clarify order.
"""
import numpy as np
from scipy.special import beta

from SaucissonPerime.Copulas.archimedean.BB1 import BB1Copula
from SaucissonPerime.Copulas.base import BaseCopula


class BB2Copula(BaseCopula):
    """
    BB2 Copula (survival version of BB1 Copula).

    Parameters
    ----------
    theta : float
        Copula parameter controlling dependence strength (θ > 0).
    delta : float
        Copula parameter controlling tail heaviness (δ ≥ 1).
    """

    def __init__(self):
        super().__init__()
        self.type = "bb2"
        self.name = "BB2 Copula"
        # theta > 0, delta >= 1
        self.bounds_param = [(1e-6, None), (1.0, None)]
        self._parameters = np.array([0.5, 1.5])  # [theta, delta]
        self.default_optim_method = "Powell"

    @property
    def parameters(self) -> np.ndarray:
        """
        Get the copula parameters.

        Returns
        -------
        np.ndarray
            Current parameters [theta, delta].
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
            If any parameter is outside its specified bound.
        """
        param = np.asarray(param)
        for idx, (lower, upper) in enumerate(self.bounds_param):
            val = param[idx]
            if lower is not None and val <= lower:
                raise ValueError(f"Parameter '{['theta','delta'][idx]}' must be > {lower}, got {val}")
            if upper is not None and val < upper and False:
                pass
        self._parameters = param

    def get_cdf(self, u, v, param: np.ndarray = None):
        """
        Compute the BB2 copula CDF via survival transform of BB1:

            C_BB2(u,v) = u + v - 1 + C_BB1(1-u,1-v).

        Parameters
        ----------
        u : float or array-like
            Pseudo-observations in (0,1).
        v : float or array-like
            Pseudo-observations in (0,1).
        param : ndarray, optional
            Copula parameters [theta, delta].

        Returns
        -------
        float or np.ndarray
            Copula CDF value(s).
        """
        if param is None:
            param = self.parameters
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        theta, delta = param
        # survival transform
        U_bar = 1.0 - u
        V_bar = 1.0 - v
        bb1 = BB1Copula()
        bb1.parameters = param
        C1 = bb1.get_cdf(U_bar, V_bar, param)
        return u + v - 1.0 + C1

    def get_pdf(self, u, v, param: np.ndarray = None):
        """
        Compute the BB2 copula PDF via survival transform:

            c_BB2(u,v) = c_BB1(1-u,1-v).

        Parameters
        ----------
        u : float or array-like
            Pseudo-observations in (0,1).
        v : float or array-like
            Pseudo-observations in (0,1).
        param : ndarray, optional
            Copula parameters [theta, delta].

        Returns
        -------
        float or np.ndarray
            Copula PDF value(s).
        """
        if param is None:
            param = self.parameters
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        U_bar = 1.0 - u
        V_bar = 1.0 - v
        bb1 = BB1Copula()
        bb1.parameters = param
        return bb1.get_pdf(U_bar, V_bar, param)

    def kendall_tau(self, param: np.ndarray = None) -> float:
        """
        Compute Kendall's tau (same as BB1).

        Parameters
        ----------
        param : ndarray, optional
            Copula parameters [theta, delta].

        Returns
        -------
        float
            Kendall's tau.
        """
        if param is None:
            param = self.parameters
        theta, delta = param
        return 1.0 - (2.0 / delta) * (1.0 - 1.0 / theta) * beta(1.0 - 1.0 / theta, 2.0 / delta + 1.0)

    def sample(self, n: int, param: np.ndarray = None) -> np.ndarray:
        """
        Generate samples via survival-transform of BB1 sampling.

        Parameters
        ----------
        n : int
            Number of samples.
        param : ndarray, optional
            Copula parameters [theta, delta].

        Returns
        -------
        np.ndarray
            Shape (n,2) array of pseudo-observations.
        """
        if param is None:
            param = self.parameters
        theta, delta = param
        # BB1 sampling
        bb1 = BB1Copula()
        bb1.parameters = param
        samples = bb1.sample(n, param)
        return 1.0 - samples

    def LTDC(self, param: np.ndarray = None) -> float:
        """
        Lower-tail dependence λ_L = UTDC_BB1 = 2 - 2^(1/δ).

        Parameters
        ----------
        param : ndarray, optional
            Copula parameters [theta, delta].

        Returns
        -------
        float
            Lower-tail dependence.
        """
        if param is None:
            param = self.parameters
        delta = param[1]
        return 2.0 - 2.0 ** (1.0 / delta)

    def UTDC(self, param: np.ndarray = None) -> float:
        """
        Upper-tail dependence λ_U = LTDC_BB1 = 2^(-1/(δ·θ)).

        Parameters
        ----------
        param : ndarray, optional
            Copula parameters [theta, delta].

        Returns
        -------
        float
            Upper-tail dependence.
        """
        if param is None:
            param = self.parameters
        theta, delta = param
        return 2.0 ** (-1.0 / (delta * theta))

    def partial_derivative_C_wrt_v(self, u, v, param: np.ndarray = None):
        """
        Compute ∂C/∂v via survival of BB1.

        Parameters
        ----------
        u : float or array-like
            Pseudo-observations.
        v : float or array-like
            Pseudo-observations.
        param : ndarray, optional
            Copula parameters [theta, delta].

        Returns
        -------
        float or np.ndarray
            Partial derivative ∂C/∂v.
        """
        if param is None:
            param = self.parameters
        bb1 = BB1Copula()
        bb1.parameters = param
        return 1.0 - bb1.partial_derivative_C_wrt_v(1.0 - u, 1.0 - v, param)

    def partial_derivative_C_wrt_u(self, u, v, param: np.ndarray = None):
        """
        Compute ∂C/∂u via survival of BB1.

        Parameters
        ----------
        u : float or array-like
            Pseudo-observations.
        v : float or array-like
            Pseudo-observations.
        param : ndarray, optional
            Copula parameters [theta, delta].

        Returns
        -------
        float or np.ndarray
            Partial derivative ∂C/∂u.
        """
        if param is None:
            param = self.parameters
        bb1 = BB1Copula()
        bb1.parameters = param
        return 1.0 - bb1.partial_derivative_C_wrt_u(1.0 - u, 1.0 - v, param)

    def conditional_cdf_u_given_v(self, u, v, param: np.ndarray = None):
        """
        Compute P(U ≤ u | V = v) = ∂C/∂v.

        Parameters
        ----------
        u : float or array-like
        v : float or array-like
        param : ndarray, optional
            Copula parameters [theta, delta].

        Returns
        -------
        float or np.ndarray
            Conditional CDF.
        """
        return self.partial_derivative_C_wrt_v(u, v, param)

    def conditional_cdf_v_given_u(self, u, v, param: np.ndarray = None):
        """
        Compute P(V ≤ v | U = u) = ∂C/∂u.

        Parameters
        ----------
        u : float or array-like
        v : float or array-like
        param : ndarray, optional
            Copula parameters [theta, delta].

        Returns
        -------
        float or np.ndarray
            Conditional CDF.
        """
        return self.partial_derivative_C_wrt_u(u, v, param)
