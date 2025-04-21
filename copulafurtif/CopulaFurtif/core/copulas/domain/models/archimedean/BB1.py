"""
BB1 Copula implementation following the project coding standard:

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

from Service.Copulas.base import BaseCopula


class BB1Copula(BaseCopula):
    """
    BB1 Copula (Two-parameter Archimedean copula).

    Parameters
    ----------
    theta : float
        Copula parameter controlling dependence strength (θ > 0).
    delta : float
        Copula parameter controlling tail heaviness (δ ≥ 1).

    Defined by:
        C(u,v) = [1 + ({u**(-θ) - 1}**δ + {v**(-θ) - 1}**δ)**(1/δ)]**(-1/θ)
    """

    def __init__(self):
        super().__init__()
        self.type = "bb1"
        self.name = "BB1 Copula"
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
                pass  # upper is None or no upper bound for theta, but delta has lower bound only
        self._parameters = param

    def get_cdf(self, u, v, param: np.ndarray = None):
        """
        Compute the BB1 copula CDF C(u,v).

        Parameters
        ----------
        u : float or array-like
            Pseudo-observations in (0,1).
        v : float or array-like
            Pseudo-observations in (0,1).
        param : ndarray, optional
            Copula parameters [theta, delta]. If None, uses self.parameters.

        Returns
        -------
        float or np.ndarray
            Copula CDF value(s).
        """
        if param is None:
            param = self.parameters
        theta, delta = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        term1 = (u ** (-theta) - 1) ** delta
        term2 = (v ** (-theta) - 1) ** delta
        inner = term1 + term2
        return (1 + inner ** (1.0 / delta)) ** (-1.0 / theta)

    def get_pdf(self, u, v, param: np.ndarray = None):
        """
        Compute the BB1 copula PDF c(u,v) via closed-form expression.

        Parameters
        ----------
        u : float or array-like
            Pseudo-observations in (0,1).
        v : float or array-like
            Pseudo-observations in (0,1).
        param : ndarray, optional
            Copula parameters [theta, delta]. If None, uses self.parameters.

        Returns
        -------
        float or np.ndarray
            Copula PDF value(s).
        """
        if param is None:
            param = self.parameters
        theta, delta = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        x = (u ** (-theta) - 1) ** delta
        y = (v ** (-theta) - 1) ** delta
        S = x + y
        term1 = (1 + S ** (1.0 / delta)) ** (-1.0 / theta - 2)
        term2 = S ** (1.0 / delta - 2)
        term3 = theta * (delta - 1) + (theta * delta + 1) * S ** (1.0 / delta)
        term4 = (x * y) ** (1 - 1.0 / delta) * (u * v) ** (-theta - 1)
        return term1 * term2 * term3 * term4

    def kendall_tau(self, param: np.ndarray = None) -> float:
        """
        Compute Kendall's tau for BB1 copula:

            τ = 1 - (2/δ)*(1 - 1/θ)*B(1 - 1/θ, 2/δ + 1)

        Parameters
        ----------
        param : ndarray, optional
            Copula parameters [theta, delta]. If None, uses self.parameters.

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
        Generate samples via Marshall–Olkin-type method.

        Parameters
        ----------
        n : int
            Number of samples.
        param : ndarray, optional
            Copula parameters [theta, delta]. If None, uses self.parameters.

        Returns
        -------
        np.ndarray
            Shape (n,2) array of pseudo-observations.
        """
        if param is None:
            param = self.parameters
        theta, delta = param
        if theta <= 0 or delta < 1:
            raise ValueError("Parameters must satisfy theta > 0 and delta >= 1.")
        V = np.random.gamma(1.0 / delta, 1.0, size=n)
        E1 = np.random.exponential(scale=1.0, size=n)
        E2 = np.random.exponential(scale=1.0, size=n)
        U = (1 + (E1 / V) ** (1.0 / delta)) ** (-1.0 / theta)
        W = (1 + (E2 / V) ** (1.0 / delta)) ** (-1.0 / theta)
        return np.column_stack((U, W))

    def LTDC(self, param: np.ndarray = None) -> float:
        """
        Compute lower tail dependence λ_L = 2^(-1/(δ·θ)).

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
        theta, delta = param
        return 2.0 ** (-1.0 / (delta * theta))

    def UTDC(self, param: np.ndarray = None) -> float:
        """
        Compute upper tail dependence λ_U = 2 - 2^(1/δ).

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
        delta = param[1]
        return 2.0 - 2.0 ** (1.0 / delta)

    def partial_derivative_C_wrt_v(self, u, v, param: np.ndarray = None):
        """
        Compute ∂C/∂v for BB1 copula.

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
        theta, delta = param
        u = np.asarray(u)
        v = np.asarray(v)
        T = (u ** (-theta) - 1) ** delta + (v ** (-theta) - 1) ** delta
        factor = (1 + T ** (1.0 / delta)) ** (-1.0 / theta - 1)
        return factor * T ** (1.0 / delta - 1) * (v ** (-theta) - 1) ** (delta - 1) * v ** (-theta - 1)

    def partial_derivative_C_wrt_u(self, u, v, param: np.ndarray = None):
        """
        Compute ∂C/∂u for BB1 copula.

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
        theta, delta = param
        u = np.asarray(u)
        v = np.asarray(v)
        T = (u ** (-theta) - 1) ** delta + (v ** (-theta) - 1) ** delta
        factor = (1 + T ** (1.0 / delta)) ** (-1.0 / theta - 1)
        return factor * T ** (1.0 / delta - 1) * (u ** (-theta) - 1) ** (delta - 1) * u ** (-theta - 1)

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
