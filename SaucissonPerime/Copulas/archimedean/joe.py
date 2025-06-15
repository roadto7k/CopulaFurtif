from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import uniform
from SaucissonPerime.Copulas.base import BaseCopula


class JoeCopula(BaseCopula):
    """
    Joe Copula (Archimedean copula).

    Parameters
    ----------
    theta : float
        Dependence parameter (θ ≥ 1). Higher θ increases upper-tail dependence.

    Attributes
    ----------
    type : str
        Copula family identifier ('joe').
    name : str
        Human-readable name.
    bounds_param : list[tuple]
        Parameter bounds: [(1.0, None)].
    param_names : list[str]
        Names of copula parameters: ['theta'].
    _parameters : np.ndarray
        Internal storage of copula parameters [theta].
    default_optim_method : str
        Default optimizer for fitting.
    """

    def __init__(self):
        super().__init__()
        self.type = 'joe'
        self.name = 'Joe Copula'
        self.bounds_param = [(1.0, None)]
        self.param_names = ['theta']
        self._parameters = np.array([1.5])  # [theta]
        self.default_optim_method = 'SLSQP'

    @property
    def parameters(self) -> np.ndarray:
        """Current copula parameters."""
        return self._parameters

    @parameters.setter
    def parameters(self, param: np.ndarray):
        """Validate and set copula parameters."""
        arr = np.asarray(param, dtype=float)
        theta = arr[0]
        if theta < 1.0:
            raise ValueError("JoeCopula: parameter 'theta' must be >= 1.")
        self._parameters = arr

    def get_cdf(self,
                u: float | np.ndarray,
                v: float | np.ndarray,
                param: np.ndarray | None = None
               ) -> float | np.ndarray:
        """
        CDF of Joe copula: C(u,v) = 1 - [ (1-u)^θ + (1-v)^θ - (1-u)^θ(1-v)^θ ]^(1/θ).

        Parameters
        ----------
        u : float or array-like
            First uniform margin in (0,1).
        v : float or array-like
            Second uniform margin in (0,1).
        param : np.ndarray, optional
            Copula parameters [theta]. If None, uses self.parameters.

        Returns
        -------
        float or np.ndarray
            Copula CDF at (u, v).
        """
        if param is None:
            theta = self.parameters[0]
        else:
            theta = param[0]
        eps = 1e-12
        u_clipped = np.clip(u, eps, 1 - eps)
        v_clipped = np.clip(v, eps, 1 - eps)

        u_pow = (1.0 - u_clipped) ** theta
        v_pow = (1.0 - v_clipped) ** theta
        base = u_pow + v_pow - u_pow * v_pow
        return 1.0 - base ** (1.0 / theta)

    def get_pdf(self,
                u: float | np.ndarray,
                v: float | np.ndarray,
                param: np.ndarray | None = None
               ) -> float | np.ndarray:
        """
        PDF of Joe copula.

        Parameters
        ----------
        u : float or array-like
            First uniform margin in (0,1).
        v : float or array-like
            Second uniform margin in (0,1).
        param : np.ndarray, optional
            Copula parameters [theta]. If None, uses self.parameters.

        Returns
        -------
        float or np.ndarray
            Copula PDF at (u, v).
        """
        if param is None:
            theta = self.parameters[0]
        else:
            theta = param[0]
        eps = 1e-12
        u_clipped = np.clip(u, eps, 1 - eps)
        v_clipped = np.clip(v, eps, 1 - eps)

        u_pow = (1.0 - u_clipped) ** theta
        v_pow = (1.0 - v_clipped) ** theta
        base = u_pow + v_pow - u_pow * v_pow
        term1 = base ** (-2.0 + 1.0 / theta)
        term2 = (1.0 - u_clipped) ** (theta - 1.0) * (1.0 - v_clipped) ** (theta - 1.0)
        term3 = (theta - 1.0) + u_pow + v_pow + u_pow * v_pow
        return term1 * term2 * term3

    def kendall_tau(self,
                    param: np.ndarray | None = None
                   ) -> float:
        """
        Estimate Kendall's tau via numerical integration of generator ratio.

        τ = 1 + 4 ∫₀¹ φ(t)/φ'(t) dt
        """
        if param is None:
            theta = self.parameters[0]
        else:
            theta = param[0]
        if theta <= 1.0:
            return 0.0

        def integrand(t: float) -> float:
            t_clip = np.clip(t, 1e-12, 1 - 1e-12)
            numerator = -np.log(1.0 - (1.0 - t_clip) ** theta)
            numerator *= (1.0 - (1.0 - t_clip) ** theta)
            denom = theta * (1.0 - t_clip) ** (theta - 1.0)
            return numerator / denom

        integral_val, _ = quad(integrand, 0.0, 1.0, limit=100)
        return 1.0 + 4.0 * integral_val

    def sample(self,
               n: int,
               param: np.ndarray | None = None
              ) -> np.ndarray:
        """
        Generate samples via conditional inversion on C_{2|1}.

        1. u ~ Uniform(0,1)
        2. w ~ Uniform(0,1)
        3. solve ∂C/∂u(u,v) = w for v.
        """
        if param is None:
            theta = self.parameters[0]
        else:
            theta = param[0]
        eps = 1e-12
        u_samples = uniform.rvs(size=n)
        v_samples = np.empty(n)
        for i, ui in enumerate(u_samples):
            w = uniform.rvs()
            func = lambda vv: self.partial_derivative_C_wrt_u(ui, vv, param) - w
            try:
                v_samples[i] = brentq(func, eps, 1 - eps)
            except ValueError:
                v_samples[i] = uniform.rvs()
        return np.column_stack((u_samples, v_samples))

    def LTDC(self,
             param: np.ndarray | None = None
            ) -> float:
        """Lower tail dependence coefficient (always 0)."""
        return 0.0

    def UTDC(self,
             param: np.ndarray | None = None
            ) -> float:
        """
        Upper tail dependence λ_U = 2 − 2^(1/θ).
        """
        if param is None:
            theta = self.parameters[0]
        else:
            theta = param[0]
        return 2.0 - 2.0 ** (1.0 / theta)

    def partial_derivative_C_wrt_v(self,
                                   u: float | np.ndarray,
                                   v: float | np.ndarray,
                                   param: np.ndarray | None = None
                                  ) -> float | np.ndarray:
        """
        Partial ∂C/∂v for conditional CDF C_{1|2}.
        """
        if param is None:
            theta = self.parameters[0]
        else:
            theta = param[0]
        eps = 1e-12
        u_clipped = np.clip(u, eps, 1 - eps)
        v_clipped = np.clip(v, eps, 1 - eps)
        A = (1 - u_clipped) ** theta + (1 - v_clipped) ** theta
        A -= (1 - u_clipped) ** theta * (1 - v_clipped) ** theta
        return A ** (1.0 / theta - 1.0) * (1 - v_clipped) ** (theta - 1.0) * (1 - (1 - u_clipped) ** theta)

    def partial_derivative_C_wrt_u(self,
                                   u: float | np.ndarray,
                                   v: float | np.ndarray,
                                   param: np.ndarray | None = None
                                  ) -> float | np.ndarray:
        """
        Partial ∂C/∂u for conditional CDF C_{2|1}.
        """
        if param is None:
            theta = self.parameters[0]
        else:
            theta = param[0]
        eps = 1e-12
        u_clipped = np.clip(u, eps, 1 - eps)
        v_clipped = np.clip(v, eps, 1 - eps)
        A = (1 - u_clipped) ** theta + (1 - v_clipped) ** theta
        A -= (1 - u_clipped) ** theta * (1 - v_clipped) ** theta
        return A ** (1.0 / theta - 1.0) * (1 - u_clipped) ** (theta - 1.0) * (1 - (1 - v_clipped) ** theta)

    def conditional_cdf_u_given_v(self,
                                  u: float | np.ndarray,
                                  v: float | np.ndarray,
                                  param: np.ndarray | None = None
                                 ) -> float | np.ndarray:
        """
        P(U ≤ u | V = v) = ∂C/∂v (since ∂C(1,v)/∂v = 1).
        """
        return self.partial_derivative_C_wrt_v(u, v, param)

    def conditional_cdf_v_given_u(self,
                                  u: float | np.ndarray,
                                  v: float | np.ndarray,
                                  param: np.ndarray | None = None
                                 ) -> float | np.ndarray:
        """
        P(V ≤ v | U = u) = ∂C/∂u (since ∂C(u,1)/∂u = 1).
        """
        return self.partial_derivative_C_wrt_u(u, v, param)
