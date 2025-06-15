import numpy as np
from scipy.stats import uniform
from scipy.optimize import brentq
from SaucissonPerime.Copulas.base import BaseCopula


class PlackettCopula(BaseCopula):
    """
    Plackett Copula class (also known as the discrete Fréchet–Hoeffding upper bound copula).

    Attributes
    ----------
    type : str
        Identifier for the copula family: "plackett".
    name : str
        Human-readable name: "Plackett Copula".
    bounds_param : list of tuple
        Parameter bounds for theta (θ > 0).
    _parameters : np.ndarray
        Internal storage of the copula parameter [θ].
    default_optim_method : str
        Default optimizer for parameter estimation.
    """

    def __init__(self):
        super().__init__()
        self.type = "plackett"
        self.name = "Plackett Copula"
        self.bounds_param = [(1e-6, None)]  # θ ∈ (0, ∞)
        self._parameters = np.array([0.5])  # initial guess for θ
        self.default_optim_method = "SLSQP"

    @property
    def parameters(self) -> np.ndarray:
        """
        Get the current copula parameter array.
        Returns
        -------
        np.ndarray
            Array containing θ.
        """
        return self._parameters

    @parameters.setter
    def parameters(self, param: np.ndarray):
        """
        Set the copula parameter θ, enforcing bounds.

        Parameters
        ----------
        param : array_like
            New parameter array [θ].

        Raises
        ------
        ValueError
            If θ is outside (0, ∞).
        """
        arr = np.asarray(param, float)
        theta = arr[0]
        low, high = self.bounds_param[0]
        if theta <= low or (high is not None and theta >= high):
            raise ValueError(f"theta must be in ({low}, {high}), got {theta}")
        self._parameters = arr

    def get_cdf(self,
                u: np.ndarray,
                v: np.ndarray,
                param: np.ndarray = None
               ) -> np.ndarray:
        """
        Compute the Plackett copula CDF C(u,v;θ).

        C(u,v) = 1/(2δ) [A - sqrt(A² - 4θδ u v)],
        where δ = θ - 1 and A = 1 + δ(u+v).

        Parameters
        ----------
        u : array_like
            Input values for U, in [0,1].
        v : array_like
            Input values for V, in [0,1].
        param : array_like, optional
            Parameter array [θ]. If None, uses self.parameters.

        Returns
        -------
        np.ndarray
            Copula CDF values at each (u,v).
        """
        if param is None:
            param = self.parameters
        theta = float(param[0])
        delta = theta - 1.0

        u = np.clip(u, 0.0, 1.0)
        v = np.clip(v, 0.0, 1.0)

        A = 1.0 + delta * (u + v)
        B = np.sqrt(A**2 - 4.0 * theta * delta * u * v)
        return 0.5 / delta * (A - B)

    def get_pdf(self,
                u: np.ndarray,
                v: np.ndarray,
                param: np.ndarray = None
               ) -> np.ndarray:
        """
        Compute the Plackett copula PDF c(u,v;θ).

        Formula:
            c(u,v) = θ [1 + δ(s - 2uv)] / [ ( (1+δs)² - 4θδuv )^(3/2) ],
        where s = u+v and δ = θ-1.

        Parameters
        ----------
        u, v : array_like
            Input values in [0,1].
        param : array_like, optional
            Parameter array [θ].

        Returns
        -------
        np.ndarray
            Density values at each (u,v).
        """
        if param is None:
            param = self.parameters
        theta = float(param[0])
        delta = theta - 1.0

        u = np.clip(u, 0.0, 1.0)
        v = np.clip(v, 0.0, 1.0)

        s = u + v
        p = u * v
        num = theta * (1.0 + delta * (s - 2.0 * p))
        inner = (1.0 + delta * s)**2 - 4.0 * theta * delta * p
        return num / inner**1.5

    def kendall_tau(self,
                    param: np.ndarray = None
                   ) -> float:
        """
        Analytical Kendall's tau for the Plackett copula:

            τ(θ) = 1 - 2(θ+1)/(3θ) + 2(θ−1)² ln θ / (3θ²).

        Parameters
        ----------
        param : array_like, optional
            Parameter array [θ].

        Returns
        -------
        float
            Kendall's τ.

        Raises
        ------
        ValueError
            If θ ≤ 0.
        """
        if param is None:
            param = self.parameters
        theta = float(param[0])
        if theta <= 0:
            raise ValueError("theta must be > 0")
        if abs(theta - 1.0) < 1e-8:
            return 0.0
        return (
            1.0
            - 2*(theta + 1)/(3*theta)
            + 2*(theta - 1)**2 * np.log(theta)/(3*theta**2)
        )

    def _cond_cdf_v_given_u(self,
                            u: float,
                            v: float,
                            theta: float
                           ) -> float:
        """
        Conditional CDF F_{V|U}(v|u) = ∂C/∂u underlying the sampling procedure.

        Parameters
        ----------
        u, v : float
            Marginal values.
        theta : float
            Copula parameter.

        Returns
        -------
        float
            Value of ∂C(u,v)/∂u.
        """
        delta = theta - 1.0
        A = 1.0 + delta*(u + v)
        B = np.sqrt(A**2 - 4.0*theta*delta*u*v)
        return 0.5*(1.0 - (A - 2.0*theta*v)/B)

    def sample(self,
               n: int,
               param: np.ndarray = None
              ) -> np.ndarray:
        """
        Draw n samples (u,v) from the Plackett copula by conditional inversion:
          1.  Sample u ~ Uniform(0,1).
          2.  For each u, draw w ~ Uniform(0,1).
          3.  Solve F_{V|U}(v|u) = w for v by bisection.

        Parameters
        ----------
        n : int
            Number of samples.
        param : array_like, optional
            Parameter array [θ].

        Returns
        -------
        np.ndarray, shape (n,2)
            Samples on [0,1]^2.
        """
        if param is None:
            param = self.parameters
        theta = float(param[0])
        eps = 1e-12

        u_s = uniform.rvs(size=n)
        v_s = np.empty(n)
        for i, u in enumerate(u_s):
            w = uniform.rvs()
            try:
                v_s[i] = brentq(
                    lambda vv: self._cond_cdf_v_given_u(u, vv, theta) - w,
                    eps, 1.0 - eps
                )
            except ValueError:
                v_s[i] = uniform.rvs()
        return np.column_stack((u_s, v_s))

    def partial_derivative_C_wrt_u(self,
                                   u: np.ndarray,
                                   v: np.ndarray,
                                   param: np.ndarray = None
                                  ) -> np.ndarray:
        """
        Compute ∂C(u,v)/∂u analytically:

            ∂C/∂u = 0.5 * [1 - (A - 2θv) / sqrt(D)],
        where A = 1 + δ(u+v), D = A² - 4θδuv, δ = θ-1.

        Parameters
        ----------
        u, v : array_like
            Marginal values.
        param : array_like, optional
            Parameter array [θ].

        Returns
        -------
        np.ndarray
            Partial derivative values.
        """
        if param is None:
            param = self.parameters
        theta = float(param[0])
        delta = theta - 1.0

        u = np.asarray(u)
        v = np.asarray(v)
        A = 1.0 + delta*(u + v)
        D = A**2 - 4.0*theta*delta*u*v
        return 0.5*(1.0 - (A - 2.0*theta*v)/np.sqrt(D))

    def partial_derivative_C_wrt_v(self,
                                   u: np.ndarray,
                                   v: np.ndarray,
                                   param: np.ndarray = None
                                  ) -> np.ndarray:
        """
        Compute ∂C(u,v)/∂v analytically:

            ∂C/∂v = 0.5 * [1 - (A - 2θu) / sqrt(D)].
        """
        if param is None:
            param = self.parameters
        theta = float(param[0])
        delta = theta - 1.0

        u = np.asarray(u)
        v = np.asarray(v)
        A = 1.0 + delta*(u + v)
        D = A**2 - 4.0*theta*delta*u*v
        return 0.5*(1.0 - (A - 2.0*theta*u)/np.sqrt(D))

    def conditional_cdf_u_given_v(self,
                                  u: np.ndarray,
                                  v: np.ndarray,
                                  param: np.ndarray = None
                                 ) -> np.ndarray:
        """
        Conditional CDF F_{U|V}(u|v) = ∂C/∂v.
        """
        num = self.partial_derivative_C_wrt_v(u, v, param)
        den = self.partial_derivative_C_wrt_v(1.0, v, param)
        return num / np.maximum(den, 1e-14)

    def conditional_cdf_v_given_u(self,
                                  u: np.ndarray,
                                  v: np.ndarray,
                                  param: np.ndarray = None
                                 ) -> np.ndarray:
        """
        Conditional CDF F_{V|U}(v|u) = ∂C/∂u.
        """
        num = self.partial_derivative_C_wrt_u(u, v, param)
        den = self.partial_derivative_C_wrt_u(u, 1.0, param)
        return num / np.maximum(den, 1e-14)

    def LTDC(self, param: np.ndarray = None) -> float:
        """Lower-tail dependence coefficient (zero)."""
        return 0.0

    def UTDC(self, param: np.ndarray = None) -> float:
        """Upper-tail dependence coefficient (zero)."""
        return 0.0
