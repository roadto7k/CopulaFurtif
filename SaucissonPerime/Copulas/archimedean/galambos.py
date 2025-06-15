import numpy as np
from scipy.stats import uniform
from scipy.optimize import brentq
from SaucissonPerime.Copulas.base import BaseCopula


class GalambosCopula(BaseCopula):
    """
    Galambos Copula class (Extreme-Value Copula).

    Parameters
    ----------
    theta : float
        Tail dependence parameter (θ > 0).

    Attributes
    ----------
    type : str
        Identifier for the copula family ('galambos').
    name : str
        Human-readable name.
    bounds_param : list of tuple
        Parameter bounds: [(1e-6, None)] for θ > 0.
    _parameters : np.ndarray
        Internal storage for copula parameter [θ].
    default_optim_method : str
        Default optimizer for parameter fitting.
    """

    def __init__(self):
        super().__init__()
        self.type = "galambos"
        self.name = "Galambos Copula"
        # θ > 0
        self.bounds_param = [(1e-6, None)]
        self._parameters = np.array([0.5], dtype=float)
        self.default_optim_method = "SLSQP"

    @property
    def parameters(self) -> np.ndarray:
        """
        Copula parameters as [theta].
        """
        return self._parameters

    @parameters.setter
    def parameters(self, param: np.ndarray):
        """
        Set copula parameters, enforcing bounds.
        """
        param = np.asarray(param, dtype=float)
        theta = param[0]
        low, high = self.bounds_param[0]
        if theta <= low or (high is not None and theta > high):
            raise ValueError(f"Parameter theta must be > {low}")
        self._parameters = param

    def get_cdf(self,
                u: np.ndarray,
                v: np.ndarray,
                param: np.ndarray = None) -> np.ndarray:
        """
        Compute the Galambos copula CDF C(u,v).

        C(u,v) = u * v * exp( [(-log u)^(-θ) + (-log v)^(-θ)]^(-1/θ) )

        Parameters
        ----------
        u : float or np.ndarray
            Values in (0,1) for U.
        v : float or np.ndarray
            Values in (0,1) for V.
        param : np.ndarray, optional
            Copula parameter [θ]. If None, uses self.parameters.

        Returns
        -------
        np.ndarray
            Copula CDF values.
        """
        if param is None:
            theta = self.parameters[0]
        else:
            theta = np.asarray(param, dtype=float)[0]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        X = (-np.log(u)) ** (-theta)
        Y = (-np.log(v)) ** (-theta)
        G = (X + Y) ** (-1.0 / theta)
        return u * v * np.exp(G)

    def get_pdf(self,
                u: np.ndarray,
                v: np.ndarray,
                param: np.ndarray = None) -> np.ndarray:
        """
        Compute the Galambos copula PDF c(u,v).

        Uses analytic expression:
          c = term1*term2 + term3*term4
        where term1..term4 are defined in the doc.

        Parameters
        ----------
        u : float or np.ndarray
            Values in (0,1) for U.
        v : float or np.ndarray
            Values in (0,1) for V.
        param : np.ndarray, optional
            Copula parameter [θ]. If None, uses self.parameters.

        Returns
        -------
        np.ndarray
            Copula PDF values.
        """
        if param is None:
            theta = self.parameters[0]
        else:
            theta = np.asarray(param, dtype=float)[0]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        x = -np.log(u)
        y = -np.log(v)
        c_val = self.get_cdf(u, v, [theta])
        term1 = c_val / (u * v)
        base = x ** (-theta) + y ** (-theta)
        term2 = 1 - base ** (-1 - 1 / theta) * (x ** (-theta - 1) + y ** (-theta - 1))
        term3 = base ** (-2 - 1 / theta) * (x * y) ** (-theta - 1)
        term4 = 1 + theta + base ** (-1 / theta)
        return term1 * term2 + term3 * term4

    def kendall_tau(self,
                    param: np.ndarray = None,
                    tol: float = 1e-10,
                    max_iter: int = 1000) -> float:
        """
        Estimate Kendall's tau via power series expansion.

        τ = 1 - (1/θ) * \sum_{j=1}^∞ (-1)^{j-1}/(j+1/θ) * C(2j,j)/4^j
        """
        if param is None:
            theta = self.parameters[0]
        else:
            theta = np.asarray(param, dtype=float)[0]
        if theta <= 0:
            raise ValueError("theta must be > 0")
        summation = 0.0
        for j in range(1, max_iter + 1):
            sign = 1.0 if (j - 1) % 2 == 0 else -1.0
            denom = j + 1.0 / theta
            if abs(denom) < tol:
                raise ValueError(f"Denominator near zero at j={j}")
            bin_coeff = math.comb(2*j, j)
            log_term = math.log(bin_coeff) - (j * math.log(4) + math.log(denom))
            term = sign * math.exp(log_term)
            summation += term
            if abs(term) < tol * abs(summation):
                break
        return 1.0 - (1.0 / theta) * summation

    def sample(self,
               n: int,
               param: np.ndarray = None) -> np.ndarray:
        """
        Generate samples via conditional inversion.

        For each:
          u ~ U(0,1), w ~ U(0,1), target=u*w
          solve C(u,v)=target for v by bisection.
        """
        if param is None:
            theta = self.parameters[0]
        else:
            theta = np.asarray(param, dtype=float)[0]
        eps = 1e-12
        u_samp = uniform.rvs(size=n)
        v_samp = np.empty(n)
        for i, u in enumerate(u_samp):
            target = u * uniform.rvs()
            f = lambda v: self.get_cdf(u, v, [theta]) - target
            try:
                v_samp[i] = brentq(f, eps, 1 - eps)
            except ValueError:
                v_samp[i] = uniform.rvs()
        return np.column_stack((u_samp, v_samp))

    def LTDC(self,
             param: np.ndarray = None) -> float:
        """
        Lower-tail dependence (0 for Galambos).
        """
        return 0.0

    def UTDC(self,
             param: np.ndarray = None) -> float:
        """
        Upper-tail dependence: 2 - 2^(1/θ).
        """
        if param is None:
            theta = self.parameters[0]
        else:
            theta = np.asarray(param, dtype=float)[0]
        return 2 - 2 ** (1.0 / theta)

    def partial_derivative_C_wrt_v(self,
                                   u: np.ndarray,
                                   v: np.ndarray,
                                   param: np.ndarray = None) -> np.ndarray:
        """
        Analytical ∂C/∂v = u*e^G * [1 - A^{-(1/θ+1)}*(-log v)^{-(θ+1)}].
        """
        if param is None:
            theta = self.parameters[0]
        else:
            theta = np.asarray(param, dtype=float)[0]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        X = (-np.log(u)) ** (-theta)
        Y = (-np.log(v)) ** (-theta)
        A = X + Y
        G = A ** (-1.0 / theta)
        return (u * np.exp(G)
                * (1.0 - A ** (-1.0 / theta - 1.0) * (-np.log(v)) ** (-theta - 1.0)))

    def partial_derivative_C_wrt_u(self,
                                   u: np.ndarray,
                                   v: np.ndarray,
                                   param: np.ndarray = None) -> np.ndarray:
        """
        Analytical ∂C/∂u by symmetry.
        """
        if param is None:
            theta = self.parameters[0]
        else:
            theta = np.asarray(param, dtype=float)[0]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        X = (-np.log(u)) ** (-theta)
        Y = (-np.log(v)) ** (-theta)
        A = X + Y
        G = A ** (-1.0 / theta)
        return (v * np.exp(G)
                * (1.0 - A ** (-1.0 / theta - 1.0) * (-np.log(u)) ** (-theta - 1.0)))

    def conditional_cdf_u_given_v(self,
                                  u: np.ndarray,
                                  v: np.ndarray,
                                  param: np.ndarray = None) -> np.ndarray:
        """
        P(U ≤ u | V=v) = ∂C/∂v ÷ ∂C/∂v at u=1.
        """
        num = self.partial_derivative_C_wrt_v(u, v, param)
        den = self.partial_derivative_C_wrt_v(1.0, v, param)
        return num / np.maximum(den, 1e-14)

    def conditional_cdf_v_given_u(self,
                                  u: np.ndarray,
                                  v: np.ndarray,
                                  param: np.ndarray = None) -> np.ndarray:
        """
        P(V ≤ v | U=u) = ∂C/∂u ÷ ∂C/∂u at v=1.
        """
        num = self.partial_derivative_C_wrt_u(u, v, param)
        den = self.partial_derivative_C_wrt_u(u, 1.0, param)
        return num / np.maximum(den, 1e-14)
