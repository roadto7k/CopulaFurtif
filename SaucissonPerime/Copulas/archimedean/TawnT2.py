import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar

from SaucissonPerime.Copulas.base import BaseCopula


class TawnT2Copula(BaseCopula):
    """
    Tawn Type-2 (asymmetric mixed) extreme-value copula.

    Parameters
    ----------
    theta : float
        Dependence strength parameter (>= 1).
    beta : float
        Asymmetry parameter in [0, 1].

    Pickands dependence function:
        A(t) = (1 - beta)*(1 - t) + [t**theta + (beta*(1 - t))**theta]**(1/theta)

    Copula CDF:
        C(u, v) = exp(- (x + y) * A(x/(x+y))),
        where x = -log(u), y = -log(v).
    """

    def __init__(self):
        super().__init__()
        self.type = "tawn2"
        self.name = "Tawn Type-2 Copula"
        self.bounds_param = [(1.0, None), (0.0, 1.0)]
        self._parameters = np.array([2.0, 0.5])  # [theta, beta]
        self.default_optim_method = "Powell"

    @property
    def parameters(self) -> np.ndarray:
        return self._parameters

    @parameters.setter
    def parameters(self, param: np.ndarray):
        self._parameters = np.asarray(param)

    def _A(self, t: float, param: np.ndarray = None) -> float:
        """
        Pickands dependence function A(t).

        Parameters
        ----------
        t : float
            Value in [0,1].
        param : ndarray, optional
            Copula parameters [theta, beta].

        Returns
        -------
        A_t : float
            Value of A(t).
        """
        if param is None:
            param = self.parameters
        theta, beta = param
        return (1 - beta) * (1 - t) + (t**theta + (beta * (1 - t))**theta)**(1.0 / theta)

    def _A_prime(self, t: float, param: np.ndarray = None) -> float:
        """
        First derivative A'(t).

        Parameters
        ----------
        t : float
            Value in [0,1].
        param : ndarray, optional
            Copula parameters [theta, beta].

        Returns
        -------
        A1_t : float
            Derivative A'(t).
        """
        if param is None:
            param = self.parameters
        theta, beta = param
        h = t**theta + (beta * (1 - t))**theta
        hp = theta * t**(theta - 1) - theta * beta * (beta * (1 - t))**(theta - 1)
        return -(1 - beta) + (1.0 / theta) * h**(1.0 / theta - 1) * hp

    def _A_double(self, t: float, param: np.ndarray = None) -> float:
        """
        Second derivative A''(t).

        Parameters
        ----------
        t : float
            Value in [0,1].
        param : ndarray, optional
            Copula parameters [theta, beta].

        Returns
        -------
        A2_t : float
            Derivative A''(t).
        """
        if param is None:
            param = self.parameters
        theta, beta = param
        h = t**theta + (beta * (1 - t))**theta
        hp = theta * t**(theta - 1) - theta * beta * (beta * (1 - t))**(theta - 1)
        hpp = theta * (theta - 1) * (t**(theta - 2) + beta**theta * (1 - t)**(theta - 2))
        term1 = (1.0 / theta) * (1.0 / theta - 1) * h**(1.0 / theta - 2) * hp**2
        term2 = (1.0 / theta) * h**(1.0 / theta - 1) * hpp
        return term1 + term2

    def get_cdf(self, u: np.ndarray, v: np.ndarray, param: np.ndarray = None) -> np.ndarray:
        """
        Copula CDF: C(u,v) = exp(-l(x,y)), where l = (x+y)*A(x/(x+y)).

        Parameters
        ----------
        u : ndarray
            First pseudo-observations in (0,1).
        v : ndarray
            Second pseudo-observations in (0,1).
        param : ndarray, optional
            Copula parameters [theta, beta].

        Returns
        -------
        C : ndarray
            Copula CDF values.
        """
        if param is None:
            param = self.parameters
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        x = -np.log(u)
        y = -np.log(v)
        s = x + y
        t = x / s
        return np.exp(-s * self._A(t, param))

    def get_pdf(self, u: np.ndarray, v: np.ndarray, param: np.ndarray = None) -> np.ndarray:
        """
        Copula PDF: c(u,v) = C(u,v) * (l_x * l_y - l_xy) / (u*v).

        Parameters
        ----------
        u : ndarray
            First pseudo-observations in (0,1).
        v : ndarray
            Second pseudo-observations in (0,1).
        param : ndarray, optional
            Copula parameters [theta, beta].

        Returns
        -------
        c : ndarray
            Copula PDF values.
        """
        if param is None:
            param = self.parameters
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        x = -np.log(u)
        y = -np.log(v)
        s = x + y
        t = x / s
        A = self._A(t, param)
        Ap = self._A_prime(t, param)
        App = self._A_double(t, param)
        Lx = A + (y / s) * Ap
        Ly = A - (x / s) * Ap
        Lxy = - (x * y / s**3) * App
        C_val = np.exp(-s * A)
        return C_val * (Lx * Ly - Lxy) / (u * v)

    def partial_derivative_C_wrt_u(self, u: np.ndarray, v: np.ndarray, param: np.ndarray = None) -> np.ndarray:
        """
        Partial derivative ∂C/∂u for conditional sampling.

        Parameters
        ----------
        u : ndarray
            First pseudo-observations.
        v : ndarray
            Second pseudo-observations.
        param : ndarray, optional
            Copula parameters [theta, beta].

        Returns
        -------
        dCdu : ndarray
            ∂C/∂u values.
        """
        if param is None:
            param = self.parameters
        C_val = self.get_cdf(u, v, param)
        x = -np.log(u)
        y = -np.log(v)
        s = x + y
        t = x / s
        A = self._A(t, param)
        Ap = self._A_prime(t, param)
        return C_val * (A / u + (y / (u * s)) * Ap)

    def partial_derivative_C_wrt_v(self, u: np.ndarray, v: np.ndarray, param: np.ndarray = None) -> np.ndarray:
        """
        Partial derivative ∂C/∂v for conditional sampling.

        Parameters
        ----------
        u : ndarray
            First pseudo-observations.
        v : ndarray
            Second pseudo-observations.
        param : ndarray, optional
            Copula parameters [theta, beta].

        Returns
        -------
        dCdv : ndarray
            ∂C/∂v values.
        """
        if param is None:
            param = self.parameters
        C_val = self.get_cdf(u, v, param)
        x = -np.log(u)
        y = -np.log(v)
        s = x + y
        t = x / s
        A = self._A(t, param)
        Ap = self._A_prime(t, param)
        return C_val * (A / v - (x / (v * s)) * Ap)

    def conditional_cdf_v_given_u(self, u: np.ndarray, v: np.ndarray, param: np.ndarray = None) -> np.ndarray:
        """
        Conditional CDF P(V <= v | U = u).

        Parameters
        ----------
        u : ndarray
            First pseudo-observations.
        v : ndarray
            Second pseudo-observations.
        param : ndarray, optional
            Copula parameters [theta, beta].

        Returns
        -------
        P : ndarray
            P(V <= v | U = u).
        """
        if param is None:
            param = self.parameters
        return self.partial_derivative_C_wrt_u(u, v, param)

    def conditional_cdf_u_given_v(self, u: np.ndarray, v: np.ndarray, param: np.ndarray = None) -> np.ndarray:
        """
        Conditional CDF P(U <= u | V = v).

        Parameters
        ----------
        u : ndarray
            First pseudo-observations.
        v : ndarray
            Second pseudo-observations.
        param : ndarray, optional
            Copula parameters [theta, beta].

        Returns
        -------
        P : ndarray
            P(U <= u | V = v).
        """
        if param is None:
            param = self.parameters
        return self.partial_derivative_C_wrt_v(u, v, param)

    def sample(self, n: int, param: np.ndarray = None) -> np.ndarray:
        """
        Generate n samples via conditional inversion:
          1. u ~ Uniform(0,1)
          2. p ~ Uniform(0,1)
          3. solve P(V<=v|U=u)=p via bisection.

        Parameters
        ----------
        n : int
            Number of samples.
        param : ndarray, optional
            Copula parameters [theta, beta].

        Returns
        -------
        samples : ndarray of shape (n,2)
            Array of (u,v) samples.
        """
        if param is None:
            param = self.parameters
        eps = 1e-6
        u = np.random.rand(n)
        v = np.empty(n)
        for i in range(n):
            p = np.random.rand()
            sol = root_scalar(
                lambda vv: self.conditional_cdf_v_given_u(u[i], vv, param) - p,
                bracket=[eps, 1 - eps], method="bisect", xtol=1e-6
            )
            v[i] = sol.root
        return np.column_stack((u, v))

    def kendall_tau(self, param: np.ndarray = None) -> float:
        """
        Compute Kendall's tau: τ = 1 - 4 ∫₀¹ A(t) dt.

        Parameters
        ----------
        param : ndarray, optional
            Copula parameters [theta, beta].

        Returns
        -------
        tau : float
            Kendall's tau.
        """
        if param is None:
            param = self.parameters
        integral, _ = quad(lambda t: self._A(t, param), 0.0, 1.0)
        return 1.0 - 4.0 * integral

    def LTDC(self, param: np.ndarray = None) -> float:
        """
        Lower-tail dependence coefficient λ_L = 0.

        Parameters
        ----------
        param : ndarray, optional
            Copula parameters, unused.

        Returns
        -------
        l : float
            Lower-tail dependence.
        """
        return 0.0

    def UTDC(self, param: np.ndarray = None) -> float:
        """
        Upper-tail dependence coefficient λ_U = 2 - 2^(1/theta).

        Parameters
        ----------
        param : ndarray, optional
            Copula parameters [theta, beta].

        Returns
        -------
        u : float
            Upper-tail dependence.
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        return 2.0 - 2.0**(1.0 / theta)
