import numpy as np
from scipy.stats import uniform
from scipy.optimize import brentq
from scipy.special import comb

from Service.Copulas.base import BaseCopula


class GalambosCopula(BaseCopula):
    """
    Galambos Copula class (Extreme Value Copula)

    Attributes
    ----------
    type : str
        Identifier for the copula family. Here, 'galambos'.
    name : str
        Human-readable name.
    bounds_param : list of tuple
        Bounds for the copula parameter(s). For Galambos, theta > 0.
    parameters_start : np.ndarray
        Starting guess for the copula parameter(s).
    n_obs : int or None
        Number of observations used in the fit.
    default_optim_method : str
        Recommended optimizer for fitting.

    Methods
    -------
    get_cdf(u, v, param)
        Computes the CDF of the Galambos copula at (u, v).
    get_pdf(u, v, param)
        Computes the PDF of the Galambos copula at (u, v).
    kendall_tau(param)
        Computes Kendall's tau for the Galambos copula.
    sample(n, param)
        Generates n samples from the Galambos copula.
    """

    def __init__(self):
        super().__init__()
        self.type = 'galambos'
        self.name = "Galambos Copula"
        self.bounds_param = [(1e-6, None)]
        self.parameters_start = np.array([0.5])
        self.n_obs = None
        self.default_optim_method = "Powell"

    def get_cdf(self, u, v, param):
        """
        Computes the cumulative distribution function (CDF) of the Galambos copula.

        Formula:
            C(u,v) = u * v * exp{ [ (-log u)^(-theta) + (-log v)^(-theta) ]^(-1/theta) }

        Parameters
        ----------
        u : array_like
            First set of uniform marginals.
        v : array_like
            Second set of uniform marginals.
        param : array_like
            Copula parameter (theta).

        Returns
        -------
        np.ndarray
            The CDF of the Galambos copula evaluated at (u, v).
        """
        theta = param[0]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        inner = ((-np.log(u)) ** (-theta) + (-np.log(v)) ** (-theta)) ** (-1 / theta)
        return u * v * np.exp(inner)

    def get_pdf(self, u, v, param):
        """
        Computes the probability density function (PDF) of the Galambos copula.

        Formula:
            Let x = -log(u) and y = -log(v), then
            term1 = C(u,v) / (u * v)
            term2 = 1 - [ (x^(-theta) + y^(-theta))^(-1 - 1/theta) * ( x^(-theta-1) + y^(-theta-1) ) ]
            term3 = (x^(-theta) + y^(-theta))^(-2 - 1/theta) * (x * y)^(-theta-1)
            term4 = 1 + theta + (x^(-theta) + y^(-theta))^(-1/theta)
            PDF = term1 * term2 + term3 * term4

        Parameters
        ----------
        u : array_like
            First set of uniform marginals.
        v : array_like
            Second set of uniform marginals.
        param : array_like
            Copula parameter (theta).

        Returns
        -------
        np.ndarray
            The PDF of the Galambos copula evaluated at (u, v).
        """
        theta = param[0]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        x = -np.log(u)
        y = -np.log(v)

        c_val = self.get_cdf(u, v, param)
        term1 = c_val / (u * v)
        base = x ** (-theta) + y ** (-theta)
        term2 = 1 - (base ** (-1 - 1 / theta)) * (x ** (-theta - 1) + y ** (-theta - 1))
        term3 = (base ** (-2 - 1 / theta)) * ((x * y) ** (-theta - 1))
        term4 = 1 + theta + (base ** (-1 / theta))

        return term1 * term2 + term3 * term4

    def kendall_tau(self, param):
        """
        Computes Kendall's tau for the Galambos copula.

        One common series representation for Kendall's tau is:
            tau = 1 - (1/theta) * sum_{j=1}^∞ [ (-1)^(j-1) / (j + 1/theta) * C(2j, j)/4^j ]
        This series is computed until convergence.

        Parameters
        ----------
        param : array_like
            Copula parameter (theta).

        Returns
        -------
        float
            Kendall's tau for the Galambos copula.
        """
        theta = param[0]
        summation = 0.0
        j = 1
        while True:
            term = ((-1) ** (j - 1) / (j + 1 / theta)) * comb(2 * j, j) / (4 ** j)
            summation += term
            if abs(term) < 1e-8:
                break
            j += 1
            if j > 1000:
                break
        tau = 1 - (1 / theta) * summation
        return tau

    def sample(self, n, param):
        """
        Generates n samples from the Galambos copula using a conditional inversion method.

        The method works as follows:
            1. Generate u ~ Uniform(0, 1)
            2. For each u, generate a second independent uniform w.
            3. Solve for v in [0, 1] such that C(u, v) = u * w.
               This inversion is performed numerically using a root finder.

        Parameters
        ----------
        n : int
            Number of samples to generate.
        param : array_like
            Copula parameter (theta).

        Returns
        -------
        np.ndarray
            An n x 2 array of samples from the Galambos copula.
        """
        eps = 1e-12
        u_samples = uniform.rvs(size=n)
        v_samples = np.empty(n)

        for i, u in enumerate(u_samples):
            # For each u, define target = u * w, where w ~ Uniform(0,1)
            target = u * uniform.rvs()
            # Define the function to find the root: f(v) = C(u,v) - target
            func = lambda v: self.get_cdf(u, v, param) - target
            try:
                v_sol = brentq(func, eps, 1 - eps)
            except ValueError:
                # In case the solver fails, use an independent draw.
                v_sol = uniform.rvs()
            v_samples[i] = v_sol

        return np.column_stack((u_samples, v_samples))

    def LTDC(self, param):
        """
        Computes the lower tail dependence coefficient for the Joe copula.

        For the Joe copula, there is NO lower tail dependence:
            LTDC = 0
        """
        return 0.0

    def UTDC(self, param):
        """
        Computes the upper tail dependence coefficient for the Joe copula.

        Formula:
            UTDC = 2 - 2 ** (1 / theta)

        This increases with theta and is in [0, 1).
        """
        theta = param[0]

        return 2 ** (-1 / theta)
