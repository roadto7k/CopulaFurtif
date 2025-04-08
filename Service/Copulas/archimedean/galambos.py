import math

import numpy as np
from scipy.stats import uniform
from scipy.optimize import brentq
from scipy.special import comb
from math import comb, log, exp

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
        self.parameters = np.array([0.5])
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

    def kendall_tau(self, param, tol=1e-10, max_iter=1000):
        """
        Compute Kendall's tau for the Galambos copula using a series expansion
        in log-space to avoid overflow:

            tau = 1 - (1/theta) * sum_{j=1}^∞ [ (-1)^(j-1) / (j + 1/theta) * C(2j, j) / 4^j ],

        Parameters
        ----------
        param : array_like
            Copula parameters, with param[0] = theta. Must satisfy theta > 0.
        tol : float, optional
            Relative tolerance for convergence of the series.
        max_iter : int, optional
            Maximum number of series terms to sum.

        Returns
        -------
        float
            The approximate Kendall's tau for the Galambos copula.

        Raises
        ------
        ValueError
            If theta <= 0 or if an invalid denominator is encountered.
        """
        theta = param[0]
        if theta <= 0:
            raise ValueError("Galambos copula requires theta > 0.")

        summation = 0.0
        for j in range(1, max_iter + 1):
            # Compute the sign (-1)^(j-1)
            sign = 1.0 if ((j - 1) % 2 == 0) else -1.0

            denom = j + (1.0 / theta)
            if abs(denom) < 1e-15:
                raise ValueError(f"Encountered nearly zero denominator at j={j}.")

            # Use math.comb without keyword arguments; returns an integer.
            bin_coeff = math.comb(2 * j, j)
            log_bin_coeff = math.log(bin_coeff)

            # Compute log(4^j * denom) = j * log(4) + log(denom)
            log_divisor = j * math.log(4.0) + math.log(denom)

            # The term in log-space
            log_term_abs = log_bin_coeff - log_divisor

            # Exponentiate to get the absolute value of the term, then apply the sign.
            term = sign * exp(log_term_abs)

            summation += term

            # Check for convergence: stop when the term is negligible relative to the sum.
            if abs(term) < tol * abs(summation):
                break
        else:
            print(f"[WARNING] Galambos tau series did not converge within {max_iter} terms. "
                  "Result may be inaccurate.")

        tau = 1.0 - (1.0 / theta) * summation
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

    def conditional_cdf_u_given_v(self, u, v, param):
        """
        Analytically computes the conditional CDF P(U ≤ u | V = v)
        for the Galambos copula.

        The Galambos copula is given by:
            C(u,v) = exp{ -[(-ln u)^(-θ) + (-ln v)^(-θ)]^(-1/θ) }.

        Let A = (-ln u)^(-θ) + (-ln v)^(-θ), then the derivative with respect
        to v is:
            ∂C(u,v)/∂v = [C(u,v)/v] * A^(-1/θ - 1) * (-ln v)^(-θ-1).

        Since C(1,v)=v and ∂C(1,v)/∂v=1, this derivative represents the
        conditional CDF of U given V=v.

        Parameters
        ----------
        u : float or array-like
            Value(s) in (0,1) for U.
        v : float or array-like
            Value(s) in (0,1) for V (conditioning variable).
        param : list or array-like, optional
            Copula parameter(s) as [θ]. If None, self.parameters is used.

        Returns
        -------
        float or np.ndarray
            The conditional CDF P(U ≤ u | V = v).
        """

        theta = param[0]

        # Ensure u and v are numpy arrays for vectorized operations.
        u = np.asarray(u)
        v = np.asarray(v)

        # Compute A = (-ln u)^(-θ) + (-ln v)^(-θ)
        A = (-np.log(u)) ** (-theta) + (-np.log(v)) ** (-theta)
        # Compute the copula CDF: C(u,v) = exp{-A^(-1/θ)}
        Cuv = np.exp(-A ** (-1 / theta))

        # Compute the derivative with respect to v
        derivative = (Cuv / v) * (A ** (-1 / theta - 1)) * ((-np.log(v)) ** (-theta - 1))
        return derivative

    def conditional_cdf_v_given_u(self, v, u, param):
        """
        Analytically computes the conditional CDF P(V ≤ v | U = u)
        for the Galambos copula.

        By symmetry, with A = (-ln u)^(-θ) + (-ln v)^(-θ), we have:
            ∂C(u,v)/∂u = [C(u,v)/u] * A^(-1/θ - 1) * (-ln u)^(-θ-1),
        which defines the conditional CDF of V given U=u.

        Parameters
        ----------
        v : float or array-like
            Value(s) in (0,1) for V.
        u : float or array-like
            Value(s) in (0,1) for U (conditioning variable).
        param : list or array-like, optional
            Copula parameter(s) as [θ]. If None, self.parameters is used.

        Returns
        -------
        float or np.ndarray
            The conditional CDF P(V ≤ v | U = u).
        """

        theta = param[0]

        u = np.asarray(u)
        v = np.asarray(v)

        # Compute A = (-ln u)^(-θ) + (-ln v)^(-θ)
        A = (-np.log(u)) ** (-theta) + (-np.log(v)) ** (-theta)
        # Compute the copula CDF: C(u,v) = exp{-A^(-1/θ)}
        Cuv = np.exp(-A ** (-1 / theta))

        # Compute the derivative with respect to u
        derivative = (Cuv / u) * (A ** (-1 / theta - 1)) * ((-np.log(u)) ** (-theta - 1))
        return derivative
