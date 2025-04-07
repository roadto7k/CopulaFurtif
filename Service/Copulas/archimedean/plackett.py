import numpy as np
from scipy.stats import uniform
from scipy.optimize import brentq
from scipy.special import comb

from Service.Copulas.base import BaseCopula


class PlackettCopula(BaseCopula):
    """
    Plackett Copula class

    Attributes
    ----------
    type : str
        Identifier for the copula family. Here, 'plackett'.
    name : str
        Human-readable name.
    bounds_param : list of tuple
        Bounds for the copula parameter. For Plackett, theta is in (0, ∞).
    parameters_start : np.ndarray
        Starting guess for the copula parameter (theta).
    n_obs : int or None
        Number of observations used in the fit.
    default_optim_method : str
        Recommended optimizer for fitting.

    Methods
    -------
    get_cdf(u, v, param)
        Computes the CDF of the Plackett copula.
    get_pdf(u, v, param)
        Computes the PDF of the Plackett copula.
    kendall_tau(param)
        Computes Kendall's tau from the Plackett parameter.
    sample(n, param)
        Generates n samples from the Plackett copula.
    """

    def __init__(self):
        super().__init__()
        self.type = 'plackett'
        self.name = "Plackett Copula"
        self.bounds_param = [(1e-6, None)]
        self.parameters = np.array([0.5])
        self.default_optim_method = "Powell"

    def get_cdf(self, u, v, param):
        """
        Computes the CDF of the Plackett copula using a reformulated expression.

        Parameters
        ----------
        u : array_like
            First uniform marginal.
        v : array_like
            Second uniform marginal.
        param : array_like
            Copula parameter (theta,) as a one-element array.

        Returns
        -------
        np.ndarray
            Copula CDF values for the input pairs.
        """
        theta = param[0]
        delta = theta - 1.0

        u = np.asarray(u)
        v = np.asarray(v)

        pre_factor = 0.5 / delta
        sum_uv = u + v
        linear = 1.0 + delta * sum_uv
        square = linear ** 2
        correction = 4.0 * theta * delta * u * v
        root_term = np.sqrt(square - correction)

        return pre_factor * (linear - root_term)

    def get_pdf(self, u, v, param):
        """
        Computes the PDF of the Plackett copula with restructured expression.

        Parameters
        ----------
        u : array_like
            First array of uniform marginals.
        v : array_like
            Second array of uniform marginals.
        param : array_like
            Copula parameter array, where param[0] = theta.

        Returns
        -------
        np.ndarray
            The copula density evaluated at each (u, v).
        """
        theta = param[0]
        delta = theta - 1.0

        uv = np.asarray(u)
        vv = np.asarray(v)

        s = uv + vv
        p = uv * vv

        top = theta * (1.0 + delta * (s - 2.0 * p))
        inner = (1.0 + delta * s) ** 2 - 4.0 * theta * delta * p
        bottom = inner ** 1.5

        return top / bottom

    def kendall_tau(self, param):
        """
        Computes Kendall's tau for the Plackett copula.

        The relationship is given by:
            τ = 1 - [2(θ+1)]/(3θ) + [2(θ-1)² * ln(θ)]/(3θ²)

        Parameters
        ----------
        param : array_like
            Copula parameter (theta), must be > 0.

        Returns
        -------
        float
            Kendall's tau.
        """
        theta = float(param[0])
        if theta <= 0:
            raise ValueError("Theta must be positive for the Plackett copula.")
        if abs(theta - 1.0) < 1e-8:
            return 0.0  # Limit case: independence

        return 1 - (2 * (theta + 1)) / (3 * theta) + (2 * (theta - 1) ** 2 * np.log(theta)) / (3 * theta ** 2)

    def sample(self, n, param):
        """
        Generates n samples from the Plackett copula using a conditional inversion approach.

        Sampling algorithm:
            1. Generate u ~ Uniform(0,1).
            2. For each u, generate w ~ Uniform(0,1) and solve for v in [0,1] such that:
                   ∂C(u,v)/∂u = w.
               The analytic derivative of C(u,v) with respect to u is:
                   dC/du = [1 - (A - 2θv)/B] / 2,
               where A = 1 + (θ-1)(u+v) and B = √(A² - 4θ(θ-1)uv).
            3. Use a root-finding method (Brent's method) to find v.

        Parameters
        ----------
        n : int
            Number of samples to generate.
        param : array_like
            Copula parameter (theta).

        Returns
        -------
        np.ndarray
            An n x 2 array of samples from the Plackett copula.
        """
        theta = param[0]
        eps = 1e-12
        u_samples = uniform.rvs(size=n)
        v_samples = np.empty(n)

        def cond_cdf(u, v):
            A = 1 + (theta - 1) * (u + v)
            B = np.sqrt(A ** 2 - 4 * theta * (theta - 1) * u * v)
            return (1 - (A - 2 * theta * v) / B) / 2

        for i, u in enumerate(u_samples):
            w = uniform.rvs()
            # Solve f(v) = cond_cdf(u, v) - w = 0 for v in [0,1]
            try:
                v_sol = brentq(lambda v: cond_cdf(u, v) - w, eps, 1 - eps)
            except ValueError:
                v_sol = uniform.rvs()
            v_samples[i] = v_sol

        return np.column_stack((u_samples, v_samples))

    def LTDC(self, param):
        """
        Computes the lower tail dependence coefficient for the plackett copula.

        Formula:
            LTDC = 0
        """
        return 0

    def UTDC(self, param):
        """
        Computes the upper tail dependence coefficient for the plackett copula.

        Clayton copula has no upper tail dependence:
            UTDC = 0
        """
        return 0.0
