import numpy as np
from scipy.stats import uniform
from scipy.integrate import quad
from scipy.optimize import brentq

from Service.Copulas.base import BaseCopula


class JoeCopula(BaseCopula):
    """
    Joe Copula class (Archimedean copula)

    Attributes
    ----------
    type : str
        Identifier for the copula family. Here, 'joe'.
    name : str
        Human-readable name.
    bounds_param : list of tuple
        Bounds for the copula parameter(s). For Joe, theta is in [1, ∞).
    parameters_start : np.ndarray
        Starting guess for the copula parameter(s).
    n_obs : int or None
        Number of observations used in the fit.
    default_optim_method : str
        Recommended optimizer for fitting.

    Methods
    -------
    get_cdf(u, v, param)
        Computes the CDF of the Joe copula at (u, v).
    get_pdf(u, v, param)
        Computes the PDF of the Joe copula at (u, v).
    kendall_tau(param)
        Computes Kendall's tau from the Joe parameter.
    sample(n, param)
        Generates n samples from the Joe copula.
    """

    def __init__(self):
        super().__init__()
        self.type = 'joe'
        self.name = "Joe Copula"
        self.bounds_param = [(1, None)]
        self.parameters = np.array([1.5])
        self.default_optim_method = "Powell"

    def get_cdf(self, u, v, param):
        """
        Computes the cumulative distribution function (CDF) of the Joe copula.

        Formula:
            u_ = (1 - u) ** theta
            v_ = (1 - v) ** theta
            C(u,v) = 1 - (u_ + v_ - u_*v_) ** (1/theta)
        """
        theta = param[0]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        u_ = (1 - u) ** theta
        v_ = (1 - v) ** theta
        return 1 - (u_ + v_ - u_ * v_) ** (1 / theta)

    def get_pdf(self, u, v, param):
        """
        Computes the probability density function (PDF) of the Joe copula.

        Formula:
            u_ = (1 - u) ** theta
            v_ = (1 - v) ** theta
            term1 = (u_ + v_ - u_*v_) ** (-2 + 1/theta)
            term2 = ((1 - u) ** (theta - 1)) * ((1 - v) ** (theta - 1))
            term3 = theta - 1 + u_ + v_ + u_*v_
            PDF = term1 * term2 * term3
        """
        theta = param[0]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        u_ = (1 - u) ** theta
        v_ = (1 - v) ** theta
        term1 = (u_ + v_ - u_ * v_) ** (-2 + 1 / theta)
        term2 = ((1 - u) ** (theta - 1)) * ((1 - v) ** (theta - 1))
        term3 = theta - 1 + u_ + v_ + u_ * v_
        return term1 * term2 * term3

    def kendall_tau(self, param):
        """
        Computes Kendall's tau for the Joe copula.

        For an Archimedean copula with generator φ, a general formula is:
            tau = 1 + 4 * ∫[0 to 1] (φ(t)/φ'(t)) dt
        For the Joe copula, the generator is:
            φ(t) = -log(1 - (1-t)^theta)
        and its derivative:
            φ'(t) = theta * (1-t)^(theta-1) / (1 - (1-t)^theta)

        Thus, the integrand becomes:
            φ(t)/φ'(t) = -log(1 - (1-t)^theta) * (1 - (1-t)^theta) / (theta*(1-t)^(theta-1))

        This method numerically integrates this expression to compute tau.
        """
        theta = param[0]
        # For theta == 1, Joe copula reduces to the independence copula, hence tau = 0.
        if theta == 1:
            return 0.0

        def integrand(t):
            # Protect against t=1 by using clipping.
            t = np.clip(t, 1e-12, 1 - 1e-12)
            numerator = -np.log(1 - (1 - t) ** theta) * (1 - (1 - t) ** theta)
            denominator = theta * (1 - t) ** (theta - 1)
            return numerator / denominator

        integral, _ = quad(integrand, 0, 1, limit=100)
        tau = 1 + 4 * integral
        return tau

    def sample(self, n, param):
        """
        Generates n samples from the Joe copula using conditional inversion.

        The conditional distribution function for V given U = u can be derived by
        differentiating the copula CDF with respect to u. For the Joe copula, one obtains:

            d/du C(u,v) = (1-u)^(theta-1) * (1 - (1-v)^theta) * ( (1-u)^theta + (1-v)^theta - (1-u)^theta*(1-v)^theta )^(1/theta - 1)

        Since d/du C(u,1) = 1, the conditional CDF is given by:

            F_{V|U}(v|u) = d/du C(u,v)

        For each sample, we:
            1. Generate u ~ Uniform(0,1)
            2. Generate w ~ Uniform(0,1)
            3. Solve for v in [0, 1] such that F_{V|U}(v|u) = w using a root-finding algorithm.
        """
        theta = param[0]
        eps = 1e-12
        u_samples = uniform.rvs(size=n)
        v_samples = np.empty(n)

        # Define the conditional CDF derivative function given u.
        def cond_cdf(v, u):
            # Clip to avoid log(0) or division by zero.
            v = np.clip(v, eps, 1 - eps)
            A = (1 - u) ** theta
            B = (1 - v) ** theta
            inner_term = A + B - A * B
            # Derivative of C(u,v) with respect to u.
            deriv = (1 - u) ** (theta - 1) * (1 - B) * inner_term ** (1 / theta - 1)
            return deriv

        for i, u in enumerate(u_samples):
            w = uniform.rvs()
            # We wish to solve f(v) = cond_cdf(v, u) - w = 0 in v in [0, 1].
            try:
                v_solution = brentq(lambda v: cond_cdf(v, u) - w, eps, 1 - eps)
            except ValueError:
                # In case of numerical issues, default to an independent draw.
                v_solution = uniform.rvs()
            v_samples[i] = v_solution

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

        return 2 - 2 ** (1 / theta)


