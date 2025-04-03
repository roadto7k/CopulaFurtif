import numpy as np
from scipy.stats import uniform
from scipy.integrate import quad

from Service.Copulas.base import BaseCopula


class FrankCopula(BaseCopula):
    """
    Frank Copula class (Archimedean copula)

    Attributes
    ----------
    type : str
        Identifier for the copula family (here, 'frank').
    name : str
        Human-readable name.
    bounds_param : list of tuple
        Parameter bounds for optimization (for Frank, the parameter can be any real number except 0).
    parameters : np.ndarray
        Initial guess for the copula parameter(s), e.g. np.array([2.0]).
    n_obs : int or None
        Number of observations used in the fit.
    default_optim_method : str
        Recommended optimization method for fitting.

    Methods
    -------
    get_cdf(u, v, param)
        Computes the CDF of the Frank copula at (u, v).
    get_pdf(u, v, param)
        Computes the PDF of the Frank copula at (u, v).
    kendall_tau(param)
        Computes Kendall's tau from the Frank parameter.
    sample(n, param)
        Generates n samples from the Frank copula.
    """

    def __init__(self):
        super().__init__()
        self.type = 'frank'
        self.name = "Frank Copula"
        self.bounds_param = [(None, None)]
        self.parameters = np.array([2.0])
        self.n_obs = None
        self.default_optim_method = "Powell"

    def get_cdf(self, u, v, param):
        """
        Computes the cumulative distribution function (CDF) of the Frank copula.

        Formula:
            a = (exp(-theta * u) - 1) * (exp(-theta * v) - 1)
            C(u, v) = -1/theta * log(1 + a / (exp(-theta) - 1))
        """
        theta = param[0]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        a = (np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1)
        return -1 / theta * np.log(1 + a / (np.exp(-theta) - 1))

    def get_pdf(self, u, v, param):
        """
        Computes the probability density function (PDF) of the Frank copula.

        Formula:
            term1 = theta * (1 - exp(-theta)) * exp(-theta * (u + v))
            term2 = [1 - exp(-theta) - (1 - exp(-theta * u)) * (1 - exp(-theta * v))]^2
            c(u, v) = term1 / term2
        """
        theta = param[0]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        exp_neg_theta = np.exp(-theta)
        term1 = theta * (1 - exp_neg_theta) * np.exp(-theta * (u + v))
        term2 = (1 - exp_neg_theta - (1 - np.exp(-theta * u)) * (1 - np.exp(-theta * v))) ** 2
        return term1 / term2

    def kendall_tau(self, param):
        """
        Computes Kendall's tau for the Frank copula.

        Formula:
            tau = 1 - (4/theta) * (1 - D1(theta))
        where D1(theta) is the Debye function of order 1, computed as:
            D1(theta) = (1/theta) * int_0^theta [t / (exp(t) - 1)] dt
        """
        theta = param[0]
        if np.abs(theta) < 1e-8:
            return 0.0  # Independence

        def debye1(t):
            return t / (np.exp(t) - 1)

        debye_val, _ = quad(debye1, 0, theta)
        D1 = debye_val / theta
        return 1 - (4 / theta) * (1 - D1)

    def sample(self, n, param):
        """
        Generates n samples from the Frank copula using the inversion method
        on the conditional distribution.

        For theta = 0, the copula is independent.

        Method:
            1. Generate u ~ U(0,1)
            2. Generate w ~ U(0,1)
            3. Compute v via the inversion of the conditional CDF:
               v = -1/theta * log(1 + (log(1 - w * (1 - exp(-theta))) / (exp(-theta * u) - 1)))
        """
        theta = param[0]
        # Use a near-zero threshold for numerical stability
        if np.abs(theta) < 1e-8:
            u = uniform.rvs(size=n)
            v = uniform.rvs(size=n)
            return np.column_stack((u, v))

        u = uniform.rvs(size=n)
        w = uniform.rvs(size=n)
        exp_neg_theta = np.exp(-theta)
        exp_neg_theta_u = np.exp(-theta * u)

        numerator = np.log(1 - w * (1 - exp_neg_theta))
        denominator = exp_neg_theta_u - 1

        # Adjust denominator to avoid division by a value too close to zero
        eps = 1e-12
        denominator = np.where(np.abs(denominator) < eps, eps, denominator)

        v = -1 / theta * np.log(1 + numerator / denominator)
        return np.column_stack((u, v))

    def LTDC(self, param):
        """
        Computes the lower tail dependence coefficient for the Frank copula.

        Frank copula has no lower tail dependence:
            LTDC = 0
        """
        return 0.0

    def UTDC(self, param):
        """
        Computes the upper tail dependence coefficient for the Frank copula.

        Frank copula has no upper tail dependence:
            UTDC = 0
        """
        return 0.0
