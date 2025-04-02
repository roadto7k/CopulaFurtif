import numpy as np
from scipy.stats import norm
from scipy.special import expm1, log1p

from Service.Copulas.base import BaseCopula


class GumbelCopula(BaseCopula):
    """
    Gumbel Copula (Archimedean)

    A copula used to model asymmetric upper tail dependence. Belongs to the Archimedean family.

    Attributes
    ----------
    type : str
        Identifier for the copula type.
    name : str
        Human-readable name.
    bounds_param : list of tuple
        Bounds for the copula parameter(s). For Gumbel: theta in [1, ∞)
    parameters : np.ndarray
        Initial guess for the copula parameter(s).
    n_obs : int or None
        Number of observations, populated during fitting.
    default_optim_method : str
        Default optimization algorithm.

    Methods
    -------
    get_cdf(u, v, param):
        Cumulative distribution function of the Gumbel copula.

    get_pdf(u, v, param):
        Probability density function of the Gumbel copula.

    kendall_tau(param):
        Computes Kendall's tau as a function of theta.

    sample(n, param):
        Generates n pseudo-observations from the copula.
    """

    def __init__(self):
        super().__init__()
        self.type = "gumbel"
        self.name = "Gumbel Copula"
        self.bounds_param = [(1+ 1e-6, None)]
        self.parameters_start = np.array(1.5)
        self.n_obs = None
        self.default_optim_method = "Powell"

    def get_cdf(self, u, v, param):
        """Gumbel copula CDF"""
        theta = param[0]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        log_u = -np.log(u)
        log_v = -np.log(v)
        sum_logs_theta = log_u ** theta + log_v ** theta
        return np.exp(-sum_logs_theta ** (1 / theta))

    def get_pdf(self, u, v, param):
        """Density of the Gumbel copula"""
        theta = param[0]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        log_u = -np.log(u)
        log_v = -np.log(v)

        sum_logs_theta = log_u ** theta + log_v ** theta
        C_uv = np.exp(-sum_logs_theta ** (1 / theta))

        term1 = 1 / (u * v)
        term2 = sum_logs_theta ** (-2 + 2 / theta)
        term3 = (log_u * log_v) ** (theta - 1)  # Corrigé ici
        term4 = theta + (theta - 1) * sum_logs_theta ** (-1 / theta)

        return C_uv * term1 * term2 * term3 * term4

    def kendall_tau(self, param):
        """Kendall's tau for Gumbel copula: tau = 1 - 1/theta"""
        theta = param[0]
        return 1 - 1 / theta

    def sample(self, n, param):
        """
        Sample from the Gumbel copula using conditional method.
        Due to complexity, we use an approximation via Marshall–Olkin method.
        """
        theta = param[0]

        if theta < 1:
            raise ValueError("Theta must be ≥ 1 for Gumbel copula.")
        elif theta == 1:
            return np.random.uniform(size=(n, 2))

        E = np.random.exponential(1, size=n)

        V = np.random.uniform(0, np.pi, size=n)
        W = np.random.exponential(1, size=n)

        # Stable distribution using CMS method
        alpha = 1.0 / theta
        S1 = (np.sin(alpha * V) / (np.cos(V)) ** (1 / alpha)) * \
             (np.cos((1 - alpha) * V) / W) ** ((1 - alpha) / alpha)
        S2 = (np.sin(alpha * V) / (np.cos(V)) ** (1 / alpha)) * \
             (np.cos((1 - alpha) * V) / np.random.exponential(1, size=n)) ** ((1 - alpha) / alpha)

        U = np.exp(-S1 ** (1 / theta) / E)
        V = np.exp(-S2 ** (1 / theta) / E)

        return np.column_stack((U, V))

    def LTDC(self, theta):
        """
        Computes the lower tail dependence coefficient for the Gumbel copula.

        Gumbel copula has no lower tail dependence:
            LTDC = 0
        """
        return 0.0

    def UTDC(self, theta):
        """
        Computes the upper tail dependence coefficient for the Gumbel copula.

        Formula:
            UTDC = 2 - 2^(1 / theta)
        """
        return 2 - 2 ** (1 / theta)

