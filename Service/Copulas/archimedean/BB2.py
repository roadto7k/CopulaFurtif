import numpy as np
from scipy.special import beta  # used in kendall_tau

from Service.Copulas.archimedean.BB1 import BB1Copula
from Service.Copulas.base import BaseCopula

class BB2Copula(BaseCopula):
    """
    BB2 Copula (Survival Version of the BB1 Copula)

    The BB2 copula is defined as the survival copula of the BB1 copula:
        C_{BB2}(u,v) = u + v - 1 + C_{BB1}(1-u, 1-v),
    where the BB1 copula is given by:
        C_{BB1}(u,v) = [1 + {(u^(-theta) - 1)^delta + (v^(-theta) - 1)^delta}^{1/delta}]^(-1/theta).

    This construction exchanges the lower and upper tail behavior compared to BB1.

    Attributes
    ----------
    type : str
        Copula identifier.
    name : str
        Human-readable name.
    bounds_param : list of tuple
        Parameter bounds; for BB2, theta > 0 and delta ≥ 1.
    parameters_start : np.ndarray
        Initial parameter guesses.
    n_obs : int or None
        Number of observations (set during fitting).
    default_optim_method : str
        Default optimization method.

    Methods
    -------
    get_cdf(u, v, param):
        Cumulative distribution function of the BB2 copula.
    get_pdf(u, v, param):
        Probability density function of the BB2 copula.
    kendall_tau(param):
        Computes Kendall's tau (remains the same as BB1).
    sample(n, param):
        Generates n pseudo-observations using the BB1 sampling method with a survival transformation.
    LTDC(param):
        Lower tail dependence coefficient (equal to BB1's upper tail dependence).
    UTDC(param):
        Upper tail dependence coefficient (equal to BB1's lower tail dependence).
    """

    def __init__(self):
        super().__init__()
        self.type = "BB2"
        self.name = "BB2 Copula"
        self.bounds_param = [(1e-6, None), (1e-6, None)]
        self.parameters = np.array([1, 1])
        self.default_optim_method = "Powell"

    def get_cdf(self, u, v, param):
        """
        Computes the cumulative distribution function (CDF) of the BB2 copula
        using the survival transform:
            C_{BB2}(u,v) = u + v - 1 + C_{BB1}(1-u,1-v).
        """
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        theta, delta = param[0], param[1]
        U_bar = 1 - u
        V_bar = 1 - v
        # Compute BB1 CDF at (1-u,1-v)
        term1 = (U_bar ** (-theta) - 1) ** delta
        term2 = (V_bar ** (-theta) - 1) ** delta
        inner = term1 + term2
        C_BB1 = (1 + inner ** (1 / delta)) ** (-1 / theta)
        return u + v - 1 + C_BB1

    def get_pdf(self, u, v, param):
        """
        Computes the probability density function (PDF) of the BB2 copula.
        Since BB2 is the survival copula of BB1, its density is given by:
            c_{BB2}(u,v) = c_{BB1}(1-u,1-v),
        using the closed-form density for BB1.
        """
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        theta, delta = param[0], param[1]
        U_bar = 1 - u
        V_bar = 1 - v
        # Compute BB1 density at (1-u,1-v)
        x = (U_bar ** (-theta) - 1) ** delta
        y = (V_bar ** (-theta) - 1) ** delta
        S = x + y
        term1 = (1 + S ** (1 / delta)) ** (-1 / theta - 2)
        term2 = S ** (1 / delta - 2)
        term3 = theta * (delta - 1) + (theta * delta + 1) * S ** (1 / delta)
        term4 = (x * y) ** (1 - 1 / delta) * (U_bar * V_bar) ** (-theta - 1)
        return term1 * term2 * term3 * term4

    def kendall_tau(self, param):
        """
        Computes Kendall's tau for the BB2 copula.
        Since the survival transform does not change Kendall's tau,
        the formula remains the same as for BB1:
            tau = 1 - (2 / delta) * (1 - 1/theta) * B(1 - 1/theta, 2/delta + 1).
        """
        theta, delta = param[0], param[1]
        return 1 - (2 / delta) * (1 - 1 / theta) * beta(1 - 1 / theta, 2 / delta + 1)

    def sample(self, n, param):
        """
        Generates n pseudo-observations from the BB2 copula.
        This is achieved by sampling from the BB1 copula using its Marshall–Olkin-type method
        and then applying the survival transformation:
            (u,v) ~ BB2  if and only if  (1-u, 1-v) ~ BB1.
        """
        theta, delta = param[0], param[1]
        if theta <= 0 or delta < 1:
            raise ValueError("Invalid parameters: theta must be > 0 and delta ≥ 1.")

        # BB1 sampling
        V = np.random.gamma(1.0 / delta, 1.0, size=n)
        E1 = np.random.exponential(scale=1.0, size=n)
        E2 = np.random.exponential(scale=1.0, size=n)
        U_BB1 = (1 + (E1 / V) ** (1 / delta)) ** (-1 / theta)
        W_BB1 = (1 + (E2 / V) ** (1 / delta)) ** (-1 / theta)
        # Apply survival transformation: (u,v) = (1 - U_BB1, 1 - W_BB1)
        u_samples = 1 - U_BB1
        v_samples = 1 - W_BB1
        return np.column_stack((u_samples, v_samples))

    def LTDC(self, param):
        """
        Computes the lower tail dependence coefficient (LTDC) for BB2.
        For the survival copula BB2, the lower tail dependence equals the upper tail dependence of BB1:
            LTDC = 2 - 2^(1/delta).
        """
        theta, delta = param[0], param[1]
        return 2 - 2 ** (1 / delta)

    def UTDC(self, param):
        """
        Computes the upper tail dependence coefficient (UTDC) for BB2.
        For the survival copula BB2, the upper tail dependence equals the lower tail dependence of BB1:
            UTDC = 2^(-1/(theta*delta)).
        """
        theta, delta = param[0], param[1]
        return 2 ** (-1 / (theta * delta))

    def conditional_cdf_u_given_v(self, u, v, param):
        """
        Computes F_{U|V}^{BB2}(u | v) for the BB2 copula using the survival
        transformation of the conditional CDF from the BB1 copula.

        That is:
            F_{U|V}^{BB2}(u | v) = 1 - F_{U|V}^{BB1}(1 - u | 1 - v).
        """

        # Ensure param is an array (flattened)
        param = np.asarray(param).flatten()
        # Create an instance of BB1 to use its conditional CDFs.
        bb1 = BB1Copula()
        bb1.parameters = param  # using the same two parameters [theta, delta]
        return 1 - bb1.conditional_cdf_u_given_v(1 - u, 1 - v, param)

    def conditional_cdf_v_given_u(self, v, u, param):
        """
        Computes F_{V|U}^{BB2}(v | u) for the BB2 copula using the survival
        transformation of the conditional CDF from the BB1 copula.

        That is:
            F_{V|U}^{BB2}(v | u) = 1 - F_{V|U}^{BB1}(1 - v | 1 - u).
        """

        param = np.asarray(param).flatten()
        bb1 = BB1Copula()
        bb1.parameters = param
        return 1 - bb1.conditional_cdf_v_given_u(1 - v, 1 - u, param)
