import numpy as np
from scipy.special import beta  # Used in Kendall's tau calculation
from Service.Copulas.base import BaseCopula

class BB1Copula(BaseCopula):
    """
    BB1 Copula (Two-parameter Archimedean Copula)

    Defined by:
        C(u,v) = [1 + { (u^(-theta) - 1)^delta + (v^(-theta) - 1)^delta }^(1/delta)]^(-1/theta)

    with theta > 0 and delta >= 1. When delta = 1, this reduces to the Clayton copula.

    Attributes
    ----------
    type : str
        Copula identifier.
    name : str
        Human-readable name of the copula.
    bounds_param : list of tuple
        Parameter bounds. For BB1: theta > 0, delta ≥ 1.
    parameters_start : np.ndarray
        Initial parameter guesses.
    n_obs : int or None
        Number of observations, populated during fitting.
    default_optim_method : str
        Default optimization method.

    Methods
    -------
    get_cdf(u, v, param):
        Cumulative distribution function of the BB1 copula.
    get_pdf(u, v, param):
        Probability density function of the BB1 copula.
    kendall_tau(param):
        Computes Kendall's tau for the copula.
    sample(n, param):
        Generates n pseudo-observations using a Marshall–Olkin-type method.
    LTDC(param):
        Lower tail dependence coefficient.
    UTDC(param):
        Upper tail dependence coefficient.
    """

    def __init__(self):
        super().__init__()
        self.type = "BB1"
        self.name = "BB1 Copula"
        self.bounds_param = [(1e-6, None), (1, None)]
        self.parameters = np.array([0.5, 1.5])
        self.default_optim_method = "Powell"

    def get_cdf(self, u, v, param):
        theta = param[0]
        delta = param[1]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        term1 = (u ** (-theta) - 1) ** delta
        term2 = (v ** (-theta) - 1) ** delta
        inner = term1 + term2
        return (1 + inner ** (1 / delta)) ** (-1 / theta)

    def get_pdf(self, u, v, param):
        """
        Computes the BB1 copula density function (PDF) using the closed-form expression.

        Based on:
            Joe (1997) - Multivariate Models and Dependence Concepts
            Hofert et al. (2012) - Sampling Archimedean Copulas

        This form avoids partial derivatives and simplifies numerical evaluation.
        """
        theta, delta = param[0], param[1]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        x = (u ** (-theta) - 1) ** delta
        y = (v ** (-theta) - 1) ** delta
        S = x + y

        term1 = (1 + S ** (1 / delta)) ** (-1 / theta - 2)
        term2 = S ** (1 / delta - 2)
        term3 = theta * (delta - 1) + (theta * delta + 1) * S ** (1 / delta)
        term4 = (x * y) ** (1 - 1 / delta) * (u * v) ** (-theta - 1)

        return term1 * term2 * term3 * term4

    def kendall_tau(self, param):
        """
        Computes Kendall's tau using the beta function.

        For BB1 copula:
            tau = 1 - (2 / delta) * (1 - 1 / theta) * B(1 - 1 / theta, 2 / delta + 1)
        """
        theta = param[0]
        delta = param[1]
        return 1 - (2 / delta) * (1 - 1 / theta) * beta(1 - 1 / theta, 2 / delta + 1)

    def sample(self, n, param):
        """
        Samples from the BB1 copula using a generalized Marshall–Olkin approach,
        as described by Hofert et al. (2012).

        This method uses a latent Gamma variable and two independent exponential draws.
        """
        theta = param[0]
        delta = param[1]

        if theta <= 0 or delta < 1:
            raise ValueError("Invalid parameters: theta must be > 0 and delta ≥ 1.")

        # Step 1: Draw V ~ Gamma(1/delta, 1)
        V = np.random.gamma(1.0 / delta, 1.0, size=n)

        # Step 2: Draw independent exponentials
        E1 = np.random.exponential(scale=1.0, size=n)
        E2 = np.random.exponential(scale=1.0, size=n)

        # Step 3: Transform into uniform margins
        U = (1 + (E1 / V) ** (1 / delta)) ** (-1 / theta)
        W = (1 + (E2 / V) ** (1 / delta)) ** (-1 / theta)

        return np.column_stack((U, W))

    def LTDC(self, param):
        """
        Computes the lower tail dependence coefficient (LTDC).

        For small u:
            C(u, u) ~ 2^(-1 / (delta * theta)) * u  → LTDC = 2^(-1 / (delta * theta))
        """
        theta = param[0]
        delta = param[1]
        return 2 ** (-1 / (delta * theta))

    def UTDC(self, param):
        """
        Computes the upper tail dependence coefficient (UTDC).

        As u → 1, the asymptotic behavior gives:
            lambda_U = 2 - 2^(1 / delta)
        """
        delta = param[1]
        return 2 - 2 ** (1 / delta)

    def conditional_cdf_u_given_v(self, u, v, param):
        """
        Analytically computes the conditional CDF P(U ≤ u | V = v)
        for the BB1 copula.

        For the BB1 copula defined as:
            C(u,v) = [ 1 + { (u^(-θ)-1)^δ + (v^(-θ)-1)^δ }^(1/δ) ]^(-1/θ),
        let T = (u^(-θ)-1)^δ + (v^(-θ)-1)^δ.

        Then, the derivative with respect to v is given by:
            ∂C(u,v)/∂v = [1+T^(1/δ)]^(-1/θ-1) * T^(1/δ-1)
                            * (v^(-θ)-1)^(δ-1) * v^(-θ-1).

        Since C(1,v)=v and ∂C(1,v)/∂v=1, we have:
            F_{U|V}(u|v) = ∂C(u,v)/∂v.

        Parameters
        ----------
        u : float or array-like
            Values in (0,1) for U.
        v : float or array-like
            Values in (0,1) for V (conditioning variable).
        param : list or array-like, optional
            Parameters [θ, δ] for the BB1 copula. If None, self.parameters is used.

        Returns
        -------
        float or np.ndarray
            The conditional CDF P(U ≤ u | V = v).
        """

        theta, delta = param[0], param[1]

        u = np.asarray(u)
        v = np.asarray(v)

        T = (u ** (-theta) - 1) ** delta + (v ** (-theta) - 1) ** delta
        factor = (1 + T ** (1 / delta)) ** (-1 / theta - 1)
        return factor * T ** (1 / delta - 1) * (v ** (-theta) - 1) ** (delta - 1) * v ** (-theta - 1)

    def conditional_cdf_v_given_u(self, v, u, param):
        """
        Analytically computes the conditional CDF P(V ≤ v | U = u)
        for the BB1 copula.

        For the BB1 copula defined as:
            C(u,v) = [ 1 + { (u^(-θ)-1)^δ + (v^(-θ)-1)^δ }^(1/δ) ]^(-1/θ),
        let T = (u^(-θ)-1)^δ + (v^(-θ)-1)^δ.

        Then, the derivative with respect to u is given by:
            ∂C(u,v)/∂u = [1+T^(1/δ)]^(-1/θ-1) * T^(1/δ-1)
                            * (u^(-θ)-1)^(δ-1) * u^(-θ-1).

        Since C(u,1)=u (and ∂C(u,1)/∂u=1), we have:
            F_{V|U}(v|u) = ∂C(u,v)/∂u.

        Parameters
        ----------
        v : float or array-like
            Values in (0,1) for V.
        u : float or array-like
            Values in (0,1) for U (conditioning variable).
        param : list or array-like, optional
            Parameters [θ, δ] for the BB1 copula. If None, self.parameters is used.

        Returns
        -------
        float or np.ndarray
            The conditional CDF P(V ≤ v | U = u).
        """

        theta, delta = param[0], param[1]

        u = np.asarray(u)
        v = np.asarray(v)

        T = (u ** (-theta) - 1) ** delta + (v ** (-theta) - 1) ** delta
        factor = (1 + T ** (1 / delta)) ** (-1 / theta - 1)
        return factor * T ** (1 / delta - 1) * (u ** (-theta) - 1) ** (delta - 1) * u ** (-theta - 1)
