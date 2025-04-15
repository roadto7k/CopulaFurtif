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
        self.parameters = np.array([1.5])
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

    def LTDC(self, param):
        """
        Computes the lower tail dependence coefficient for the Gumbel copula.

        Gumbel copula has no lower tail dependence:
            LTDC = 0
        """
        return 0.0

    def UTDC(self, param):
        """
        Computes the upper tail dependence coefficient for the Gumbel copula.

        Formula:
            UTDC = 2 - 2^(1 / theta)
        """
        theta = param[0]

        return 2 - 2 ** (1 / theta)

    def partial_derivative_C_wrt_v(self, u, v, param):
        """
        Compute the partial derivative ∂C(u,v)/∂v for the Gumbel copula.

        Using the formula:
            ∂C(u,v)/∂v = [C(u,v)/v] * A^(1/θ - 1) * (-ln v)^(θ - 1),
        where A = (-ln u)^θ + (-ln v)^θ.

        Parameters
        ----------
        u : float or array-like
            Values in (0,1) for U.
        v : float or array-like
            Values in (0,1) for V.
        param : iterable
            Copula parameter(s) as [theta].

        Returns
        -------
        float or np.ndarray
            The partial derivative ∂C(u,v)/∂v.
        """
        theta = param[0]
        u = np.asarray(u)
        v = np.asarray(v)
        A = (-np.log(u))**theta + (-np.log(v))**theta
        Cuv = np.exp(-A**(1/theta))
        return (Cuv / v) * (A)**(1/theta - 1) * ((-np.log(v))**(theta - 1))

    def partial_derivative_C_wrt_u(self, u, v, param):
        """
        Compute the partial derivative ∂C(u,v)/∂u for the Gumbel copula.

        Using the symmetric formula:
            ∂C(u,v)/∂u = [C(u,v)/u] * A^(1/θ - 1) * (-ln u)^(θ - 1),
        where A = (-ln u)^θ + (-ln v)^θ.

        Parameters
        ----------
        u : float or array-like
            Values in (0,1) for U.
        v : float or array-like
            Values in (0,1) for V.
        param : iterable
            Copula parameter(s) as [theta].

        Returns
        -------
        float or np.ndarray
            The partial derivative ∂C(u,v)/∂u.
        """
        theta = param[0]
        u = np.asarray(u)
        v = np.asarray(v)
        A = (-np.log(u))**theta + (-np.log(v))**theta
        Cuv = np.exp(-A**(1/theta))
        return (Cuv / u) * (A)**(1/theta - 1) * ((-np.log(u))**(theta - 1))

    def conditional_cdf_u_given_v(self, u, v, param):
        """
        Compute the conditional CDF P(U ≤ u | V = v) for the Gumbel copula.

        Defined as:
            F_{U|V}(u|v) = [∂C(u,v)/∂v] / [∂C(1,v)/∂v].

        Since C(1,v)=v and therefore ∂C(1,v)/∂v=1,
        the normalization is automatic—but the division is performed for consistency.

        Parameters
        ----------
        u : float or array-like
            The u-value (in (0,1)) at which the conditional CDF is evaluated.
        v : float or array-like
            The fixed v-value (in (0,1)).
        param : iterable
            Copula parameter(s) as [theta].

        Returns
        -------
        float or np.ndarray
            The computed conditional CDF P(U ≤ u | V = v).
        """
        num = self.partial_derivative_C_wrt_v(u, v, param)
        den = self.partial_derivative_C_wrt_v(1.0, v, param)
        eps = 1e-14
        den = np.maximum(den, eps)
        return num / den

    def conditional_cdf_v_given_u(self, v, u, param):
        """
        Compute the conditional CDF P(V ≤ v | U = u) for the Gumbel copula.

        Defined as:
            F_{V|U}(v|u) = [∂C(u,v)/∂u] / [∂C(u,1)/∂u].

        Since C(u,1)=u and hence ∂C(u,1)/∂u=1,
        the normalization is automatic—but we include the division for consistency.

        Parameters
        ----------
        v : float or array-like
            The v-value (in (0,1)) at which the conditional CDF is evaluated.
        u : float or array-like
            The fixed u-value (in (0,1)).
        param : iterable
            Copula parameter(s) as [theta].

        Returns
        -------
        float or np.ndarray
            The computed conditional CDF P(V ≤ v | U = u).
        """
        num = self.partial_derivative_C_wrt_u(u, v, param)
        den = self.partial_derivative_C_wrt_u(u, 1.0, param)
        eps = 1e-14
        den = np.maximum(den, eps)
        return num / den

