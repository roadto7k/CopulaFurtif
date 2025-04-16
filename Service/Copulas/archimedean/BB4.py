import numpy as np
from scipy.optimize import brentq

from Service.Copulas.base import BaseCopula

class BB4Copula(BaseCopula):
    """
    BB4 Copula class

    Attributes
    ----------
    family : str
        Identifier for the copula family. Here, "bb4".
    name : str
        Human-readable name for output/logging.
    bounds_param : list of tuple
        Bounds for the copula parameters, used in optimization:
        [(mu_lower, mu_upper), (delta_lower, delta_upper)].
    parameters : np.ndarray
        Initial guess for the copula parameters [mu, delta].
    default_optim_method : str
        Default optimizer method for parameter fitting.

    Methods
    -------
    get_cdf(u, v, param)
        Computes the CDF of the BB4 copula at (u, v).
    get_pdf(u, v, param)
        Computes the PDF of the BB4 copula at (u, v).
    sample(n, param)
        Generates n samples via conditional inversion.
    kendall_tau(param)
        (Not implemented) Computes Kendall's tau.
    LTDC(param)
        Lower tail dependence coefficient (here: 0).
    UTDC(param)
        Upper tail dependence coefficient: 2 - 2^(1/delta).
    partial_derivative_C_wrt_u(u, v, param)
        Computes ∂C/∂u.
    partial_derivative_C_wrt_v(u, v, param)
        Computes ∂C/∂v.
    conditional_cdf_u_given_v(u, v, param)
        P(U ≤ u | V = v).
    conditional_cdf_v_given_u(u, v, param)
        P(V ≤ v | U = u).
    """
    def __init__(self):
        super().__init__()
        self.type = "bb4"
        self.name = "BB4 Copula"
        # mu > 0, delta > 0
        self.bounds_param = [(1e-6, None), (1e-6, None)]
        self.parameters = np.array([1.0, 1.0])
        self.default_optim_method = "SLSQP"

    def get_cdf(self, u, v, param):
        """
        Computes the cumulative distribution function of the BB4 copula.

        Parameters
        ----------
        u, v : float or array-like
            Pseudo-observations in (0, 1).
        param : list or array-like
            Copula parameters: [mu, delta].

        Returns
        -------
        float or np.ndarray
            Copula CDF value(s) at (u, v).
        """
        mu, delta = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        x = u ** (-mu)
        y = v ** (-mu)
        T = (x - 1) ** (-delta) + (y - 1) ** (-delta)
        A = T ** (-1.0 / delta)
        z = x + y - 1 - A
        return z ** (-1.0 / mu)

    def get_pdf(self, u, v, param):
        """
        Computes the probability density function of the BB4 copula.

        Uses analytic second derivatives of the copula generator.

        Parameters
        ----------
        u, v : float or array-like
            Pseudo-observations in (0, 1).
        param : list or array-like
            Copula parameters: [mu, delta].

        Returns
        -------
        float or np.ndarray
            Copula PDF value(s) at (u, v).
        """
        mu, delta = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        x = u ** (-mu)
        y = v ** (-mu)
        T = (x - 1) ** (-delta) + (y - 1) ** (-delta)
        A = T ** (-1.0 / delta)
        z = x + y - 1 - A
        # derivatives
        dzdu = -mu * u ** (-mu - 1) * (1 - (x - 1) ** (-delta - 1) * T ** (-1.0 / delta - 1))
        dzdv = -mu * v ** (-mu - 1) * (1 - (y - 1) ** (-delta - 1) * T ** (-1.0 / delta - 1))
        d2zdudv = -(mu ** 2) * (delta + 1) * u ** (-mu - 1) * v ** (-mu - 1) * \
            (x - 1) ** (-delta - 1) * (y - 1) ** (-delta - 1) * T ** (-1.0 / delta - 2)
        dCdz = -1.0 / mu * z ** (-1.0 / mu - 1)
        d2Cdz2 = (1.0 / mu) * (1.0 / mu + 1) * z ** (-1.0 / mu - 2)
        return d2Cdz2 * dzdu * dzdv + dCdz * d2zdudv

    def kendall_tau(self, param):
        """
        Kendall's tau is not implemented for BB4.
        """
        raise NotImplementedError("Kendall's tau not implemented for BB4.")

    def sample(self, n, param):
        """
        Generates n samples from the BB4 copula via conditional inversion.

        Parameters
        ----------
        n : int
            Number of samples.
        param : list or array-like
            Copula parameters: [mu, delta].

        Returns
        -------
        np.ndarray
            n x 2 array of pseudo-observations.
        """
        samples = np.empty((n, 2))
        for i in range(n):
            u = np.random.rand()
            p = np.random.rand()
            root = brentq(
                lambda v: self.conditional_cdf_v_given_u(u, v, param) - p,
                1e-6,
                1 - 1e-6
            )
            samples[i, 0] = u
            samples[i, 1] = root
        return samples

    def LTDC(self, param):
        """
        Lower tail dependence coefficient for BB4 (zero).
        """
        return 0.0

    def UTDC(self, param):
        """
        Upper tail dependence coefficient: 2 - 2^(1/delta).
        """
        delta = param[1]
        return 2 - 2 ** (1.0 / delta)

    def partial_derivative_C_wrt_u(self, u, v, param):
        """
        Computes ∂C(u,v)/∂u for the BB4 copula.
        """
        mu, delta = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        x = u ** (-mu)
        y = v ** (-mu)
        T = (x - 1) ** (-delta) + (y - 1) ** (-delta)
        A = T ** (-1.0 / delta)
        z = x + y - 1 - A
        dzdu = -mu * u ** (-mu - 1) * (1 - (x - 1) ** (-delta - 1) * T ** (-1.0 / delta - 1))
        dCdz = -1.0 / mu * z ** (-1.0 / mu - 1)
        return dCdz * dzdu

    def partial_derivative_C_wrt_v(self, u, v, param):
        """
        Computes ∂C(u,v)/∂v for the BB4 copula.
        """
        mu, delta = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        x = u ** (-mu)
        y = v ** (-mu)
        T = (x - 1) ** (-delta) + (y - 1) ** (-delta)
        A = T ** (-1.0 / delta)
        z = x + y - 1 - A
        dzdv = -mu * v ** (-mu - 1) * (1 - (y - 1) ** (-delta - 1) * T ** (-1.0 / delta - 1))
        dCdz = -1.0 / mu * z ** (-1.0 / mu - 1)
        return dCdz * dzdv

    def conditional_cdf_u_given_v(self, u, v, param):
        """
        P(U ≤ u | V = v) = ∂C/∂v / ∂C/∂v at u=1
        """
        num = self.partial_derivative_C_wrt_v(u, v, param)
        den = self.partial_derivative_C_wrt_v(1.0, v, param)
        return num / den

    def conditional_cdf_v_given_u(self, u, v, param):
        """
        P(V ≤ v | U = u) = ∂C/∂u / ∂C/∂u at v=1
        """
        num = self.partial_derivative_C_wrt_u(u, v, param)
        den = self.partial_derivative_C_wrt_u(u, 1.0, param)
        return num / den
