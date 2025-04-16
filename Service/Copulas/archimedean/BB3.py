import numpy as np
from scipy.optimize import brentq

from Service.Copulas.base import BaseCopula


class BB3Copula(BaseCopula):
    """
    BB3 Copula class

    Attributes
    ----------
    family : str
        Identifier for the copula family. Here, "bb3".
    name : str
        Human-readable name for output/logging.
    bounds_param : list of tuple
        Bounds for the copula parameters, used in optimization:
        [(d_lower, d_upper), (q_lower, q_upper)].
    parameters : np.ndarray
        Initial guess for the copula parameters [d, q].
    default_optim_method : str
        Default optimizer method for parameter fitting.
    """

    def __init__(self):
        """
        Initialize BB3 copula with default settings.

        Sets the copula type, name, parameter bounds, initial guesses,
        and the default optimization method.
        """
        super().__init__()
        self.type = "bb3"
        self.name = "BB3 Copula"
        # d > 0, q >= 1
        self.bounds_param = [(1e-6, None), (1.0, None)]
        self.parameters = np.array([1.0, 1.0])
        self.default_optim_method = "SLSQP"

    def _h(self, s, param):
        """
        Generator function h(s) for BB3 copula.

        h(s) = [ (1/d) * log(1 + s) ]^(1/q)

        Parameters
        ----------
        s : float or np.ndarray
            Argument to the generator function.
        param : array_like
            Copula parameters [d, q].

        Returns
        -------
        float or np.ndarray
            Value of generator function at s.
        """
        d, q = param
        return (np.log1p(s) / d)**(1.0 / q)

    def _h_prime(self, s, param):
        """
        First derivative of generator function h with respect to s.

        Parameters
        ----------
        s : float or np.ndarray
            Argument to the generator function.
        param : array_like
            Copula parameters [d, q].

        Returns
        -------
        float or np.ndarray
            First derivative h'(s).
        """
        d, q = param
        g = np.log1p(s) / d
        return (1.0 / (q * d * (1.0 + s))) * g**(1.0 / q - 1.0)

    def _h_double(self, s, param):
        """
        Second derivative of generator function h with respect to s.

        Parameters
        ----------
        s : float or np.ndarray
            Argument to the generator function.
        param : array_like
            Copula parameters [d, q].

        Returns
        -------
        float or np.ndarray
            Second derivative h''(s).
        """
        d, q = param
        g = np.log1p(s) / d
        A = 1.0 / (q * d * (1.0 + s)**2)
        term1 = -g**(1.0/q - 1.0)
        term2 = (1.0/q - 1.0) * g**(1.0/q - 2.0) / d
        return A * (term1 + term2)

    def get_cdf(self, u, v, param):
        """
        Compute the cumulative distribution function C(u, v) of the BB3 copula.

        Parameters
        ----------
        u, v : float or np.ndarray
            Uniform margins on [0, 1].
        param : array_like
            Copula parameters [d, q].

        Returns
        -------
        float or np.ndarray
            Copula CDF evaluated at (u, v).
        """
        d, q = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        # inverse generator: s = phi^{-1}(u)
        s_u = np.expm1(d * (-np.log(u))**q)
        s_v = np.expm1(d * (-np.log(v))**q)
        s = s_u + s_v
        # C(u,v) = exp(-h(s))
        return np.exp(-self._h(s, param))

    def get_pdf(self, u, v, param):
        """
        Compute the probability density function c(u, v) of the BB3 copula.

        Parameters
        ----------
        u, v : float or np.ndarray
            Uniform margins on [0, 1].
        param : array_like
            Copula parameters [d, q].

        Returns
        -------
        float or np.ndarray
            Copula PDF evaluated at (u, v).
        """
        d, q = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        # inverse generator values
        s_u = np.expm1(d * (-np.log(u))**q)
        s_v = np.expm1(d * (-np.log(v))**q)
        s = s_u + s_v
        # generator derivatives
        h = self._h(s, param)
        h1 = self._h_prime(s, param)
        h2 = self._h_double(s, param)
        phi_dd = np.exp(-h) * (h1**2 - h2)
        # derivative of inverse generator at u, v
        phi_inv_u_prime = -d * q * np.exp(d * (-np.log(u))**q) * ((-np.log(u))**(q - 1) / u)
        phi_inv_v_prime = -d * q * np.exp(d * (-np.log(v))**q) * ((-np.log(v))**(q - 1) / v)
        return phi_dd * phi_inv_u_prime * phi_inv_v_prime

    def kendall_tau(self, param):
        """
        Compute Kendall's tau for the BB3 copula.

        Parameters
        ----------
        param : array_like
            Copula parameters [d, q].

        Raises
        ------
        NotImplementedError
            Method not implemented.
        """
        raise NotImplementedError("Kendall's tau not implemented for BB3.")

    def sample(self, n, param):
        """
        Generate samples from the BB3 copula via conditional sampling.

        Parameters
        ----------
        n : int
            Number of samples to generate.
        param : array_like
            Copula parameters [d, q].

        Returns
        -------
        np.ndarray
            An (n, 2) array of samples from the copula.
        """
        samples = np.empty((n, 2))
        for i in range(n):
            u = np.random.rand()
            p = np.random.rand()
            # find v such that P(V <= v | U = u) = p
            root = brentq(
                lambda v: self.conditional_cdf_v_given_u(u, v, param) - p,
                1e-6,
                1 - 1e-6,
            )
            samples[i, 0] = u
            samples[i, 1] = root
        return samples

    def LTDC(self, param):
        """
        Lower tail dependence coefficient of the BB3 copula.

        Parameters
        ----------
        param : array_like
            Copula parameters [d, q].

        Returns
        -------
        float
            Lower tail dependence (always 0 for BB3).
        """
        return 0.0

    def UTDC(self, param):
        """
        Upper tail dependence coefficient of the BB3 copula.

        Parameters
        ----------
        param : array_like
            Copula parameters [d, q].

        Returns
        -------
        float
            Upper tail dependence: 2 - 2^(1/q).
        """
        q = param[1]
        return 2 - 2**(1.0 / q)

    def partial_derivative_C_wrt_v(self, u, v, param):
        """
        Compute the partial derivative ∂C/∂v of the copula CDF.

        Parameters
        ----------
        u, v : float or np.ndarray
            Points at which to evaluate the derivative.
        param : array_like
            Copula parameters [d, q].

        Returns
        -------
        float or np.ndarray
            Partial derivative of C with respect to v.
        """
        d, q = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        s_u = np.expm1(d * (-np.log(u))**q)
        s_v = np.expm1(d * (-np.log(v))**q)
        s = s_u + s_v
        h1 = self._h_prime(s, param)
        phi_p = -h1 * np.exp(-self._h(s, param))
        phi_inv_v_prime = -d * q * np.exp(d * (-np.log(v))**q) * ((-np.log(v))**(q - 1) / v)
        return phi_p * phi_inv_v_prime

    def partial_derivative_C_wrt_u(self, u, v, param):
        """
        Compute the partial derivative ∂C/∂u of the copula CDF.

        Parameters
        ----------
        u, v : float or np.ndarray
            Points at which to evaluate the derivative.
        param : array_like
            Copula parameters [d, q].

        Returns
        -------
        float or np.ndarray
            Partial derivative of C with respect to u.
        """
        d, q = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        s_u = np.expm1(d * (-np.log(u))**q)
        s_v = np.expm1(d * (-np.log(v))**q)
        s = s_u + s_v
        h1 = self._h_prime(s, param)
        phi_p = -h1 * np.exp(-self._h(s, param))
        phi_inv_u_prime = -d * q * np.exp(d * (-np.log(u))**q) * ((-np.log(u))**(q - 1) / u)
        return phi_p * phi_inv_u_prime

    def conditional_cdf_u_given_v(self, u, v, param):
        """
        Conditional CDF P(U <= u | V = v).

        Parameters
        ----------
        u, v : float or np.ndarray
            Points at which to evaluate the conditional CDF.
        param : array_like
            Copula parameters [d, q].

        Returns
        -------
        float or np.ndarray
            Conditional CDF value.
        """
        num = self.partial_derivative_C_wrt_v(u, v, param)
        den = self.partial_derivative_C_wrt_v(1.0, v, param)
        return num / den

    def conditional_cdf_v_given_u(self, u, v, param):
        """
        Conditional CDF P(V <= v | U = u).

        Parameters
        ----------
        u, v : float or np.ndarray
            Points at which to evaluate the conditional CDF.
        param : array_like
            Copula parameters [d, q].

        Returns
        -------
        float or np.ndarray
            Conditional CDF value.
        """
        num = self.partial_derivative_C_wrt_u(u, v, param)
        den = self.partial_derivative_C_wrt_u(u, 1.0, param)
        return num / den
