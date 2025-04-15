import numpy as np
from scipy.stats import uniform

from Service.Copulas.base import BaseCopula


class ClaytonCopula(BaseCopula):
    """
    Clayton Copula class (Archimedean copula)

    Attributes
    ----------
    type : str
        Identifier for the copula family. Here, 'clayton'.
    name : str
        Human-readable name for output/logging.
    bounds_param : list of tuple
        Bounds for the copula parameters, used in optimization.
        For Clayton: [(1e-6, None)]  where theta > 0 controls lower tail dependence.
    parameters : np.ndarray
        Initial guess for the copula parameter(s), passed to the optimizer.
    n_obs : int or None
        Number of observations used in the fit.
    default_optim_method : str
        Recommended optimizer for fitting.

    Methods
    -------
    get_cdf(u, v, param)
        Computes the CDF of the Clayton copula at (u, v).
    get_pdf(u, v, param)
        Computes the PDF of the Clayton copula at (u, v).
    kendall_tau(param)
        Computes Kendall's tau from the Clayton parameter.
    sample(n, param)
        Generates n samples from the Clayton copula.
    """

    def __init__(self):
        super().__init__()
        self.type = 'clayton'
        self.name = "Clayton Copula"
        self.bounds_param = [(1e-6, None)]
        self.parameters = np.array([0.5])  # initial guess for theta
        self.default_optim_method = "Powell"

    def get_cdf(self, u, v, param):
        """
        Computes the cumulative distribution function of the Clayton copula.
        """
        theta = param[0]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        return (u ** (-theta) + v ** (-theta) - 1) ** (-1 / theta)

    def get_pdf(self, u, v, param):
        """
        Computes the probability density function of the Clayton copula.
        """
        theta = param[0]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        term1 = (theta + 1) * (u * v) ** (-theta - 1)
        term2 = (u ** (-theta) + v ** (-theta) - 1) ** (-2 - 1 / theta)
        return term1 * term2

    def kendall_tau(self, param):
        """
        Computes Kendall's tau for the Clayton copula.

        tau = theta / (theta + 2)
        """
        theta = param[0]
        return theta / (theta + 2)

    def sample(self, n, param):
        """
        Generates n samples from the Clayton copula using the inverse transform method.

        Parameters
        ----------
        n : int
            Number of samples
        param : list or np.ndarray
            Parameter list (contains theta)

        Returns
        -------
        np.ndarray
            n x 2 array of uniform samples from the Clayton copula
        """
        theta = param[0]
        if theta <= 0:
            raise ValueError("Clayton copula requires theta > 0")

        u = uniform.rvs(size=n)
        w = uniform.rvs(size=n)
        v = (w ** (-theta / (1 + theta)) * (u ** (-theta) - 1) + 1) ** (-1 / theta)

        return np.column_stack((u, v))

    def LTDC(self, param):
        """
        Computes the lower tail dependence coefficient for the Clayton copula.

        Formula:
            LTDC = 2^(-1 / theta)
        """
        theta = param[0]

        return 2 ** (-1 / theta)

    def UTDC(self, param):
        """
        Computes the upper tail dependence coefficient for the Clayton copula.

        Clayton copula has no upper tail dependence:
            UTDC = 0
        """
        return 0.0

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """
        Compute the partial derivative ∂C(u,v)/∂v for the Clayton copula.

        The Clayton copula is defined by:
            C(u,v) = (u^(-theta) + v^(-theta) - 1)^(-1/theta).
        Let A = u^(-theta) + v^(-theta) - 1. Then,
            ∂C(u,v)/∂v = v^(-theta-1) * A^(-1/theta-1).

        Parameters
        ----------
        u : float or array-like
            Value(s) of u in (0, 1).
        v : float or array-like
            Value(s) of v in (0, 1).
        param : list or array-like, optional
            Copula parameter(s) in the form [theta]. If None, self.parameters is used.

        Returns
        -------
        float or np.ndarray
            The partial derivative ∂C(u,v)/∂v.
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        u = np.asarray(u)
        v = np.asarray(v)
        A = u ** (-theta) + v ** (-theta) - 1
        return v ** (-theta - 1) * A ** (-1 / theta - 1)

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """
        Compute the partial derivative ∂C(u,v)/∂u for the Clayton copula.

        With the same notation as before, we obtain:
            ∂C(u,v)/∂u = u^(-theta-1) * A^(-1/theta-1),
        where A = u^(-theta) + v^(-theta) - 1.

        Parameters
        ----------
        u : float or array-like
            Value(s) of u in (0, 1).
        v : float or array-like
            Value(s) of v in (0, 1).
        param : list or array-like, optional
            Copula parameter(s) in the form [theta]. If None, self.parameters is used.

        Returns
        -------
        float or np.ndarray
            The partial derivative ∂C(u,v)/∂u.
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        u = np.asarray(u)
        v = np.asarray(v)
        A = u ** (-theta) + v ** (-theta) - 1
        return u ** (-theta - 1) * A ** (-1 / theta - 1)

    def conditional_cdf_u_given_v(self, u, v, param=None):
        """
        Compute the conditional probability P(U ≤ u | V = v) for the Clayton copula.

        Since the copula has uniform margins (i.e. C(1,v)=v so that f_V(v)=1),
        the conditional CDF equals the partial derivative with respect to v:

            P(U ≤ u | V = v) = ∂C(u,v)/∂v.

        Parameters
        ----------
        u : float or array-like
            Value(s) of u in (0, 1).
        v : float or array-like
            Value(s) of v in (0, 1).
        param : list or array-like, optional
            Copula parameter(s) in the form [theta]. If None, self.parameters is used.

        Returns
        -------
        float or np.ndarray
            The conditional probability P(U ≤ u | V = v).
        """
        return self.partial_derivative_C_wrt_v(u, v, param)

    def conditional_cdf_v_given_u(self, u, v, param=None):
        """
        Compute the conditional probability P(V ≤ v | U = u) for the Clayton copula.

        Similarly, since C(u,1)=u so that f_U(u)=1, we obtain:

            P(V ≤ v | U = u) = ∂C(u,v)/∂u.

        Parameters
        ----------
        u : float or array-like
            Value(s) of u in (0, 1).
        v : float or array-like
            Value(s) of v in (0, 1).
        param : list or array-like, optional
            Copula parameter(s) in the form [theta]. If None, self.parameters is used.

        Returns
        -------
        float or np.ndarray
            The conditional probability P(V ≤ v | U = u).
        """
        return self.partial_derivative_C_wrt_u(u, v, param)

