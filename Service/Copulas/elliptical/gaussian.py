import numpy as np
from scipy.special import erfinv
from scipy.stats import norm, multivariate_normal

from Service.Copulas.base import BaseCopula


class GaussianCopula(BaseCopula):
    """
    Gaussian Copula class

    Attributes
    ----------
    family : str
        Identifier for the copula family. Here, "gaussian".
    name : str
        Human-readable name for output/logging.
    bounds_param : list of tuple
        Bounds for the copula parameters, used in optimization.
        For Gaussian: [(-0.999, 0.999)]
    parameters : np.ndarray
        Initial guess for the copula parameter(s), passed to the optimizer.

    Methods
    -------
    get_cdf(u, v, param)
        Computes the CDF of the Gaussian copula at (u, v).
    get_pdf(u, v, param)
        Computes the PDF of the Gaussian copula at (u, v).
    sample(n, param)
        Generates n samples from the copula, in [0,1]^2.
    """

    def __init__(self):
        super().__init__()
        self.type = "gaussian"
        self.name = "Gaussian Copula"
        self.bounds_param = [(-0.999, 0.999)]
        self.parameters = np.array([0.0])  # Initial guess for correlation coefficient
        self.n_obs = None # Number of data for the fit
        self.default_optim_method = "SLSQP"  # or "trust-constr"

    def get_cdf(self, u, v, param):
        """
        Computes the cumulative distribution function of the Gaussian copula.

        Parameters
        ----------
        u, v : float or array-like
            Pseudo-observations in [0, 1].
        param : list or array-like
            Copula parameter(s): param[0] is the correlation coefficient (ρ).

        Returns
        -------
        float or np.ndarray
            Copula CDF value(s) at (u, v)
        """
        rho = param[0]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        y1 = norm.ppf(u)
        y2 = norm.ppf(v)
        cov = [[1, rho], [rho, 1]]

        if np.isscalar(y1) and np.isscalar(y2):
            return multivariate_normal.cdf([y1, y2], mean=[0, 0], cov=cov)
        else:
            return np.array([
                multivariate_normal.cdf([a, b], mean=[0, 0], cov=cov)
                for a, b in zip(y1, y2)
            ])

    def get_pdf(self, u, v, param):
        """
        Computes the probability density function of the Gaussian copula.

        Parameters
        ----------
        u, v : float or array-like
            Pseudo-observations in [0, 1].
        param : list or array-like
            Copula parameter(s): param[0] is the correlation coefficient (ρ).

        Returns
        -------
        float or np.ndarray
            Copula PDF value(s) at (u, v)
        """
        rho = param[0]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        x = np.sqrt(2) * erfinv(2 * u - 1)
        y = np.sqrt(2) * erfinv(2 * v - 1)
        det_rho = 1 - rho ** 2

        exponent = -((x ** 2 + y ** 2) * rho ** 2 - 2 * x * y * rho) / (2 * det_rho)
        return (1.0 / np.sqrt(det_rho)) * np.exp(exponent)

    def kendall_tau(self, param):
        rho = param[0]
        return (2 / np.pi) * np.arcsin(rho)

    def sample(self, n, param):
        """
        Generate n samples from the Gaussian copula.

        Parameters
        ----------
        n : int
            Number of samples to generate
        param : list or array-like
            Copula parameter(s): param[0] is the correlation coefficient (ρ).

        Returns
        -------
        np.ndarray
            n x 2 array of pseudo-observations (u, v)
        """
        rho = param[0]
        cov = np.array([[1, rho], [rho, 1]])
        L = np.linalg.cholesky(cov)

        z = np.random.randn(n, 2)
        correlated = z @ L.T

        u = norm.cdf(correlated[:, 0])
        v = norm.cdf(correlated[:, 1])

        return np.column_stack((u, v))



