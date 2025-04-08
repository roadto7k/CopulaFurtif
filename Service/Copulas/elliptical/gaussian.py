import numpy as np
from scipy.special import erfinv
from scipy.stats import norm, multivariate_normal
import scipy.stats as st

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

    def LTDC(self, param):
        """
        Computes the lower tail dependence coefficient for the Gaussian copula.

        Gaussian copula has no tail dependence:
            LTDC = 0
        """
        return 0.0

    def UTDC(self, param):
        """
        Computes the upper tail dependence coefficient for the Gaussian copula.

        Gaussian copula has no tail dependence:
            UTDC = 0
        """
        return 0.0

    def IAD(self, data):
        """
        Skipped IAD computation due to high computational cost for elliptical copulas.
        """
        print(f"[INFO] IAD is disabled for {self.name} due to performance limitations.")
        return np.nan

    def AD(self, data):
        """
        Skipped AD computation due to high computational cost for elliptical copulas.
        """
        print(f"[INFO] AD is disabled for {self.name} due to performance limitations.")
        return np.nan

    def conditional_cdf_u_given_v(self, u, v, param):
        """
        Analytically computes the conditional CDF P(U ≤ u | V = v) for the Gaussian copula.

        Using the relationship for a bivariate normal,
            P(U ≤ u | V = v) = Φ((Φ⁻¹(u) - ρ Φ⁻¹(v)) / sqrt(1-ρ²))
        """

        rho = param[0]

        # Transform u and v via the inverse normal
        x = st.norm.ppf(u)
        y = st.norm.ppf(v)
        # Compute the conditional CDF
        cond = st.norm.cdf((x - rho * y) / np.sqrt(1 - rho ** 2))
        return cond

    def conditional_cdf_v_given_u(self, v, u, param):
        """
        Analytically computes the conditional CDF P(V ≤ v | U = u) for the Gaussian copula.

        Using the symmetric relationship:
            P(V ≤ v | U = u) = Φ((Φ⁻¹(v) - ρ Φ⁻¹(u)) / sqrt(1-ρ²))
        """

        rho = param[0]

        x = st.norm.ppf(u)
        y = st.norm.ppf(v)
        cond = st.norm.cdf((y - rho * x) / np.sqrt(1 - rho ** 2))
        return cond




