import numpy as np
from scipy.special import gammaln
from scipy.stats import kendalltau
from scipy.stats import multivariate_t
from scipy.stats import t

from Service.Copulas.base import BaseCopula


class StudentCopula(BaseCopula):
    """
    Student-t Copula class

    Attributes
    ----------
    family : str
        Identifier for the copula family. Here, "student".
    name : str
        Human-readable name for output/logging.
    bounds_param : list of tuple
        Bounds for the copula parameters: correlation and degrees of freedom.
    parameters : np.ndarray
        Initial guess for the copula parameters [rho, df].
    n_obs : int
        Number of observations used during fitting (optional).

    Methods
    -------
    get_cdf(u, v, param)
        Computes the CDF of the Student copula at (u, v).
    get_pdf(u, v, param)
        Computes the PDF of the Student copula at (u, v).
    sample(n, param)
        Generates n samples in [0,1]^2 from the Student copula.
    kendall_tau(param)
        Theoretical Kendall's tau for given parameters.
    """

    def __init__(self):
        super().__init__()
        self.type = "student"
        self.name = "Student-t Copula"
        self.bounds_param = [(-0.999, 0.999), (1e-6, 50.0)]
        self.parameters = np.array([0.5, 4.0])  # [rho, df]
        self.default_optim_method = "Powell"  # or "trust-constr"

    def get_cdf(self, u, v, params):
        """
        Robust CDF of the bivariate Student-t copula.

        Parameters
        ----------
        u : float or array-like
            First pseudo-observation(s) in [0,1].
        v : float or array-like
            Second pseudo-observation(s) in [0,1].
        params : list or array-like
            [rho, nu] where rho is the correlation and nu the degrees of freedom.

        Returns
        -------
        float or np.ndarray
            The copula CDF evaluated at (u, v).
        """
        rho, nu = params

        # Ensure u and v are numpy arrays
        u_arr = np.asarray(u)
        v_arr = np.asarray(v)

        # Force inputs to be at least 1D for broadcasting
        u_vec = np.atleast_1d(u_arr)
        v_vec = np.atleast_1d(v_arr)

        # Handle boundary cases: when u or v = 0 or 1
        mask_zero = (u_vec <= 0) | (v_vec <= 0)
        mask_u_one = (u_vec >= 1)
        mask_v_one = (v_vec >= 1)

        # Clip inputs to avoid -inf/+inf from t.ppf
        eps = 1e-12
        u_clipped = np.clip(u_vec, eps, 1 - eps)
        v_clipped = np.clip(v_vec, eps, 1 - eps)

        # Inverse transform: convert to t quantiles
        t1 = t.ppf(u_clipped, df=nu)
        t2 = t.ppf(v_clipped, df=nu)

        # Stack data for multivariate t CDF evaluation
        data = np.column_stack([t1, t2])
        cov = np.array([[1, rho], [rho, 1]])
        mv_t = multivariate_t(loc=[0, 0], shape=cov, df=nu)

        # Force the result to be at least 1D
        result = np.atleast_1d(np.asarray(mv_t.cdf(data)))

        # Enforce boundary values
        result[mask_zero] = 0.0
        result[mask_u_one] = v_vec[mask_u_one]
        result[mask_v_one] = u_vec[mask_v_one]

        # Return scalar if original input was scalar
        if np.isscalar(u) and np.isscalar(v):
            return result.item()
        return result

    def get_pdf(self, u, v, param):
        """
        Computes the PDF of the bivariate Student-t copula using the numerically stable
        log-domain formulation.

        This implementation is based on:
        Joe, H. (2014). *Dependence Modeling with Copulas*, CRC Press.
        See Equation (4.32), p.181.

        Instead of using the standard gamma function (which can overflow/underflow for large or
        small degrees of freedom), this version uses `scipy.special.gammaln`, which computes the
        natural logarithm of the gamma function:
            gammaln(x) = log(Gamma(x))

        By applying all operations (products, powers, divisions) in the logarithmic domain, we
        improve numerical stability. The final result is then exponentiated to return the true
        PDF value.

        Parameters
        ----------
        u, v : float or np.ndarray
            Marginal CDF values (pseudo-observations) in [0, 1].

        param : list or tuple of length 2
            Parameters of the Student copula:
            - rho ∈ [-1, 1] : the linear correlation coefficient
            - nu > 0 : the degrees of freedom of the Student-t distribution

        Returns
        -------
        float or np.ndarray
            The copula density value(s) c(u, v) ∈ ℝ⁺.
        """
        rho, nu = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        u_ = t.ppf(u, df=nu)
        v_ = t.ppf(v, df=nu)

        det_rho = 1 - rho ** 2
        quad_form = (u_ ** 2 - 2 * rho * u_ * v_ + v_ ** 2) / det_rho

        # Log of gamma terms
        log_num = gammaln((nu + 2) / 2) + gammaln(nu / 2)
        log_den = 2 * gammaln((nu + 1) / 2)

        # Log of determinant and products
        log_det = 0.5 * np.log(det_rho)
        log_prod_uv = ((nu + 1) / 2) * (np.log1p((u_ ** 2) / nu) + np.log1p((v_ ** 2) / nu))
        log_den_term = ((nu + 2) / 2) * np.log1p(quad_form / nu)

        # Final log copula density
        log_copula = log_num - log_den - log_det + log_prod_uv - log_den_term

        return np.exp(log_copula)

    def sample(self, n, param):
        """
        Generate random samples from the Student-t copula.

        Parameters
        ----------
        n : int
            Number of samples to generate.
        param : list or array
            Copula parameters [rho, nu].

        Returns
        -------
        np.ndarray
            n x 2 array of pseudo-observations (u, v).
        """
        rho, nu = param
        cov = np.array([[1, rho], [rho, 1]])
        L = np.linalg.cholesky(cov)

        z = np.random.standard_normal((n, 2))
        chi2 = np.random.chisquare(df=nu, size=n)
        scaled = z @ L.T / np.sqrt(chi2 / nu)[:, None]

        u = t.cdf(scaled[:, 0], df=nu)
        v = t.cdf(scaled[:, 1], df=nu)
        return np.column_stack((u, v))

    def kendall_tau(self, param, n_samples=10000, random_state=None):
        """
        Estimate Kendall's tau for the bivariate Student-t copula using Monte Carlo simulation.

        The Student-t copula with parameters [rho, nu] (correlation and degrees of freedom)
        does not have a closed-form expression for Kendall's tau when nu is finite.
        To obtain an estimate, we generate n_samples pseudo-observations from the copula:

          1. We sample from a bivariate Student-t distribution with correlation matrix
             [[1, rho], [rho, 1]] and degrees of freedom nu.
          2. We transform the marginals using the univariate Student-t CDF to get uniform
             pseudo-observations in [0, 1].
          3. Finally, we compute the empirical Kendall's tau from the generated samples.

        This Monte Carlo method provides an approximation to the theoretical Kendall's tau,
        which is especially useful when nu is small (and the copula exhibits tail dependence).

        Parameters
        ----------
        param : list or tuple
            Copula parameters [rho, nu] where:
              - rho ∈ [-1, 1] is the linear correlation coefficient.
              - nu > 0 is the degrees of freedom of the Student-t distribution.
        n_samples : int, optional
            Number of samples to generate for the simulation (default is 10,000).
        random_state : int or np.random.RandomState, optional
            Seed or random state for reproducibility.

        Returns
        -------
        float
            The estimated Kendall's tau for the Student-t copula.
        """
        rho, nu = param
        # Setup random state for reproducibility
        rng = np.random.RandomState(random_state) if random_state is not None else np.random

        # Create the correlation matrix and its Cholesky decomposition
        cov = np.array([[1, rho], [rho, 1]])
        L = np.linalg.cholesky(cov)

        # Generate samples from the multivariate t-distribution
        z = rng.standard_normal((n_samples, 2))
        chi2 = rng.chisquare(df=nu, size=n_samples)
        # Scale to obtain t-distributed samples
        scaled = (z @ L.T) / np.sqrt(chi2 / nu)[:, None]

        # Transform each margin with the Student-t CDF to obtain pseudo-observations in [0,1]
        u = t.cdf(scaled[:, 0], df=nu)
        v = t.cdf(scaled[:, 1], df=nu)
        samples = np.column_stack((u, v))

        # Compute Kendall's tau on the generated samples
        tau, _ = kendalltau(samples[:, 0], samples[:, 1])
        return tau

    def LTDC(self, param):
        """
        Computes the lower tail dependence coefficient for the Student copula.

        For a bivariate Student-t copula with correlation ρ and degrees of freedom ν,
        the tail dependence coefficient is:

            LTDC = 2 * T_{ν+1} ( -sqrt((ν+1)*(1-ρ)/(1+ρ)) )

        where T_{ν+1} is the CDF of a Student-t distribution with (ν+1) degrees of freedom.
        """
        rho = param[0]
        nu = param[1]
        return 2 * t.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), df=nu + 1)

    def UTDC(self, param):
        """
        Computes the upper tail dependence coefficient for the Student copula.

        For the bivariate Student-t copula, the upper tail dependence is equal to the lower tail dependence:

            UTDC = 2 * T_{ν+1} ( -sqrt((ν+1)*(1-ρ)/(1+ρ)) )
        """
        # They are identical for the Student copula.
        return self.LTDC(param)

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
        Compute the conditional CDF P(U ≤ u | V = v) for the Student copula.

        Parameters
        ----------
        u : float or array-like
            Value(s) in (0,1) corresponding to U = t_ν(x), i.e. the univariate Student CDF.
        v : float or array-like
            Value(s) in (0,1) corresponding to V = t_ν(y).
        param : list or array-like, optional
            The copula parameters [ρ, ν]. If None, self.parameters is used.

        Returns
        -------
        float or np.ndarray
            Conditional probability P(U ≤ u | V = v).
        """

        rho = param[0]
        nu = param[1]

        # Inverse CDF (quantile) for Student-t with ν degrees of freedom
        x = t.ppf(u, nu)
        y = t.ppf(v, nu)

        # For a bivariate Student-t, the conditional distribution X|Y=y is t-distributed
        # with ν+1 degrees of freedom, mean = ρ*y, and scale = sqrt(((ν+y²)(1-ρ²))/(ν+1))
        scale = np.sqrt(((nu + y ** 2) * (1 - rho ** 2)) / (nu + 1))

        # Compute the conditional CDF using the Student-t CDF with ν+1 degrees of freedom
        cond_cdf = t.cdf((x - rho * y) / scale, df=nu + 1)
        return cond_cdf

    def conditional_cdf_v_given_u(self, v, u, param):
        """
        Compute the conditional CDF P(V ≤ v | U = u) for the Student copula.

        Parameters
        ----------
        v : float or array-like
            Value(s) in (0,1) corresponding to V = t_ν(y).
        u : float or array-like
            Value(s) in (0,1) corresponding to U = t_ν(x).
        param : list or array-like, optional
            The copula parameters [ρ, ν]. If None, self.parameters is used.

        Returns
        -------
        float or np.ndarray
            Conditional probability P(V ≤ v | U = u).
        """

        rho = param[0]
        nu = param[1]

        x = t.ppf(u, nu)
        y = t.ppf(v, nu)

        # Ici, la distribution conditionnelle Y|X=x est t-distribuée avec ν+1 degrés de liberté,
        # mean = ρ*x et scale = sqrt(((ν+x²)(1-ρ²))/(ν+1))
        scale = np.sqrt(((nu + x ** 2) * (1 - rho ** 2)) / (nu + 1))

        cond_cdf = t.cdf((y - rho * x) / scale, df=nu + 1)
        return cond_cdf


