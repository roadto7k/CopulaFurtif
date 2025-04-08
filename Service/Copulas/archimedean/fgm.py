import numpy as np
from scipy.stats import uniform
from Service.Copulas.base import BaseCopula

class FGMCopula(BaseCopula):
    """
    Farlie-Gumbel-Morgenstern (FGM) Copula class.

    Attributes
    ----------
    type : str
        Identifier for the copula family. Here, 'fgm'.
    name : str
        Human-readable name.
    bounds_param : list of tuple
        Bounds for the copula parameter(s). For FGM, theta is in [-1, 1].
    parameters_start : np.ndarray
        Starting guess for the copula parameter(s).
    n_obs : int or None
        Number of observations used in the fit.
    default_optim_method : str
        Recommended optimizer for fitting.

    Methods
    -------
    get_cdf(u, v, param)
        Computes the CDF of the FGM copula at (u, v).
    get_pdf(u, v, param)
        Computes the PDF of the FGM copula at (u, v).
    kendall_tau(param)
        Computes Kendall's tau from the FGM parameter.
    sample(n, param)
        Generates n samples from the FGM copula.
    """

    def __init__(self):
        super().__init__()
        self.type = 'fgm'
        self.name = "FGM Copula"
        self.bounds_param = [(-1, 1 - 1e-6)]
        self.parameters = np.array(0)
        self.default_optim_method = "Powell"

    def get_cdf(self, u, v, param):
        """
        Computes the cumulative distribution function (CDF) of the FGM copula.

        Formula:
            C(u, v) = u * v * [1 + theta * (1 - u) * (1 - v)]

        Parameters
        ----------
        u : array_like
            First set of uniform marginals.
        v : array_like
            Second set of uniform marginals.
        param : array_like
            Copula parameter (theta).

        Returns
        -------
        np.ndarray
            The CDF evaluated at (u, v).
        """
        theta = param[0]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        return u * v * (1 + theta * (1 - u) * (1 - v))

    def get_pdf(self, u, v, param):
        """
        Computes the probability density function (PDF) of the FGM copula.

        Formula:
            c(u, v) = 1 + theta * (1 - 2u) * (1 - 2v)

        Parameters
        ----------
        u : array_like
            First set of uniform marginals.
        v : array_like
            Second set of uniform marginals.
        param : array_like
            Copula parameter (theta).

        Returns
        -------
        np.ndarray
            The PDF evaluated at (u, v).
        """
        theta = param[0]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        return 1 + theta * (1 - 2 * u) * (1 - 2 * v)

    def kendall_tau(self, param):
        """
        Computes Kendall's tau for the FGM copula.

        For FGM, Kendall's tau is given by:
            tau = 2 * theta / 9

        Parameters
        ----------
        param : array_like
            Copula parameter (theta).

        Returns
        -------
        float
            Kendall's tau.
        """
        theta = param[0]
        return 2 * theta / 9

    def sample(n, param):
        """
        Generates n samples from the Farlie-Gumbel-Morgenstern (FGM) copula
        using conditional inversion.

        The conditional CDF of V given U = u is:
            F_{V|U}(v|u) = v + theta * (1 - 2u) * (v - v^2)

        To sample from this copula:
        - Generate u ~ Uniform(0,1)
        - For each u, generate w ~ Uniform(0,1)
        - Solve the quadratic equation for v:
            A * v^2 - (1 + A) * v + w = 0
          where A = theta * (1 - 2u)

        Parameters
        ----------
        n : int
            Number of samples to generate.
        param : array_like
            Copula parameter array, where param[0] = theta ∈ [-1, 1]

        Returns
        -------
        np.ndarray
            A (n, 2) array of samples from the FGM copula.
        """
        theta = param[0]
        eps = 1e-12  # numerical tolerance

        u = uniform.rvs(size=n)
        w = uniform.rvs(size=n)
        A = theta * (1 - 2 * u)

        # Initialize v with fallback (independent copula)
        v = np.copy(w)

        # Identify where A is not close to 0 (i.e., non-independent case)
        mask = np.abs(A) > eps
        A_masked = A[mask]
        w_masked = w[mask]

        # Quadratic coefficients: A * v^2 - (1 + A) * v + w = 0
        a = A_masked
        b = -(1 + A_masked)
        c = w_masked

        discriminant = b ** 2 - 4 * a * c

        # Handle cases where discriminant is positive
        valid = discriminant >= -1e-14  # tolerance to avoid numerical errors
        sqrt_D = np.sqrt(np.maximum(discriminant, 0))

        v1 = (-b - sqrt_D) / (2 * a)
        v2 = (-b + sqrt_D) / (2 * a)

        # Select valid roots in [0, 1]
        chosen_v = np.where((0 <= v1) & (v1 <= 1), v1,
                            np.where((0 <= v2) & (v2 <= 1), v2, w_masked))

        v[mask] = chosen_v
        return np.column_stack((u, v))

    def LTDC(self, param):
        """
        Computes the lower tail dependence coefficient for the fgm copula.

        Gumbel copula has no lower tail dependence:
            LTDC = 0
        """
        return 0.0

    def UTDC(self, param):
        """
        Computes the upper tail dependence coefficient for the fgm copula.

        Formula:
            0
        """
        return 0

    def conditional_cdf_u_given_v(self, u, v, param):
        """
        Computes the conditional CDF P(U ≤ u | V = v) for the FGM copula analytically.

        The FGM copula is defined as:
            C(u,v) = u*v + θ * u*v*(1-u)*(1-v),
        so that the derivative with respect to v is:
            ∂C(u,v)/∂v = u + θ * u*(1-u)*(1-2v).

        Since C(1,v)=v (and its derivative with respect to v is 1),
        we set:
            F_{U|V}(u | v) = u + θ * u*(1-u)*(1-2v)

        Parameters
        ----------
        u : float or array-like
            Value(s) in [0, 1] for U.
        v : float or array-like
            Value(s) in [0, 1] for V (the conditioning variable).
        param : list or array-like, optional
            Copula parameter(s) as [θ]. If None, self.parameters is used.

        Returns
        -------
        float or np.ndarray
            Conditional CDF P(U ≤ u | V = v).
        """

        theta = param[0]
        u = np.asarray(u)
        v = np.asarray(v)
        return u + theta * u * (1 - u) * (1 - 2 * v)

    def conditional_cdf_v_given_u(self, v, u, param):
        """
        Computes the conditional CDF P(V ≤ v | U = u) for the FGM copula analytically.

        By symmetry, for the FGM copula:
            C(u,v) = u*v + θ * u*v*(1-u)*(1-v),
        the derivative with respect to u is:
            ∂C(u,v)/∂u = v + θ * v*(1-v)*(1-2u).

        Therefore, we define:
            F_{V|U}(v | u) = v + θ * v*(1-v)*(1-2u)

        Parameters
        ----------
        v : float or array-like
            Value(s) in [0, 1] for V.
        u : float or array-like
            Value(s) in [0, 1] for U (the conditioning variable).
        param : list or array-like, optional
            Copula parameter(s) as [θ]. If None, self.parameters is used.

        Returns
        -------
        float or np.ndarray
            Conditional CDF P(V ≤ v | U = u).
        """

        theta = param[0]
        u = np.asarray(u)
        v = np.asarray(v)
        return v + theta * v * (1 - v) * (1 - 2 * u)
