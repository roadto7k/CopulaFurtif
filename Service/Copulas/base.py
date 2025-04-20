import numpy as np

from Service.Copulas.metrics.gof import compute_iad_score, AD_score, kendall_tau_distance, compute_aic, compute_bic


class BaseCopula:
    """
    Base class for all bivariate copulas.
    Defines the common API: get_pdf, get_cdf, sample, etc.
    """

    def __init__(self):

        self.type = None
        self.name = None
        # self.parameters = None
        # self.bounds_param = None
        self.log_likelihood_ = None
        self.n_obs = None

    def get_pdf(self, u, v, params):
        """
        Computes the density of the copula at (u, v).
        This method should be overridden in the subclass.
        """
        raise NotImplementedError

    def get_cdf(self, u, v, params):
        """
        Computes the cumulative distribution function (CDF) of the copula at (u, v).
        This method should be overridden in the subclass.
        """
        raise NotImplementedError

    def kendall_tau(self, param):
        """
        Computes Kendall's tau for the given parameter.
        This method should be overridden in the subclass.
        """
        raise NotImplementedError

    def sample(self, n, params):
        """
        Generates n samples (u_i, v_i) from the copula with the given parameters.
        This method should be overridden in the subclass.
        """
        raise NotImplementedError

    def _check_is_fitted(self):
        """
        Internal check to ensure the copula has been fitted before calling evaluation methods.
        Raises a RuntimeError if `log_likelihood_` is None.
        """
        if self.log_likelihood_ is None:
            raise RuntimeError(f"The copula '{self.name}' must be fitted before calling this method.")

    def IAD(self, data):
        """
        Compute the Integrated Anderson-Darling (IAD) score for the copula.

        Parameters
        ----------
        data : array-like
            Pseudo-observations [u, v].

        Returns
        -------
        float
            IAD score (lower = better fit).
        """
        self._check_is_fitted()
        return compute_iad_score(self, data)

    def AD(self, data):
        """
        Compute the Anderson-Darling (AD) goodness-of-fit statistic.

        Parameters
        ----------
        data : array-like
            Pseudo-observations [u, v].

        Returns
        -------
        float
            AD score (lower = better fit, more sensitive to tail deviation).
        """
        self._check_is_fitted()
        return AD_score(self, data)

    def kendall_tau_error(self, data):
        """
        Compute the absolute error between empirical and theoretical Kendall's tau.

        Parameters
        ----------
        data : array-like
            Raw data [X, Y] used to compute the empirical tau.

        Returns
        -------
        float
            Absolute tau error |tau_model - tau_empirical|.
        """
        self._check_is_fitted()
        return kendall_tau_distance(self, data)

    def AIC(self):
        """
        Compute the Akaike Information Criterion for the fitted copula.

        Returns
        -------
        float
            AIC score (lower = better trade-off between fit and complexity).
        """
        self._check_is_fitted()
        return compute_aic(self)

    def BIC(self):
        """
        Compute the Bayesian Information Criterion for the fitted copula.

        Returns
        -------
        float
            BIC score (penalizes complexity more than AIC).

        Raises
        ------
        RuntimeError if the copula has not been fitted.
        """
        self._check_is_fitted()
        return compute_bic(self)

    def partial_derivative_C_wrt_v(self, u, v, param):
        """
        Analytically computes ∂C(u,v)/∂v.
        """

        raise NotImplementedError

    def partial_derivative_C_wrt_u(self, u, v, param):
        """
        Analytically computes ∂C(u,v)/∂u.
        """

        raise NotImplementedError

    def conditional_cdf_u_given_v(self, u, v, param=None):
        """
        Analytically computes the conditional CDF P(U ≤ u | V = v).
        """

        raise NotImplementedError

    def conditional_cdf_v_given_u(self, v, u, param=None):
        """
        Analytically computes the conditional CDF P(V ≤ v | U = u).
        """

        raise NotImplementedError

    def residual_heatmap(self, u, v, bins=50, cmap="coolwarm", figsize=(6, 5), show=True):
        """
        Plot a residual heatmap: Empirical CDF - Copula Model CDF.

        Parameters
        ----------
        u : array-like
            First marginal (must be ∈ (0, 1)).
        v : array-like
            Second marginal (must be ∈ (0, 1)).
        bins : int
            Number of bins (grid resolution) for the 2D histogram.
        cmap : str
            Colormap used for the heatmap.
        figsize : tuple
            Size of the matplotlib figure.
        show : bool
            Whether to display the plot immediately.

        Returns
        -------
        np.ndarray
            2D array of residuals (empirical CDF - model CDF).
        """
        import matplotlib.pyplot as plt

        assert self.parameters is not None, "Copula must be fitted first (missing parameters)."

        u = np.asarray(u).flatten()
        v = np.asarray(v).flatten()
        assert u.ndim == 1 and v.ndim == 1, "Inputs u and v must be 1D arrays."

        n = len(u)
        grid = np.linspace(0, 1, bins)
        U, V = np.meshgrid(grid, grid, indexing="ij")

        # Empirical copula values
        emp_copula = np.array([
            np.mean((u <= U[i, j]) & (v <= V[i, j]))
            for i in range(bins) for j in range(bins)
        ]).reshape(bins, bins)

        # Model copula values
        model_copula = self.get_cdf(U.ravel(), V.ravel(), self.parameters).reshape(bins, bins)

        # Residuals = Empirical - Model
        residuals = emp_copula - model_copula

        if show:
            plt.figure(figsize=figsize)
            plt.imshow(residuals, origin="lower", cmap=cmap, extent=[0, 1, 0, 1], aspect="auto")
            plt.colorbar(label="Empirical - Model CDF")
            plt.title(f"Residual Heatmap: {self.name}")
            plt.xlabel("u")
            plt.ylabel("v")
            plt.tight_layout()
            plt.show()

        return residuals

    def check_axioms(self, atol=1e-6, verbose=True):
        """
        Check that the copula satisfies the basic Sklar axioms numerically.

        Parameters
        ----------
        atol : float
            Absolute tolerance allowed for numerical differences.
        verbose : bool
            If True, print which axioms pass or fail.

        Returns
        -------
        dict
            Dictionary of boolean results for each axiom.
        """
        # Choix de points de test
        u_vals = np.linspace(0.01, 0.99, 20)
        v_vals = np.linspace(0.01, 0.99, 20)

        p = self.parameters
        results = {}

        # A1: C(u, 0) == 0
        results["C(u,0)=0"] = np.allclose([self.get_cdf(u, 0.0, p) for u in u_vals], 0.0, atol=atol)

        # A2: C(0, v) == 0
        results["C(0,v)=0"] = np.allclose([self.get_cdf(0.0, v, p) for v in v_vals], 0.0, atol=atol)

        # A3: C(u,1) == u
        results["C(u,1)=u"] = np.allclose(
            [self.get_cdf(u, 1.0, p) for u in u_vals], u_vals, atol=atol
        )

        # A4: C(1,v) == v
        results["C(1,v)=v"] = np.allclose(
            [self.get_cdf(1.0, v, p) for v in v_vals], v_vals, atol=atol
        )

        # A5: C(u,v) is increasing in u and v (discrete test on grid)
        grid = np.linspace(0.01, 0.99, 10)
        increasing = True
        for i in range(len(grid) - 1):
            for j in range(len(grid) - 1):
                u1, u2 = grid[i], grid[i + 1]
                v1, v2 = grid[j], grid[j + 1]
                c1 = self.get_cdf(u1, v1, p)
                c2 = self.get_cdf(u2, v2, p)
                if c2 < c1 - atol:
                    increasing = False
                    break
        results["C is increasing"] = increasing

        if verbose:
            for k, v in results.items():
                print(f"{k}: {'Good' if v else 'Wrong'}")

        return results


