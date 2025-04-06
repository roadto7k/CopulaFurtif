import numpy as np
import matplotlib.pyplot as plt

from Service.Copulas.fitting.estimation import pseudo_obs
from Service.Copulas.metrics.gof import compute_iad_score, AD_score, kendall_tau_distance, compute_aic, compute_bic


class BaseCopula:
    """
    Base class for all bivariate copulas.
    Defines the common API: get_pdf, get_cdf, sample, etc.
    """

    def __init__(self):

        self.type = None
        self.name = None
        self.parameters = None
        self.bounds_param = None
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


