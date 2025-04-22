"""
CopulaFurtif - Advanced Copula Interfaces and Metrics

This module provides mixins and base classes that extend copula models with 
additional capabilities such as model selection criteria (AIC, BIC), 
Kendall's tau error computation, and tail/conditional dependence logic.

Classes:
    SupportsTailDependence: Abstract base class for tail dependence metrics.
    ModelSelectionMixin: Adds model selection methods (AIC, BIC) and error metrics.
    AdvancedCopulaFeatures: Provides conditional distributions and derivative-based methods.

Functions (imported):
    compute_aic: Compute the Akaike Information Criterion for a copula.
    compute_bic: Compute the Bayesian Information Criterion for a copula.
    kendall_tau_distance: Compute the error between empirical and model Kendall's tau.
"""

from CopulaFurtif.core.copulas.domain.estimation.gof import compute_aic, compute_bic, kendall_tau_distance
import numpy as np


class SupportsTailDependence:
    """
    Interface for copulas that support tail dependence measures.

    Methods:
        LTDC(param): Lower tail dependence coefficient.
        UTDC(param): Upper tail dependence coefficient.
    """

    def LTDC(self, param):
        """
        Compute the lower tail dependence coefficient.

        Args:
            param (list or np.ndarray): Copula parameters.

        Raises:
            NotImplementedError: Must be implemented in subclass.
        """
        raise NotImplementedError

    def UTDC(self, param):
        """
        Compute the upper tail dependence coefficient.

        Args:
            param (list or np.ndarray): Copula parameters.

        Raises:
            NotImplementedError: Must be implemented in subclass.
        """
        raise NotImplementedError


class ModelSelectionMixin:
    """
    Mixin for adding model selection and evaluation methods to a copula.

    Methods:
        AIC(): Compute the Akaike Information Criterion.
        BIC(): Compute the Bayesian Information Criterion.
        kendall_tau_error(data): Compute the Kendall's tau distance error.
    """

    def AIC(self):
        """
        Compute the Akaike Information Criterion (AIC).

        Returns:
            float: The AIC score.

        Raises:
            RuntimeError: If the copula has not been fitted (log-likelihood is None).
        """
        if self.log_likelihood_ is None:
            raise RuntimeError("Copula must be fitted before computing AIC.")
        return compute_aic(self)

    def BIC(self):
        """
        Compute the Bayesian Information Criterion (BIC).

        Returns:
            float: The BIC score.

        Raises:
            RuntimeError: If log-likelihood or number of observations is missing.
        """
        if self.log_likelihood_ is None or self.n_obs is None:
            raise RuntimeError("Missing log-likelihood or n_obs.")
        return compute_bic(self)

    def kendall_tau_error(self, data):
        """
        Compute the Kendall's tau distance between empirical and model estimates.

        Args:
            data (array-like): Bivariate input data, shape (n_samples, 2).

        Returns:
            float: Distance/error value.
        """
        return kendall_tau_distance(self, data)


class AdvancedCopulaFeatures:
    """
    Adds support for partial derivatives and conditional distributions.

    Methods:
        partial_derivative_C_wrt_u(u, v, param): ∂C/∂u of the copula C(u,v).
        partial_derivative_C_wrt_v(u, v, param): ∂C/∂v of the copula C(u,v).
        conditional_cdf_u_given_v(u, v, param): CDF of U given V.
        conditional_cdf_v_given_u(u, v, param): CDF of V given U.
        IAD(data): Placeholder for Integrated Absolute Distance (returns NaN).
        AD(data): Placeholder for Anderson-Darling metric (returns NaN).
    """

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """
        Compute the partial derivative ∂C/∂u.

        Args:
            u (float or array-like): Value(s) of u.
            v (float or array-like): Value(s) of v.
            param (list or np.ndarray, optional): Copula parameters.

        Raises:
            NotImplementedError: Must be implemented in subclass.
        """
        raise NotImplementedError

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """
        Compute the partial derivative ∂C/∂v.

        Args:
            u (float or array-like): Value(s) of u.
            v (float or array-like): Value(s) of v.
            param (list or np.ndarray, optional): Copula parameters.

        Raises:
            NotImplementedError: Must be implemented in subclass.
        """
        raise NotImplementedError

    def conditional_cdf_u_given_v(self, u, v, param=None):
        """
        Compute the conditional CDF of U given V.

        Args:
            u (float or array-like): Value(s) of u.
            v (float or array-like): Value(s) of v.
            param (list or np.ndarray, optional): Copula parameters.

        Returns:
            float or np.ndarray: Conditional CDF values.
        """
        return self.partial_derivative_C_wrt_v(u, v, param)

    def conditional_cdf_v_given_u(self, u, v, param=None):
        """
        Compute the conditional CDF of V given U.

        Args:
            u (float or array-like): Value(s) of u.
            v (float or array-like): Value(s) of v.
            param (list or np.ndarray, optional): Copula parameters.

        Returns:
            float or np.ndarray: Conditional CDF values.
        """
        return self.partial_derivative_C_wrt_u(u, v, param)

    def IAD(self, data):
        """
        Placeholder for Integrated Absolute Distance (IAD).

        Args:
            data (array-like): Input dataset.

        Returns:
            float: NaN, indicating the metric is not implemented.
        """
        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        """
        Placeholder for Anderson-Darling goodness-of-fit test.

        Args:
            data (array-like): Input dataset.

        Returns:
            float: NaN, indicating the metric is not implemented.
        """
        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan
