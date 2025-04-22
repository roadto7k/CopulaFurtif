"""
CopulaFurtif - Abstract Base Class for Copula Models

This module defines the abstract base class `CopulaModel`, which provides the
interface for all bivariate copula models within the CopulaFurtif framework.

All custom copula implementations should inherit from this class and provide 
concrete implementations of the abstract methods.

Attributes:
    _parameters (list or np.ndarray): Internal parameters of the copula.
    bounds_param (list of tuple): Parameter bounds for optimization.
    log_likelihood_ (float): Log-likelihood of the fitted model.
    n_obs (int): Number of observations used in fitting.

Abstract Methods:
    get_cdf(u, v, param): Compute the cumulative distribution function.
    get_pdf(u, v, param): Compute the probability density function.
    kendall_tau(param): Compute Kendall's tau from the model.
    sample(n, param): Generate random samples from the copula.
"""

from abc import ABC, abstractmethod
import numpy as np


class CopulaModel(ABC):
    """
    Abstract base class for all bivariate copula models.

    This class defines the required interface for any copula implementation 
    used in the CopulaFurtif pipeline. Each concrete subclass must implement 
    the CDF, PDF, Kendall's tau, and sampling methods.

    Attributes:
        _parameters (list or np.ndarray): Internal copula parameters.
        bounds_param (list of tuple): Bounds for parameter optimization.
        log_likelihood_ (float): Log-likelihood from the most recent fit.
        n_obs (int): Number of observations in the dataset.
    """

    def __init__(self):
        """
        Initialize the copula model with default attributes.
        """
        self._parameters = None
        self.bounds_param = None
        self.log_likelihood_ = None
        self.n_obs = None

    @abstractmethod
    def get_cdf(self, u, v, param=None):
        """
        Compute the cumulative distribution function C(u, v).

        Args:
            u (float or np.ndarray): First marginal (in [0, 1]).
            v (float or np.ndarray): Second marginal (in [0, 1]).
            param (list or np.ndarray, optional): Copula parameters. Defaults to current parameters.

        Returns:
            float or np.ndarray: Value(s) of the CDF.
        """
        pass

    @abstractmethod
    def get_pdf(self, u, v, param=None):
        """
        Compute the probability density function c(u, v).

        Args:
            u (float or np.ndarray): First marginal (in [0, 1]).
            v (float or np.ndarray): Second marginal (in [0, 1]).
            param (list or np.ndarray, optional): Copula parameters. Defaults to current parameters.

        Returns:
            float or np.ndarray: Value(s) of the PDF.
        """
        pass

    @abstractmethod
    def kendall_tau(self, param=None):
        """
        Compute the analytical Kendall's tau implied by the copula.

        Args:
            param (list or np.ndarray, optional): Copula parameters. Defaults to current parameters.

        Returns:
            float: Kendall's tau coefficient.
        """
        pass

    @abstractmethod
    def sample(self, n, param=None):
        """
        Generate n random samples from the copula.

        Args:
            n (int): Number of samples to generate.
            param (list or np.ndarray, optional): Copula parameters. Defaults to current parameters.

        Returns:
            np.ndarray: Array of shape (n, 2) with samples from the copula.
        """
        pass