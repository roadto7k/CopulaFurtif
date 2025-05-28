"""
Farlie-Gumbel-Morgenstern (FGM) Copula implementation.

The FGM copula is a simple bivariate copula that allows for weak dependence modeling.
It is limited in the range of dependence it can express, making it suitable mainly
for didactic or illustrative purposes.

Attributes:
    name (str): Human-readable name of the copula.
    type (str): Copula identifier.
    bounds_param (list of tuple): Bounds for the copula parameter [theta] ∈ [-1, 1].
    parameters (np.ndarray): Copula parameter [theta].
    default_optim_method (str): Default optimization method for parameter fitting.
"""

import numpy as np
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


class FGMCopula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """FGM Copula model."""

    def __init__(self):
        """Initialize the FGM copula with default parameters and bounds."""
        super().__init__()
        self.name = "FGM Copula"
        self.type = "fgm"
        self.bounds_param = [(-1.0, 1.0)]  # [theta]
        self.param_names = ["theta"]
        self.parameters = [0.3]
        self.default_optim_method = "SLSQP"

    def get_cdf(self, u, v, param=None):
        """Compute the copula CDF C(u, v).

        Args:
            u (float or np.ndarray): First input in (0, 1).
            v (float or np.ndarray): Second input in (0, 1).
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float or np.ndarray: CDF value(s).
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        return u * v * (1 + theta * (1 - u) * (1 - v))

    def get_pdf(self, u, v, param=None):
        """Compute the copula PDF c(u, v).

        Args:
            u (float or np.ndarray): First input in (0, 1).
            v (float or np.ndarray): Second input in (0, 1).
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float or np.ndarray: PDF value(s).
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        return 1 + theta * (1 - 2 * u) * (1 - 2 * v)

    def sample(self, n, param=None):
        """Generate random samples from the FGM copula (approximate).

        Note:
            The current implementation returns independent uniform samples (approximate).

        Args:
            n (int): Number of samples to generate.
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            np.ndarray: Samples of shape (n, 2).
        """
        if param is None:
            param = self.parameters
        u = np.random.rand(n)
        v = np.random.rand(n)
        return np.column_stack((u, v))  # NOTE: approximate sample, not exact FGM

    def kendall_tau(self, param=None):
        """Compute Kendall's tau for the FGM copula.

        Args:
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float: Kendall's tau.
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        return (2 * theta) / 9
