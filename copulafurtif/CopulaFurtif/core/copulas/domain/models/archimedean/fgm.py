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
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel, CopulaParameters
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


class FGMCopula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """FGM Copula model."""

    def __init__(self):
        """Initialize the FGM copula with default parameters and bounds."""
        super().__init__()
        self.name = "FGM Copula"
        self.type = "fgm"
        self.default_optim_method = "SLSQP"
        self.init_parameters(CopulaParameters([0.3],  [(-1.0, 1.0)], ["theta"]))
        
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
            param = self.get_parameters()
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
            param = self.get_parameters()
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
            param = self.get_parameters()
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
            param = self.get_parameters()
        theta = param[0]
        return (2 * theta) / 9

    def LTDC(self, param: np.ndarray = None) -> float:
        """
        Compute the lower tail dependence coefficient (LTDC) for the FGM copula.

        Args:
            param (np.ndarray, optional): Copula parameters [theta]. Defaults to self.parameters.

        Returns:
            float: Lower tail dependence (0.0 for FGM).
        """

        return 0.0

    def UTDC(self, param: np.ndarray = None) -> float:
        """
        Compute the upper tail dependence coefficient (UTDC) for the FGM copula.

        Args:
            param (np.ndarray, optional): Copula parameters [theta]. Defaults to self.parameters.

        Returns:
            float: Upper tail dependence (0.0 for FGM).
        """

        return 0.0

    def partial_derivative_C_wrt_v(self, u, v, param: np.ndarray = None):
        """
        Compute the partial derivative ∂C(u,v)/∂v of the FGM copula CDF.

        Args:
            u (float or np.ndarray): First margin in (0,1).
            v (float or np.ndarray): Second margin in (0,1).
            param (np.ndarray, optional): Copula parameters [theta]. Defaults to self.parameters.

        Returns:
            float or np.ndarray: Value of ∂C/∂v = u + θ·u·(1−u)·(1−2v).
        """

        if param is None:
            param = self.get_parameters()
        theta = param[0]
        u = np.asarray(u)
        v = np.asarray(v)
        return u + theta * u * (1.0 - u) * (1.0 - 2.0*v)

    def partial_derivative_C_wrt_u(self, u, v, param: np.ndarray = None):
        """
        Compute the partial derivative ∂C(u,v)/∂u of the FGM copula CDF.

        Args:
            u (float or np.ndarray): First margin in (0,1).
            v (float or np.ndarray): Second margin in (0,1).
            param (np.ndarray, optional): Copula parameters [theta]. Defaults to self.parameters.

        Returns:
            float or np.ndarray: Value of ∂C/∂u = v + θ·v·(1−v)·(1−2u).
        """

        return self.partial_derivative_C_wrt_u(v, u, param)