"""
Clayton Copula implementation.

The Clayton copula is a popular Archimedean copula used to model asymmetric 
dependence, especially in the lower tail. It is parameterized by a single positive 
parameter theta > 0, which controls the strength of dependence.

Attributes:
    name (str): Human-readable name of the copula.
    type (str): Identifier for the copula family.
    bounds_param (list of tuple): Bounds for the copula parameter [theta].
    parameters (np.ndarray): Copula parameter [theta].
    default_optim_method (str): Default optimization method for fitting.
"""

import numpy as np
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel, CopulaParameters
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence
from numpy.random import default_rng


class ClaytonCopula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """Clayton copula model."""

    def __init__(self):
        """Initialize the Clayton copula with default parameters and bounds."""
        super().__init__()
        self.name = "Clayton Copula"
        self.type = "clayton"
        # self.bounds_param = [(0.01, 30.0)]  # [theta]
        # self.param_names = ["theta"]
        # self.parameters = [2.0]
        self.default_optim_method = "SLSQP"
        self.init_parameters(CopulaParameters([2.0],[(0.01, 30.0)] , ["theta"] ))

    def get_cdf(self, u, v, param=None):
        """Compute the copula CDF C(u, v).

        Args:
            u (float or np.ndarray): First input in (0,1).
            v (float or np.ndarray): Second input in (0,1).
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float or np.ndarray: Value(s) of the CDF.
        """
        if param is None:
            param = self.get_parameters()
        theta = param[0]
        return np.maximum((u ** -theta + v ** -theta - 1) ** (-1 / theta), 0.0)

    def get_pdf(self, u, v, param=None):
        """Compute the copula PDF c(u, v).

        Args:
            u (float or np.ndarray): First input in (0,1).
            v (float or np.ndarray): Second input in (0,1).
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float or np.ndarray: Value(s) of the PDF.
        """
        if param is None:
            param = self.get_parameters()
        theta = param[0]
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)

        # --- keep a hair away from the boundaries
        eps = 1e-15
        u_safe = np.clip(u, eps, 1.0 - eps)
        v_safe = np.clip(v, eps, 1.0 - eps)

        # --- log-form to prevent overflow / 0**neg → inf
        log_num = np.log(theta + 1) - (theta + 1) * (np.log(u_safe) + np.log(v_safe))
        log_denom = (2 + 1 / theta) * np.log(u_safe ** (-theta) + v_safe ** (-theta) - 1.0)
        pdf = np.exp(log_num - log_denom)

        # exact boundaries → density tends to 0
        mask = (u <= 0) | (v <= 0) | (u >= 1) | (v >= 1)
        if np.any(mask):
            pdf = np.where(mask, 0.0, pdf)

        return pdf

    def sample(self, n, param=None, rng=None):
        """
        Draw `n` i.i.d. samples from a 2-D Clayton copula.

        Parameters
        ----------
        n : int
            Size of the sample.
        param : array-like, optional
            `[theta]`. If None, uses current parameters.
        rng : np.random.Generator, optional
            Reproducible random generator.

        Returns
        -------
        ndarray, shape (n, 2)
            Pseudo-observations (U, V) in (0, 1)².
        """
        if rng is None:
            rng = default_rng()

        # ---- parameter -------------------------------------------------
        if param is None:
            theta = float(self.get_parameters()[0])
        else:
            theta = float(param[0])

        # independence limit
        if abs(theta) < 1e-8:
            return rng.random((n, 2))

        if theta <= 0:
            raise ValueError("Clayton sampler requires theta > 0.")

        k = 1.0 / theta                       # Gamma shape parameter

        # 1) shared mixing variable S ~ Gamma(k, 1)
        S = rng.gamma(shape=k, scale=1.0, size=n)

        # 2) independent exponentials
        E1 = rng.exponential(scale=1.0, size=n)
        E2 = rng.exponential(scale=1.0, size=n)

        # 3) transform
        U = (1.0 + E1 / S) ** (-1.0 / theta)
        V = (1.0 + E2 / S) ** (-1.0 / theta)

        # tiny clipping for numerical safety (optional)
        eps = 1e-15
        np.clip(U, eps, 1.0 - eps, out=U)
        np.clip(V, eps, 1.0 - eps, out=V)

        return np.column_stack((U, V))

    def kendall_tau(self, param=None):
        """Compute Kendall's tau for the Clayton copula.

        Args:
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float: Kendall's tau.
        """
        if param is None:
            param = self.get_parameters()
        theta = param[0]
        return theta / (theta + 2)

    def LTDC(self, param=None):
        """Lower tail dependence coefficient for Clayton copula.

        Args:
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float: LTDC value.
        """
        if param is None:
            param = self.get_parameters()
        theta = param[0]
        return 2 ** (-1 / theta)

    def UTDC(self, param=None):
        """Upper tail dependence coefficient (always 0 for Clayton copula).

        Args:
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float: 0.0
        """
        return 0.0

    def IAD(self, data):
        """Integrated Absolute Deviation (disabled for Clayton copula).

        Args:
            data (array-like): Input data (unused).

        Returns:
            float: NaN
        """
        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        """Anderson–Darling test statistic (disabled for Clayton copula).

        Args:
            data (array-like): Input data (unused).

        Returns:
            float: NaN
        """
        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """Compute ∂C(u, v)/∂u.

        Args:
            u (float or np.ndarray): First input in (0,1).
            v (float or np.ndarray): Second input in (0,1).
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float or np.ndarray: Partial derivative values.
        """
        if param is None:
            param = self.get_parameters()
        theta = param[0]
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        eps = 1e-15
        u_safe = np.clip(u, eps, 1.0 - eps)
        v_safe = np.clip(v, eps, 1.0 - eps)

        log_top = (-1 / theta - 1) * np.log(u_safe ** (-theta) + v_safe ** (-theta) - 1)
        log_top += (-theta - 1) * np.log(u_safe)
        return np.exp(log_top)

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """Compute ∂C(u, v)/∂v via symmetry.

        Args:
            u (float or np.ndarray): First input in (0,1).
            v (float or np.ndarray): Second input in (0,1).
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float or np.ndarray: Partial derivative values.
        """
        return self.partial_derivative_C_wrt_u(v, u, param)
