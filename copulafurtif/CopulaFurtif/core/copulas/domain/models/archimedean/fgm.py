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
        self.init_parameters(CopulaParameters(np.array([0.3]),  [(-1.0, 1.0)], ["theta"]))
        
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

        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

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

        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        theta = param[0]
        return 1 + theta * (1 - 2 * u) * (1 - 2 * v)

    def sample(self, n: int, param=None, rng=None):
        """Generate random samples from the FGM copula (exact, 2D) via conditional inversion.

        Copula: C(u,v) = u v [1 + theta (1-u)(1-v)], theta in [-1, 1].

        Sampling:
            U ~ Unif(0,1)
            W ~ Unif(0,1)
            Solve W = v + a v(1-v) with a = theta (1 - 2U).

        Args:
            n (int): number of samples
            param (np.ndarray, optional): [theta]
            rng (np.random.Generator, optional): RNG

        Returns:
            np.ndarray: shape (n,2)
        """

        if param is None:
            param = self.get_parameters()
        theta = float(np.asarray(param, dtype=float).ravel()[0])

        # FGM parameter domain
        if theta < -1.0 or theta > 1.0:
            raise ValueError(f"FGM theta must be in [-1,1], got {theta}")

        if rng is None:
            rng = np.random.default_rng()

        u = rng.random(int(n))
        w = rng.random(int(n))

        a = theta * (1.0 - 2.0 * u)  # a in [-|theta|, |theta|]
        eps = 1e-12

        v = np.empty_like(u)

        # If a ~ 0 => conditional CDF is ~ v, so v = w
        mask0 = np.abs(a) < 1e-12
        v[mask0] = w[mask0]

        # Solve quadratic for others
        am = a[~mask0]
        wm = w[~mask0]

        # Discriminant: (1+a)^2 - 4 a w
        disc = (1.0 + am) ** 2 - 4.0 * am * wm
        disc = np.maximum(disc, 0.0)
        sqrt_disc = np.sqrt(disc)

        # Two roots
        r1 = ((1.0 + am) - sqrt_disc) / (2.0 * am)
        r2 = ((1.0 + am) + sqrt_disc) / (2.0 * am)

        # Pick the root in [0,1]
        # (there will be exactly one valid root for admissible parameters)
        in1 = (r1 >= -eps) & (r1 <= 1.0 + eps)
        vm = np.where(in1, r1, r2)

        # Clip to open interval for numerical safety
        vm = np.clip(vm, eps, 1.0 - eps)
        v[~mask0] = vm

        u = np.clip(u, eps, 1.0 - eps)
        return np.column_stack((u, v))

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

        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

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

        return self.partial_derivative_C_wrt_v(v, u, param=param)

    def blomqvist_beta(self, param=None) -> float:
        if param is None:
            param = self.get_parameters()
        theta = float(np.asarray(param, dtype=float).ravel()[0])
        return theta / 4.0

    def init_from_data(self, u, v):
        u = np.asarray(u, dtype=float).ravel()
        v = np.asarray(v, dtype=float).ravel()
        mask = np.isfinite(u) & np.isfinite(v)
        u = u[mask];
        v = v[mask]
        if u.size < 10:
            return self.get_parameters()

        # open interval safety
        eps = 1e-12
        u = np.clip(u, eps, 1.0 - eps)
        v = np.clip(v, eps, 1.0 - eps)

        # Empirical C(0.5, 0.5) ~ proportion of (u<=0.5, v<=0.5)
        c_hat = np.mean((u <= 0.5) & (v <= 0.5))
        beta_emp = 4.0 * c_hat - 1.0
        beta_emp = float(np.clip(beta_emp, -0.25, 0.25))
        theta0 = 4.0 * beta_emp  # donc theta0 ∈ [-1, 1]  # in [-4, 4] but true FGM theta in [-1,1]

        lo, hi = self.get_bounds()[0]
        eps_th = 1e-6
        theta0 = float(np.clip(theta0, lo + eps_th, hi - eps_th))

        self.set_parameters([theta0])
        return self.get_parameters()