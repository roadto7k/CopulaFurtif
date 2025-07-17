"""
Gumbel Copula implementation.

The Gumbel copula is an Archimedean copula that captures upper tail dependence, 
commonly used for modeling extreme value dependence structures. It is suitable 
for positively dependent data where strong upper tail correlation is expected.

Attributes:
    name (str): Human-readable name of the copula.
    type (str): Identifier of the copula type.
    bounds_param (list of tuple): Parameter bounds for theta ∈ (1.01, 30.0).
    parameters (np.ndarray): Copula parameter [theta].
    default_optim_method (str): Optimization method used during fitting.
"""

import numpy as np
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel, CopulaParameters
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence
from numpy.random import default_rng


class GumbelCopula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """Gumbel Copula model."""

    def __init__(self):
        """Initialize the Gumbel copula with default parameter and bounds."""
        super().__init__()
        self.name = "Gumbel Copula"
        self.type = "gumbel"
        self.default_optim_method = "SLSQP"
        self.init_parameters(CopulaParameters([2.0], [(1, 30)], ["theta"]))

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
        log_u = -np.log(u)
        log_v = -np.log(v)
        sum_pow = (log_u ** theta + log_v ** theta) ** (1 / theta)
        return np.exp(-sum_pow)

    def get_pdf(self, u, v, param=None):
        """
        Evaluate the bivariate Gumbel–Hougaard copula density c(u, v; θ).

        C(u, v) = exp(-( (-log u)**θ + (-log v)**θ )**(1/θ)),  θ ≥ 1.

        Parameters
        ----------
        u, v : float or array_like
            Values in (0,1) at which to evaluate the density.
        param : sequence-like, optional
            Copula parameter [θ]. If None, uses the object's current parameter.

        Returns
        -------
        pdf : ndarray or float
            Copula density evaluated at (u, v).
        """
        # parameter handling
        if param is None:
            theta = float(self.get_parameters()[0])
        else:
            theta = float(param[0])

        # numerical guard (avoid log(0))
        eps = 1e-15
        u = np.clip(u, eps, 1.0 - eps)
        v = np.clip(v, eps, 1.0 - eps)

        # shorthand
        x = -np.log(u)                # >0
        y = -np.log(v)                # >0
        A = x**theta + y**theta       # φ(u) + φ(v)
        C = A**(1.0 / theta)          # A^{1/θ}

        # density formula
        pref = np.exp(-C) / (u * v)
        core = (x**(theta - 1)) * (y**(theta - 1)) * A**(1.0 / theta - 2.0)
        bracket = (C + theta - 1.0)

        return pref * core * bracket

    @staticmethod
    def _rpositive_stable(alpha, size, rng):
        """
        Draw `size` i.i.d. positive α-stable variables S
        with Laplace transform  E[e^{-t S}] = exp(-t^{α}),  α∈(0,1].
        """
        if alpha == 1.0:  # degenerate at 1
            return np.ones(size)

        U = rng.uniform(low=0.0, high=np.pi, size=size)
        E = rng.exponential(scale=1.0, size=size)

        sinαU = np.sin(alpha * U)
        sinU = np.sin(U)
        cosU = np.cos(U)

        factor1 = (sinαU / sinU) ** (1.0 / alpha)
        factor2 = (np.sin((1.0 - alpha) * U) / E) ** ((1.0 - alpha) / alpha)

        return factor1 * factor2

    def sample(self, n, param=None, rng=None):
        """
        Generate `n` samples from a 2-D Gumbel copula.

        Parameters
        ----------
        n : int
            Sample size.
        param : array-like ([theta]), optional
        rng : numpy.random.Generator, optional

        Returns
        -------
        ndarray, shape (n, 2)
            (U, V) in (0, 1)² following Gumbel(θ).
        """
        if rng is None:
            rng = default_rng()

        # -------- parameter handling ----------------------------------
        if param is None:
            theta = float(self.get_parameters()[0])
        else:
            theta = float(param[0])

        # independence (θ≈1)
        if abs(theta - 1.0) < 1e-8:
            return rng.random((n, 2))

        if theta < 1.0:
            raise ValueError("Gumbel copula requires θ ≥ 1.")

        alpha = 1.0 / theta                     # stable index ∈ (0,1]

        # -------- 1) mixing variable S ~ positive stable (α)
        S = self._rpositive_stable(alpha, n, rng)

        # -------- 2) independent exponentials
        E1 = rng.exponential(scale=1.0, size=n)
        E2 = rng.exponential(scale=1.0, size=n)

        # -------- 3) transform via generator inverse
        U = np.exp(-E1 / S)
        V = np.exp(-E2 / S)

        # tiny clip for numerical safety
        eps = 1e-15
        np.clip(U, eps, 1.0 - eps, out=U)
        np.clip(V, eps, 1.0 - eps, out=V)

        return np.column_stack((U, V))
    def kendall_tau(self, param=None):
        """Compute Kendall's tau for the Gumbel copula.

        Args:
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float: Kendall's tau.
        """
        if param is None:
            param = self.get_parameters()
        theta = param[0]
        return 1 - 1 / theta

    def LTDC(self, param=None):
        """Lower tail dependence coefficient (0 for Gumbel copula).

        Args:
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float: LTDC value.
        """
        return 0

    def UTDC(self, param=None):
        """Upper tail dependence coefficient (same as LTDC for Gumbel copula).

        Args:
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float: UTDC value.
        """
        if param is None:
            param = self.get_parameters()
        theta = param[0]
        return 2 - 2 ** (1 / theta)

    def IAD(self, data):
        """Integrated Absolute Deviation (disabled for Gumbel copula).

        Args:
            data (array-like): Input data (unused).

        Returns:
            float: NaN.
        """
        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        """Anderson–Darling test statistic (disabled for Gumbel copula).

        Args:
            data (array-like): Input data (unused).

        Returns:
            float: NaN.
        """
        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """Compute ∂C(u,v)/∂u.

        Args:
            u (float or np.ndarray): U values.
            v (float or np.ndarray): V values.
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float or np.ndarray: Partial derivative values.
        """
        if param is None:
            param = self.get_parameters()
        theta = param[0]
        log_u = -np.log(u)
        log_v = -np.log(v)
        A = log_u ** theta + log_v ** theta
        C = A ** (1 / theta)
        return np.exp(-C) * log_u ** (theta - 1) * (log_v ** theta + log_u ** theta) ** (1 / theta - 1) / u

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """Compute ∂C(u,v)/∂v via symmetry.

        Args:
            u (float or np.ndarray): U values.
            v (float or np.ndarray): V values.
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float or np.ndarray: Partial derivative values.
        """
        return self.partial_derivative_C_wrt_u(v, u, param)
