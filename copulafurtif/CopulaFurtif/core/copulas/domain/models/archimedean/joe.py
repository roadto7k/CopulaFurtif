"""
Joe Copula implementation.

The Joe copula is an asymmetric Archimedean copula that models strong upper tail dependence.
It is often used in finance and insurance to capture co-movements in the upper tails
of joint distributions.

Attributes:
    name (str): Human-readable name of the copula.
    type (str): Identifier for the copula family.
    bounds_param (list of tuple): Parameter bounds for theta ∈ (1.01, 30.0).
    parameters (np.ndarray): Copula parameter [theta].
    default_optim_method (str): Optimization method used for fitting.
"""

import numpy as np
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel, CopulaParameters
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence
from numpy.random import default_rng
from scipy.special import digamma, polygamma


class JoeCopula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """Joe Copula model."""

    def __init__(self):
        """Initialize the Joe copula with default parameters and bounds."""
        super().__init__()
        self.name = "Joe Copula"
        self.type = "joe"
        self.default_optim_method = "SLSQP"
        self.init_parameters(CopulaParameters([2.0],[(1.01, 30.0)] , ["theta"]))

    def get_cdf(self, u, v, param=None):
        """Compute the copula CDF C(u, v).

        Args:
            u (float or np.ndarray): First input in (0, 1).
            v (float or np.ndarray): Second input in (0, 1).
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float or np.ndarray: Value(s) of the CDF.
        """
        if param is None:
            theta = float(self.get_parameters()[0])
        else:
            theta = float(param[0])

        ubar = 1.0 - u
        vbar = 1.0 - v

        # S = ū^θ + v̄^θ  − ū^θ v̄^θ
        S = ubar ** theta + vbar ** theta - (ubar ** theta) * (vbar ** theta)
        return 1.0 - S ** (1.0 / theta)

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
            theta = float(self.get_parameters()[0])
        else:
            theta = float(param[0])
            # shorthand
        ubar = 1.0 - u
        vbar = 1.0 - v
        a = ubar ** theta
        b = vbar ** theta
        ab = a * b
        S = a + b - ab  # = ū^θ + v̄^θ − ū^θ v̄^θ

        # c(u,v) = S^(1/θ − 2) * ū^(θ−1) * v̄^(θ−1) * [θ − 1 + S]
        coef = S ** (1.0 / theta - 2.0)
        marg = (ubar ** (theta - 1.0)) * (vbar ** (theta - 1.0))
        hook = (theta - 1.0) + S

        return coef * marg * hook

    def sample(self, n, param=None, rng=None, tol=1e-12, max_iter=60):
        """
        Draw n samples from a Joe copula (θ ≥ 1) using conditional
        inversion with a monotone bisection solver.

        Parameters
        ----------
        n : int
        param : [θ], optional
        rng : np.random.Generator, optional
        tol : float
            Absolute tolerance for the root solver.
        max_iter : int
            Maximum bisection iterations.

        Returns
        -------
        (n, 2) ndarray of (U, V)
        """
        if rng is None:
            rng = default_rng()

        theta = float(self.get_parameters()[0]) if param is None else float(param[0])
        if theta < 1.0:
            raise ValueError("Joe copula requires θ ≥ 1.")
        if abs(theta - 1.0) < 1e-12:              # independence limit
            return rng.random((n, 2))

        # Step 1: draw U ~ Unif(0,1) and W ~ Unif(0,1)
        U = rng.random(n)
        W = rng.random(n)

        # Pre-compute constants for each sample
        u_bar  = 1.0 - U
        a      = u_bar**theta
        u_fac  = u_bar**(theta - 1.0)            # (1-u)^{θ-1}

        # Storage for V
        V = np.empty(n)

        # Bisection on b = (1-v)^θ in (0,1)
        for i in range(n):
            ai  = a[i]
            ui  = u_fac[i]
            wi  = W[i]

            # f(b) = C_{2|1}(v|u) - w   (monotonically ↓ in b)
            def f(b):
                S = ai + b - ai*b
                return ( S**(1.0/theta - 1.0) * ui * (1.0 - b) ) - wi

            lo, hi = 0.0, 1.0
            f_lo   = f(lo)      # ≈ +1
            f_hi   = f(hi - tol)  # ≈ −w  (negative)

            # Bisection
            for _ in range(max_iter):
                mid = 0.5*(lo + hi)
                f_mid = f(mid)
                if abs(f_mid) < tol:
                    break
                if f_mid * f_lo > 0:   # same sign => root in upper half
                    lo, f_lo = mid, f_mid
                else:
                    hi = mid
            b_root = 0.5*(lo + hi)

            # back-transform to v
            V[i] = 1.0 - b_root**(1.0/theta)

        # Light clipping
        eps = 1e-15
        np.clip(V, eps, 1.0 - eps, out=V)
        return np.column_stack((U, V))

    def kendall_tau(self, param=None):
        """
        Closed-form Kendall's τ for the Joe copula (θ ≥ 1).

        τ(θ)=1+2/(2−θ)·[ψ(2)−ψ(2/θ+1)].
        Handles θ≈1 (independence) and θ≈2 (removable singularity).
        """
        θ = float(self.get_parameters()[0]) if param is None else float(param[0])

        # independence
        if abs(θ - 1.0) < 1e-12:
            return 0.0

        # removable singularity at θ = 2
        if abs(θ - 2.0) < 1e-10:
            return 1.0 - polygamma(1, 2.0)  # trigamma(2)

        return 1.0 + 2.0 * (digamma(2.0) - digamma(2.0 / θ + 1.0)) / (2.0 - θ)

    def LTDC(self, param=None):
        """Lower tail dependence coefficient (0 for Joe copula).

        Args:
            param (np.ndarray, optional): Copula parameter.

        Returns:
            float: 0.0
        """
        return 0.0

    def UTDC(self, param=None):
        """Upper tail dependence coefficient for the Joe copula.

        Args:
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float: UTDC value.
        """
        if param is None:
            theta = float(self.get_parameters()[0])
        else:
            theta = float(param[0])

        return 2 - 2 ** (1 / theta)

    def IAD(self, data):
        """Integrated Absolute Deviation (disabled for Joe copula).

        Args:
            data (array-like): Input data (unused).

        Returns:
            float: NaN.
        """
        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        """Anderson–Darling test statistic (disabled for Joe copula).

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
            theta = float(self.get_parameters()[0])
        else:
            theta = float(param[0])

        A = (1 - u) ** theta
        B = (1 - v) ** theta
        Z = A + B - A * B

        return (1 - u) ** (theta - 1) * (1 - B) * Z ** (1 / theta - 1)

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

    def conditional_cdf_u_given_v(self, u, v, param=None):
        """Compute conditional CDF P(U ≤ u | V = v).

        Args:
            u (float or np.ndarray): U values.
            v (float or np.ndarray): V values.
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float or np.ndarray: Conditional CDF values.
        """
        return self.partial_derivative_C_wrt_v(u, v, param)

    def conditional_cdf_v_given_u(self, u, v, param=None):
        """Compute conditional CDF P(V ≤ v | U = u).

        Args:
            u (float or np.ndarray): U values.
            v (float or np.ndarray): V values.
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float or np.ndarray: Conditional CDF values.
        """
        return self.partial_derivative_C_wrt_u(u, v, param)
