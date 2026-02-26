"""
Ali-Mikhail-Haq (AMH) Copula implementation.

The AMH copula is an Archimedean copula with a single parameter that allows for a wide 
range of dependence structures, though it has limited tail dependence. This class supports 
CDF/PDF evaluation, parameter fitting, sampling (placeholder), and basic diagnostics.

Attributes:
    name (str): Human-readable name of the copula.
    type (str): Copula identifier.
    bounds_param (list of tuple): Bounds for the parameter theta ∈ (-0.999, 1.0).
    parameters (np.ndarray): Copula parameter [theta].
    default_optim_method (str): Optimization method used during fitting.
"""

import numpy as np
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel, CopulaParameters
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence
from numpy.random import default_rng


class AMHCopula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """AMH Copula model."""

    def __init__(self):
        """Initialize the AMH copula with default parameter and bounds."""
        super().__init__()
        self.name = "AMH Copula"
        self.type = "amh"
        # self.bounds_param = [(-1.0, 1.0)]
        # self.param_names = ["theta"]
        # self.parameters = [0.3]
        self.default_optim_method = "SLSQP"
        self.init_parameters(CopulaParameters(np.array([0.3]),
                            [(-1.0, 1.0)], 
                            ["theta"]))

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
        u = np.clip(u, eps, 1.0 - eps)
        v = np.clip(v, eps, 1.0 - eps)

        theta = param[0]
        num = u * v
        denom = 1 - theta * (1 - u) * (1 - v)
        return num / denom

    def get_pdf(self, u, v, param=None):
        """
        Evaluate the Ali–Mikhail–Haq (AMH) copula density c(u, v; θ).

        C(u, v) = u v / [1 - θ (1-u)(1-v)],   θ ∈ (-1, 1).

        Parameters
        ----------
        u, v : array_like
            Points in (0,1) where the density is evaluated (broadcastable).
        param : sequence-like, optional
            Copula parameter [θ]; if None, uses current instance parameter.

        Returns
        -------
        pdf : ndarray
            Density values c(u,v;θ) with NumPy broadcasting.
        """
        # parameter
        theta = float(self.get_parameters()[0]) if param is None else float(param[0])

        # guard against endpoints (avoid division by zero when D→0 numerically)
        eps = 1e-12
        u = np.clip(u, eps, 1.0 - eps)
        v = np.clip(v, eps, 1.0 - eps)

        one_minus_u = 1.0 - u
        one_minus_v = 1.0 - v

        D = 1.0 - theta * one_minus_u * one_minus_v             # denom in C(u,v)

        # Numerator N in stable (1-u),(1-v) form
        N = (
            1.0 + theta
            + theta * (theta + 1.0) * one_minus_u * one_minus_v
            - 2.0 * theta * (one_minus_u + one_minus_v)
        )

        return N / (D**3)

    def sample(self, n, param=None, rng=None):
        """
        Draw `n` i.i.d. samples from the 2-D AMH copula.

        Parameters
        ----------
        n : int
            Sample size.
        param : array-like, optional
            Single element `[theta]`. If None, uses current parameters.
        rng : numpy.random.Generator, optional
            Pass your own RNG for repeatability.

        Returns
        -------
        ndarray, shape (n, 2)
            Pseudo-observations (U, V) in (0,1)².
        """
        # ---------------- RNG + θ -------------------------------------
        if rng is None:
            rng = default_rng()

        if param is None:
            theta = float(self.get_parameters()[0])
        else:
            theta = float(param[0])

        # domain check  (−1 < θ < 1, θ≠1)
        if not (-1.0 < theta < 1.0):
            raise ValueError("AMH parameter must be in (−1, 1).")

        # independence
        if abs(theta) < 1e-12:
            return rng.random((n, 2))

        # ---------------- core algorithm ------------------------------
        # 1) U ~ Unif(0,1),  Z ~ Unif(0,1)
        U = rng.random(n)
        Z = rng.random(n)

        a = theta
        k = 1.0 - U
        b = 1.0 - a * k          # = 1 − θ(1−U)
        c = a * k                # = θ(1−U)
        d = 1.0 - a              # = 1 − θ

        # Quadratic coefficients in V  (derived from C_{2|1}(v|u) = z)
        A = Z * c * c - a
        B = 2.0 * Z * b * c - d
        C = Z * b * b

        disc = np.maximum(B * B - 4.0 * A * C, 0.0)
        sqrt_disc = np.sqrt(disc)

        V1 = (-B + sqrt_disc) / (2.0 * A)
        V2 = (-B - sqrt_disc) / (2.0 * A)

        # pick the root that falls in (0,1)
        V = np.where((V1 > 0.0) & (V1 < 1.0), V1, V2)

        # handle the (rare) A ≈ 0 limit numerically
        mask = np.abs(A) < 1e-12
        V[mask] = -C[mask] / B[mask]

        # clip to avoid exact 0/1
        eps = 1e-15
        np.clip(V, eps, 1.0 - eps, out=V)
        return np.column_stack((U, V))

    def kendall_tau(self, param=None):
        """Compute Kendall's tau for the AMH copula.

        Args:
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float: Kendall's tau.
        """
        if param is None:
            param = self.get_parameters()
        theta = param[0]
        if abs(theta) < 1e-12:
            return 0.0
        return 1.0 - 2.0 * ((1 - theta) ** 2 * np.log1p(-theta) + theta) / (3 * theta ** 2)

    def LTDC(self, param=None):
        """Lower tail dependence coefficient (always 0 for AMH copula).

        Args:
            param (np.ndarray, optional): Copula parameter.

        Returns:
            float: 0.0
        """
        return 0.0

    def UTDC(self, param=None):
        """Upper tail dependence coefficient (always 0 for AMH copula).

        Args:
            param (np.ndarray, optional): Copula parameter.

        Returns:
            float: 0.0
        """
        return 0.0

    def IAD(self, data):
        """Integrated Absolute Deviation (disabled for AMH copula).

        Args:
            data (array-like): Input data (unused).

        Returns:
            float: NaN.
        """
        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        """Anderson–Darling statistic (disabled for AMH copula).

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
            u (float or np.ndarray): U value(s).
            v (float or np.ndarray): V value(s).
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float or np.ndarray: Conditional CDF value.
        """
        if param is None:
            param = self.get_parameters()
        theta = param[0]
        eps = 1e-12
        u = np.clip(u, eps, 1.0 - eps)
        v = np.clip(v, eps, 1.0 - eps)

        D = 1.0 - theta * (1.0 - u) * (1.0 - v)
        h = v * (1.0 - theta * (1.0 - v)) / (D ** 2)
        h = np.clip(h, 0.0, 1.0)
        return float(h) if np.ndim(h) == 0 else h

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """Compute ∂C(u,v)/∂v (via symmetry).

        Args:
            u (float or np.ndarray): U value(s).
            v (float or np.ndarray): V value(s).
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float or np.ndarray: Conditional CDF value.
        """
        return self.partial_derivative_C_wrt_u(v, u, param)

    def blomqvist_beta(self, param=None) -> float:
        """Blomqvist's beta for AMH: beta = theta / (4 - theta)."""
        import numpy as np

        if param is None:
            param = self.get_parameters()
        theta = float(param[0])

        beta = theta / (4.0 - theta)
        return float(beta)

    def init_from_data(self, u, v):
        """
        Initialize theta from data using empirical Blomqvist beta (preferred for AMH).

        Empirical beta:
            beta_hat = 4 * C_n(1/2,1/2) - 1
                    = 4 * mean( 1{u<=0.5, v<=0.5} ) - 1

        AMH inversion:
            beta(theta) = theta / (4 - theta)
            theta(beta) = 4*beta / (1 + beta)
        """

        u = np.asarray(u, dtype=float).ravel()
        v = np.asarray(v, dtype=float).ravel()

        mask = np.isfinite(u) & np.isfinite(v)
        u = u[mask]
        v = v[mask]
        if u.size < 20:
            return self.get_parameters()

        eps_uv = 1e-12
        u = np.clip(u, eps_uv, 1.0 - eps_uv)
        v = np.clip(v, eps_uv, 1.0 - eps_uv)

        # empirical C(1/2, 1/2)
        c_hat = float(np.mean((u <= 0.5) & (v <= 0.5)))
        beta_emp = 4.0 * c_hat - 1.0
        if not np.isfinite(beta_emp):
            return self.get_parameters()

        # AMH theoretical beta range for theta in (-1, 1): beta in (-1/5, 1/3)
        beta_lo = -1.0 / 5.0
        beta_hi = 1.0 / 3.0
        eps_b = 1e-6
        beta_emp = float(np.clip(beta_emp, beta_lo + eps_b, beta_hi - eps_b))

        # invert beta -> theta
        theta0 = 4.0 * beta_emp / (1.0 + beta_emp)

        # exclusive bounds safety
        low, high = self.get_bounds()[0]  # expected (-1, 1)
        eps_th = 1e-6
        theta0 = float(np.clip(theta0, low + eps_th, high - eps_th))

        self.set_parameters([theta0])
        return self.get_parameters()
