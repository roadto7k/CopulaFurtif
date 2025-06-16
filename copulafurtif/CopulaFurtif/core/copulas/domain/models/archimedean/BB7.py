import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence
from scipy.special import beta as beta_fn


class BB7Copula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """
    BB7 Copula (Joe-Clayton) Archimedean copula.

    Attributes:
        name (str): Name of the copula.
        type (str): Identifier for the copula family.
        bounds_param (list of tuple): Parameter bounds for optimization.
        parameters (np.ndarray): Copula parameters [theta, delta].
        default_optim_method (str): Optimization method.
    """

    def __init__(self):
        """Initialize the BB7 copula with default parameters."""
        super().__init__()
        self.name = "BB7 Copula"
        self.type = "bb7"
        self.bounds_param = [(1e-6, None), (1e-6, None)]  # [theta, delta]
        self.param_names = ["theta", "delta"]
        self.parameters = [1.0, 1.0]
        self.default_optim_method = "Powell"


    def _phi(self, t, theta, delta):
        """
        Compute the φ-transform for the BB7 copula.

        Args:
            t (float or array-like): Input variable in (0,1).
            theta (float): Copula parameter θ.
            delta (float): Copula parameter δ.

        Returns:
            float or numpy.ndarray: Value of φ(t) = (φJ⁻ᵟ − 1)/δ, where φJ = 1 − (1−t)ᵗʰᵉᵗᵃ.
        """

        t = np.clip(t, 1e-12, 1 - 1e-12)
        phiJ = 1.0 - (1.0 - t)**theta
        return (phiJ**(-delta) - 1.0) / delta

    def _phi_inv(self, s, theta, delta):
        """
        Compute the inverse φ-transform for the BB7 copula.

        Args:
            s (float or array-like): Transformed variable ≥ 0.
            theta (float): Copula parameter θ.
            delta (float): Copula parameter δ.

        Returns:
            float or numpy.ndarray: Value of φ⁻¹(s).
        """

        s = np.maximum(s, 0.0)
        phiC_inv = (1.0 + delta * s)**(-1.0 / delta)
        return 1.0 - (1.0 - phiC_inv)**(1.0 / theta)

    def get_cdf(self, u, v, param=None):
        """
        Evaluate the BB7 copula cumulative distribution function at (u, v).

        Args:
            u (float or array-like): First uniform margin in (0,1).
            v (float or array-like): Second uniform margin in (0,1).
            param (Sequence[float], optional): Copula parameters (θ, δ). Defaults to self.parameters.

        Returns:
            float or numpy.ndarray: CDF value C(u, v).
        """

        if param is None:
            param = self.parameters
        theta, delta = param
        phi_u = self._phi(u, theta, delta)
        phi_v = self._phi(v, theta, delta)
        return self._phi_inv(phi_u + phi_v, theta, delta)

    def get_pdf(self, u, v, param=None):
        """
        Approximate the BB7 copula probability density function at (u, v) via finite differences.

        Args:
            u (float or array-like): First uniform margin in (0,1).
            v (float or array-like): Second uniform margin in (0,1).
            param (Sequence[float], optional): Copula parameters (θ, δ). Defaults to self.parameters.

        Returns:
            float or numpy.ndarray: Approximate PDF c(u, v).
        """

        if param is None:
            param = self.parameters
        eps = 1e-6
        c = self.get_cdf
        return (
            c(u+eps, v+eps, param) - c(u+eps, v-eps, param)
            - c(u-eps, v+eps, param) + c(u-eps, v-eps, param)
        ) / (4.0 * eps**2)

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """
        Approximate the partial derivative ∂C(u,v)/∂u for the BB7 copula via finite differences.

        Args:
            u (float or array-like): First uniform margin in (0,1).
            v (float or array-like): Second uniform margin in (0,1).
            param (Sequence[float], optional): Copula parameters (θ, δ). Defaults to self.parameters.

        Returns:
            float or numpy.ndarray: Approximate ∂C/∂u at (u, v).
        """

        if param is None:
            param = self.parameters
        eps = 1e-6
        c = self.get_cdf
        return (c(u+eps, v, param) - c(u-eps, v, param)) / (2.0 * eps)

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """
        Approximate the partial derivative ∂C(u,v)/∂v for the BB7 copula.

        Args:
            u (float or array-like): First uniform margin in (0,1).
            v (float or array-like): Second uniform margin in (0,1).
            param (Sequence[float], optional): Copula parameters (θ, δ). Defaults to self.parameters.

        Returns:
            float or numpy.ndarray: Approximate ∂C/∂v at (u, v).
        """

        return self.partial_derivative_C_wrt_u(v, u, param)

    def conditional_cdf_u_given_v(self, u, v, param=None):
        """
        Compute the conditional CDF P(U ≤ u | V = v) for the BB7 copula.

        Args:
            u (float or array-like): Value of U in (0,1).
            v (float or array-like): Conditioning value of V in (0,1).
            param (Sequence[float], optional): Copula parameters (θ, δ). Defaults to self.parameters.

        Returns:
            float or numpy.ndarray: Conditional CDF of U given V.
        """

        num = self.partial_derivative_C_wrt_v(u, v, param)
        den = self.partial_derivative_C_wrt_v(1.0, v, param)
        return num / den

    def conditional_cdf_v_given_u(self, u, v, param=None):
        """
        Compute the conditional CDF P(V ≤ v | U = u) for the BB7 copula.

        Args:
            u (float or array-like): Conditioning value of U in (0,1).
            v (float or array-like): Value of V in (0,1).
            param (Sequence[float], optional): Copula parameters (θ, δ). Defaults to self.parameters.

        Returns:
            float or numpy.ndarray: Conditional CDF of V given U.
        """

        num = self.partial_derivative_C_wrt_u(u, v, param)
        den = self.partial_derivative_C_wrt_u(u, 1.0, param)
        return num / den

    def sample(self, n, param=None):
        """
        Generate random samples from the BB7 copula using conditional inversion.

        Args:
            n (int): Number of samples to generate.
            param (Sequence[float], optional): Copula parameters (θ, δ). Defaults to self.parameters.

        Returns:
            numpy.ndarray: Array of shape (n, 2) with uniform samples on [0,1]^2.
        """

        if param is None:
            param = self.parameters
        samples = np.empty((n, 2))
        eps = 1e-6
        for i in range(n):
            u = np.random.rand()
            p = np.random.rand()
            root = brentq(
                lambda vv: self.conditional_cdf_v_given_u(u, vv, param) - p,
                eps, 1.0 - eps
            )
            samples[i] = [u, root]
        return samples

    def LTDC(self, param=None):
        """
        Compute the lower tail dependence coefficient (LTDC) for the BB7 copula.

        Args:
            param (Sequence[float], optional): Copula parameters (θ, δ). Defaults to self.parameters.

        Returns:
            float: LTDC value = lim₍u→0₎ C(u,u)/u.
        """

        if param is None:
            param = self.parameters
        eps = 1e-6
        return self.get_cdf(eps, eps, param) / eps

    def UTDC(self, param=None):
        """
        Compute the upper tail dependence coefficient (UTDC) for the BB7 copula.

        Args:
            param (Sequence[float], optional): Copula parameters (θ, δ). Defaults to self.parameters.

        Returns:
            float: UTDC value = 2 − lim₍u→1₎ [1 − 2u + C(u,u)]/(1−u).
        """

        if param is None:
            param = self.parameters
        eps = 1e-6
        u = 1.0 - eps
        return 2.0 - (1.0 - 2*u + self.get_cdf(u, u, param)) / eps

    def kendall_tau(self, param=None, analytic=True):
        """
        Compute Kendall's tau for the BB7 (Joe–Clayton) copula.

        Args:
            param (Sequence[float], optional): (theta, delta). Defaults to self.parameters.
            analytic (bool): Use closed-form formula (default) or numeric quad fallback.

        Returns:
            float: Kendall's tau.
        """
        if param is None:
            param = self.parameters
        theta, delta = map(float, param)

        if analytic:
            # Closed-form via Beta function
            y = 2.0 / theta - 1.0  # second Beta parameter
            tau = 1.0 - 4.0 / (delta * theta ** 2) * (
                    beta_fn(2.0, y) - beta_fn(delta + 2.0, y)
            )
            return tau

        # --- numeric fallback (robust near pathological values) ---
        def phi(t):
            t = np.clip(t, 1e-12, 1 - 1e-12)
            phiJ = 1.0 - (1.0 - t) ** theta
            return (phiJ ** (-delta) - 1.0) / delta

        def phi_prime(t):
            t = np.clip(t, 1e-12, 1 - 1e-12)
            phiJ = 1.0 - (1.0 - t) ** theta
            dphiJ_dt = theta * (1.0 - t) ** (theta - 1)
            return -phiJ ** (-delta - 1) * dphiJ_dt

        integrand = lambda t: phi(t) / phi_prime(t)
        integral, _ = quad(integrand, 0.0, 1.0, epsabs=1e-10, epsrel=1e-10)
        return 1.0 + 4.0 * integral

    def IAD(self, data):
        """
        Return NaN for the Integrated Anderson-Darling (IAD) statistic for BB7.

        Args:
            data (Sequence[array-like, array-like]): Ignored pseudo-observations.

        Returns:
            float: Always returns numpy.nan.
        """

        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        """
        Return NaN for the Anderson-Darling (AD) statistic for BB7.

        Args:
            data (Sequence[array-like, array-like]): Ignored pseudo-observations.

        Returns:
            float: Always returns numpy.nan.
        """

        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan
