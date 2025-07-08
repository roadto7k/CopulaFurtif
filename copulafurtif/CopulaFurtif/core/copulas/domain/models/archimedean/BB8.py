import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel, CopulaParameters
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


class BB8Copula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """
    BB8 Copula (Durante et al.):
      C(u,v) = [1 - (1-A)*(1-B)]^(1/theta),
      where A = [1 - (1-u)^theta]^delta,
            B = [1 - (1-v)^theta]^delta.

    Attributes:
        name (str): Human-readable name of the copula.
        type (str): Identifier for the copula family.
        bounds_param (list of tuple): Bounds for copula parameters.
        parameters (np.ndarray): Current copula parameters.
        default_optim_method (str): Default method for optimization.
    """

    def __init__(self):
        super().__init__()
        self.name = "BB8 Copula (Durante)"
        self.type = "bb8"
        self.default_optim_method = "Powell"
        self.init_parameters(CopulaParameters([2.0, 0.7], [(1.0, np.inf), (0.0, 1.0)], ["theta", "delta"]))
        
    def get_cdf(self, u, v, param=None):
        """
        Evaluate the BB8 copula cumulative distribution function at (u, v).

        Args:
            u (float or array-like): First uniform margin in (0,1).
            v (float or array-like): Second uniform margin in (0,1).
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float or np.ndarray: CDF value C(u, v).
        """

        if param is None:
            param = self.get_parameters()
        theta, delta = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        A = (1.0 - (1.0 - u)**theta)**delta
        B = (1.0 - (1.0 - v)**theta)**delta
        inner = 1.0 - (1.0 - A)*(1.0 - B)
        return inner**(1.0/theta)

    def get_pdf(self, u, v, param=None):
        """
        Approximate the BB8 copula probability density function at (u, v) via finite differences.

        Args:
            u (float or array-like): First uniform margin in (0,1).
            v (float or array-like): Second uniform margin in (0,1).
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float or np.ndarray: Approximate PDF c(u, v).
        """

        if param is None:
            param = self.get_parameters()
        eps = 1e-6
        c = self.get_cdf
        return (
            c(u+eps, v+eps, param) - c(u+eps, v-eps, param)
            - c(u-eps, v+eps, param) + c(u-eps, v-eps, param)
        ) / (4.0 * eps**2)

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """
        Approximate the partial derivative ∂C(u,v)/∂u for the BB8 copula via finite differences.

        Args:
            u (float or array-like): First margin in (0,1).
            v (float or array-like): Second margin in (0,1).
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float or np.ndarray: Approximate ∂C/∂u at (u, v).
        """

        if param is None:
            param = self.get_parameters()
        eps = 1e-6
        c = self.get_cdf
        return (c(u+eps, v, param) - c(u, v, param)) / eps

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """
        Approximate the partial derivative ∂C(u,v)/∂v for the BB8 copula via finite differences.

        Args:
            u (float or array-like): First margin in (0,1).
            v (float or array-like): Second margin in (0,1).
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float or np.ndarray: Approximate ∂C/∂v at (u, v).
        """

        if param is None:
            param = self.get_parameters()
        eps = 1e-6
        c = self.get_cdf
        return (c(u, v+eps, param) - c(u, v, param)) / eps

    def conditional_cdf_v_given_u(self, u, v, param=None):
        """
        Compute the conditional CDF P(V ≤ v | U = u) for the BB8 copula.

        Args:
            u (float or array-like): Conditioning value of U in (0,1).
            v (float or array-like): Value of V in (0,1).
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float or np.ndarray: Conditional CDF of V given U.
        """

        return self.partial_derivative_C_wrt_u(u, v, param)

    def conditional_cdf_u_given_v(self, u, v, param=None):
        """
        Compute the conditional CDF P(U ≤ u | V = v) for the BB8 copula.

        Args:
            u (float or array-like): Value of U in (0,1).
            v (float or array-like): Conditioning value of V in (0,1).
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float or np.ndarray: Conditional CDF of U given V.
        """

        return self.partial_derivative_C_wrt_v(u, v, param)

    def sample(self, n, param=None):
        """
        Generate random samples from the BB8 copula using conditional inversion.

        Args:
            n (int): Number of samples to generate.
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            numpy.ndarray: Array of shape (n, 2) with uniform samples on [0,1]^2.
        """

        if param is None:
            param = self.get_parameters()
        samples = np.empty((n, 2))
        eps = 1e-6
        for i in range(n):
            u = np.random.rand()
            p = np.random.rand()
            root = root_scalar(
                lambda vv: self.partial_derivative_C_wrt_u(u, vv, param) - p,
                bracket=[eps, 1 - eps], method='bisect', xtol=1e-6
            )
            samples[i] = [u, root.root]
        return samples

    def LTDC(self, param=None):
        """
        Compute the lower tail dependence coefficient (LTDC) for the BB8 copula.

        Args:
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float: LTDC value = lim_{u→0} C(u,u)/u.
        """

        if param is None:
            param = self.get_parameters()
        u = 1e-6
        return self.get_cdf(u, u, param) / u

    def UTDC(self, param=None):
        """
        Compute the upper tail dependence coefficient (UTDC) for the BB8 copula.

        Args:
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float: UTDC value = lim_{u→1} [1 − 2u + C(u,u)]/(1−u).
        """

        if param is None:
            param = self.get_parameters()
        u = 1.0 - 1e-6
        return (1 - 2*u + self.get_cdf(u, u, param)) / (1 - u)

    def kendall_tau(self, param=None):
        """
        Compute Kendall's tau for the BB8 copula.

        τ(θ,δ) = 1 + 4 ∫₀¹ g(t;θ,δ) dt
        with
            g(t) = -log( ((1 - δ t)**θ - 1) / ((1 - δ)**θ - 1) )
                   * (1 - δ t - (1 - δ t)**(-θ) + (1 - δ t)**(-θ) δ t)
                   / (θ δ)

        Args
        ----
        param : sequence (theta, delta), optional
            Defaults to the object’s current parameters.

        Returns
        -------
        float
            Theoretical Kendall’s τ.
        """
        # unpack parameters and basic validation
        if param is None:
            param = self.get_parameters()
        theta, delta = param
        if theta <= 1 or not (0 < delta < 1):
            raise ValueError("BB8 requires θ>1 and 0<δ<1")

        # pre-compute denominator once (never zero inside admissible set)
        denom = (1.0 - delta) ** theta - 1.0

        # integrand as a plain Python lambda (quad handles the loops)
        def _g(t):
            base = 1.0 - delta * t
            num = base ** theta - 1.0
            log_term = -np.log(num / denom)
            aux = 1.0 - delta * t - base ** (-theta) + base ** (-theta) * delta * t
            return log_term * aux / (theta * delta)

        # high-accuracy adaptive integration on [0,1]
        integral, _ = quad(_g, 0.0, 1.0, epsabs=1e-10, epsrel=1e-10, limit=200)

        return 1.0 + 4.0 * integral

    def IAD(self, data):
        """
        Return NaN for the Integrated Anderson-Darling (IAD) statistic for BB8.

        Args:
            data (Sequence[array-like, array-like]): Ignored pseudo-observations.

        Returns:
            float: Always returns numpy.nan.
        """

        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        """
        Return NaN for the Anderson-Darling (AD) statistic for BB8.

        Args:
            data (Sequence[array-like, array-like]): Ignored pseudo-observations.

        Returns:
            float: Always returns numpy.nan.
        """

        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan
