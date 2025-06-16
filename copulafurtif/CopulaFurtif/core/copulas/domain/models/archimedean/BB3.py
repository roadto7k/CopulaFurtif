import numpy as np
from scipy.optimize import brentq
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


class BB3Copula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """
    BB3 Copula (Two-parameter Archimedean copula).

    Attributes:
        name (str): Human-readable name of the copula.
        type (str): Identifier for the copula family.
        bounds_param (list of tuple): Parameter bounds [d > 0, q >= 1].
        parameters (np.ndarray): Current copula parameters.
        default_optim_method (str): Optimization method.
    """

    def __init__(self):
        """Initialize BB3 copula with default parameters d=1.0, q=1.0."""
        super().__init__()
        self.name = "BB3 Copula"
        self.type = "bb3"
        self.bounds_param = [(1e-6, None), (1.0, None)]  # [d, q]
        self.param_names = ["d", "q"]
        self.parameters = [1.0, 1.0]
        self.default_optim_method = "Powell"


    def _h(self, s, param=None):
        """
        Compute the generator function h(s) = (log1p(s) / d)^(1/q).

        Args:
            s (float or array-like): Input to the generator.
            param (Sequence[float], optional): Copula parameters (d, q). Defaults to self.parameters.

        Returns:
            float or np.ndarray: Value of h(s).
        """

        if param is None:
            param = self.parameters
        d, q = param
        return (np.log1p(s) / d) ** (1.0 / q)

    def _h_prime(self, s, param=None):
        """
        Compute the first derivative hʼ(s) of the generator function.

        Args:
            s (float or array-like): Input to the generator.
            param (Sequence[float], optional): Copula parameters (d, q). Defaults to self.parameters.

        Returns:
            float or np.ndarray: Value of hʼ(s).
        """

        if param is None:
            param = self.parameters
        d, q = param
        g = np.log1p(s) / d
        return (1.0 / (q * d * (1.0 + s))) * g ** (1.0 / q - 1.0)

    def _h_double(self, s, param=None):
        """
        Compute the second derivative h″(s) of the generator function.

        Args:
            s (float or array-like): Input to the generator.
            param (Sequence[float], optional): Copula parameters (d, q). Defaults to self.parameters.

        Returns:
            float or np.ndarray: Value of h″(s).
        """

        if param is None:
            param = self.parameters
        d, q = param
        g = np.log1p(s) / d
        A = 1.0 / (q * d * (1.0 + s) ** 2)
        term1 = -g ** (1.0 / q - 1.0)
        term2 = (1.0 / q - 1.0) * g ** (1.0 / q - 2.0) / d
        return A * (term1 + term2)

    def get_cdf(self, u, v, param=None):
        """
        Evaluate the copula CDF at points (u, v) using the Gumbel generator.

        Args:
            u (float or array-like): First uniform margin in (0,1).
            v (float or array-like): Second uniform margin in (0,1).
            param (Sequence[float], optional): Copula parameters (d, q). Defaults to self.parameters.

        Returns:
            float or np.ndarray: CDF value C(u, v).
        """

        if param is None:
            param = self.parameters
        d, q = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        s_u = np.expm1(d * (-np.log(u)) ** q)
        s_v = np.expm1(d * (-np.log(v)) ** q)
        s = s_u + s_v
        return np.exp(-self._h(s, param))

    def get_pdf(self, u, v, param=None):
        """
        Evaluate the copula PDF at points (u, v) using the Gumbel generator.

        Args:
            u (float or array-like): First uniform margin in (0,1).
            v (float or array-like): Second uniform margin in (0,1).
            param (Sequence[float], optional): Copula parameters (d, q). Defaults to self.parameters.

        Returns:
            float or np.ndarray: PDF value c(u, v).
        """

        if param is None:
            param = self.parameters
        d, q = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        s_u = np.expm1(d * (-np.log(u)) ** q)
        s_v = np.expm1(d * (-np.log(v)) ** q)
        s = s_u + s_v
        h = self._h(s, param)
        h1 = self._h_prime(s, param)
        h2 = self._h_double(s, param)
        phi_dd = np.exp(-h) * (h1 ** 2 - h2)
        phi_inv_u_prime = -d * q * np.exp(d * (-np.log(u)) ** q) * (((-np.log(u)) ** (q - 1)) / u)
        phi_inv_v_prime = -d * q * np.exp(d * (-np.log(v)) ** q) * (((-np.log(v)) ** (q - 1)) / v)
        return phi_dd * phi_inv_u_prime * phi_inv_v_prime

    def kendall_tau(self, param=None, n=201):
        """
        Compute Kendall’s tau implied by the copula via numerical integration.

        Args:
            param (Sequence[float], optional): Copula parameters (d, q). Defaults to self.parameters.
            n (int, optional): Number of grid points per margin. Defaults to 201.

        Returns:
            float: Theoretical Kendall’s tau (4∫∫C(u,v)dudv − 1).
        """

        if param is None:
            param = self.parameters
        eps = 1e-6
        u = np.linspace(eps, 1 - eps, n)
        U, V = np.meshgrid(u, u)
        Z = self.get_cdf(U, V, param)
        integral = np.trapz(np.trapz(Z, u, axis=1), u)
        return 4.0 * integral - 1.0

    def sample(self, n, param=None):
        """
        Generate random samples from the copula using conditional inversion.

        Args:
            n (int): Number of samples to generate.
            param (Sequence[float], optional): Copula parameters (d, q). Defaults to self.parameters.

        Returns:
            numpy.ndarray: Array of shape (n, 2) with uniform samples on [0,1]².
        """

        if param is None:
            param = self.parameters
        samples = np.empty((n, 2))
        for i in range(n):
            u = np.random.rand()
            p = np.random.rand()
            root = brentq(lambda v_val: self.conditional_cdf_v_given_u(u, v_val, param) - p, 1e-6, 1 - 1e-6)
            samples[i] = [u, root]
        return samples

    def LTDC(self, param=None):
        """
        Compute the lower tail dependence coefficient (LTDC) of the copula.

        Args:
            param (Sequence[float], optional): Copula parameters (d, q). Defaults to self.parameters.

        Returns:
            float: LTDC value (0.0 for this copula).
        """

        return 0.0

    def UTDC(self, param=None):
        """
        Compute the upper tail dependence coefficient (UTDC) of the copula.

        Args:
            param (Sequence[float], optional): Copula parameters (d, q). Defaults to self.parameters.

        Returns:
            float: UTDC value (2 − 2^(1/q)).
        """


        if param is None:
            param = self.parameters
        q = param[1]
        return 2.0 - 2.0 ** (1.0 / q)

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """
        Compute the partial derivative ∂C(u,v)/∂u of the copula CDF.

        Args:
            u (float or array-like): First margin in (0,1).
            v (float or array-like): Second margin in (0,1).
            param (Sequence[float], optional): Copula parameters (d, q). Defaults to self.parameters.

        Returns:
            float or numpy.ndarray: Value of ∂C/∂u at (u,v).
        """

        if param is None:
            param = self.parameters
        d, q = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        s_u = np.expm1(d * (-np.log(u)) ** q)
        s_v = np.expm1(d * (-np.log(v)) ** q)
        s = s_u + s_v
        h1 = self._h_prime(s, param)
        phi_p = -h1 * np.exp(-self._h(s, param))
        phi_inv_u_prime = -d * q * np.exp(d * (-np.log(u)) ** q) * (((-np.log(u)) ** (q - 1)) / u)
        return phi_p * phi_inv_u_prime

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """
        Compute the conditional CDF P(V ≤ v | U = u).

        Args:
            u (float or array-like): Conditioning value of U in (0,1).
            v (float or array-like): Value of V in (0,1).
            param (Sequence[float], optional): Copula parameters (d, q). Defaults to self.parameters.

        Returns:
            float or numpy.ndarray: Conditional CDF of V given U.
        """

        return self.partial_derivative_C_wrt_u(v, u, param)

    def conditional_cdf_u_given_v(self, u, v, param=None):
        """
        Compute the conditional CDF P(V ≤ v | U = u).

        Args:
            u (float or array-like): Conditioning value of U in (0,1).
            v (float or array-like): Value of V in (0,1).
            param (Sequence[float], optional): Copula parameters (d, q). Defaults to self.parameters.

        Returns:
            float or numpy.ndarray: Conditional CDF of V given U.
        """

        return self.partial_derivative_C_wrt_v(u, v, param) / self.partial_derivative_C_wrt_v(1.0, v, param)

    def conditional_cdf_v_given_u(self, u, v, param=None):
        """
        Compute the conditional CDF P(V ≤ v | U = u).

        Args:
            u (float or array-like): Conditioning value of U in (0,1).
            v (float or array-like): Value of V in (0,1).
            param (Sequence[float], optional): Copula parameters (d, q). Defaults to self.parameters.

        Returns:
            float or numpy.ndarray: Conditional CDF of V given U.
        """

        return self.partial_derivative_C_wrt_u(u, v, param) / self.partial_derivative_C_wrt_u(u, 1.0, param)

    def IAD(self, data):
        """
        Return NaN for the IAD statistic, as it is disabled for this copula.

        Args:
            data (Sequence[array-like, array-like]): Ignored pseudo-observations.

        Returns:
            float: Always returns numpy.nan.
        """

        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        """
        Return NaN for the AD statistic, as it is disabled for this copula.

        Args:
            data (Sequence[array-like, array-like]): Ignored pseudo-observations.

        Returns:
            float: Always returns numpy.nan.
        """

        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan
