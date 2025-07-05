import numpy as np
from scipy.optimize import brentq
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


class BB4Copula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """
    BB4 Copula (Two-parameter Archimedean copula).

    Attributes:
        name (str): Human-readable name of the copula.
        type (str): Identifier for the copula family.
        bounds_param (list of tuple): Bounds for copula parameters [mu, delta].
        parameters (np.ndarray): Current copula parameters.
        default_optim_method (str): Optimization method.
    """

    def __init__(self):
        """Initialize BB4 copula with default parameters mu=1.0, delta=1.0."""
        super().__init__()
        self.name = "BB4 Copula"
        self.type = "bb4"
        self.bounds_param = [(1e-6, None), (1e-6, None)]  # [mu, delta]
        self.param_names = ["mu", "delta"]
        self.parameters = [1.0, 1.0]
        self.default_optim_method = "Powell"

    def get_cdf(self, u, v, param=None):
        """
        Evaluate the copula cumulative distribution function at (u, v).

        Args:
            u (float or array-like): First uniform margin in (0,1).
            v (float or array-like): Second uniform margin in (0,1).
            param (Sequence[float], optional): Copula parameters (mu, delta). Defaults to self.parameters.

        Returns:
            float or np.ndarray: CDF value C(u, v).
        """

        if param is None:
            param = self.parameters
        mu, delta = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        x = u ** (-mu)
        y = v ** (-mu)
        T = (x - 1) ** (-delta) + (y - 1) ** (-delta)
        A = T ** (-1.0 / delta)
        z = x + y - 1.0 - A
        return z ** (-1.0 / mu)

    def get_pdf(self, u, v, param=None):
        """
        Evaluate the copula probability density function at (u, v).

        Args:
            u (float or array-like): First uniform margin in (0,1).
            v (float or array-like): Second uniform margin in (0,1).
            param (Sequence[float], optional): Copula parameters (mu, delta). Defaults to self.parameters.

        Returns:
            float or np.ndarray: PDF value c(u, v).
        """

        if param is None:
            param = self.parameters
        mu, delta = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        x = u ** (-mu)
        y = v ** (-mu)
        T = (x - 1) ** (-delta) + (y - 1) ** (-delta)
        A = T ** (-1.0 / delta)
        z = x + y - 1.0 - A
        dzdu = -mu * u ** (-mu - 1) * (1.0 - (x - 1) ** (-delta - 1) * T ** (-1.0 / delta - 1))
        dzdv = -mu * v ** (-mu - 1) * (1.0 - (y - 1) ** (-delta - 1) * T ** (-1.0 / delta - 1))
        d2zdudv = -mu**2 * (delta + 1) * u**(-mu - 1) * v**(-mu - 1) * (x - 1)**(-delta - 1) * (y - 1)**(-delta - 1) * T**(-1.0 / delta - 2)
        dCdz = -1.0 / mu * z ** (-1.0 / mu - 1)
        d2Cdz2 = (1.0 / mu) * (1.0 / mu + 1.0) * z ** (-1.0 / mu - 2)
        return d2Cdz2 * dzdu * dzdv + dCdz * d2zdudv

    def kendall_tau(self, param=None, n=201):
        """
        Compute Kendall’s tau implied by the copula via numerical integration.

        Args:
            param (Sequence[float], optional): Copula parameters (mu, delta). Defaults to self.parameters.
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
            param (Sequence[float], optional): Copula parameters (mu, delta). Defaults to self.parameters.

        Returns:
            np.ndarray: Array of shape (n, 2) with uniform samples on [0,1]².
        """

        if param is None:
            param = self.parameters
        samples = np.empty((n, 2))
        for i in range(n):
            u = np.random.rand()
            p = np.random.rand()
            root = brentq(lambda vv: self.conditional_cdf_v_given_u(u, vv, param) - p, 1e-6, 1 - 1e-6)
            samples[i] = [u, root]
        return samples

    def LTDC(self, param=None):
        """
        Compute the lower tail dependence coefficient (LTDC) of the copula.

        Args:
            param (Sequence[float], optional): Copula parameters (mu, delta). Defaults to self.parameters.

        Returns:
            float: LTDC value (0.0 for this copula).
        """

        if param is None:
            param = self.parameters
        mu, delta = param
        return (2.0 - 2.0 ** (-1.0 / delta)) ** (-1.0 / mu)

    def UTDC(self, param=None):
        """
        Compute the upper tail dependence coefficient (UTDC) of the copula.

        Args:
            param (Sequence[float], optional): Copula parameters (mu, delta). Defaults to self.parameters.

        Returns:
            float: UTDC value (2 − 2^(1/δ)).
        """

        if param is None:
            param = self.parameters
        delta = param[1]
        return 2.0 - 2.0 ** (-1.0 / delta)

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """
        Compute the partial derivative ∂C(u,v)/∂u of the copula CDF.

        Args:
            u (float or array-like): First margin in (0,1).
            v (float or array-like): Second margin in (0,1).
            param (Sequence[float], optional): Copula parameters (mu, delta). Defaults to self.parameters.

        Returns:
            float or np.ndarray: Value of ∂C/∂u at (u, v).
        """

        if param is None:
            param = self.parameters
        mu, delta = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        x = u ** (-mu)
        y = v ** (-mu)
        T = (x - 1) ** (-delta) + (y - 1) ** (-delta)
        h1 = - (1.0 / mu) * ((x + y - 1.0 - T ** (-1.0 / delta)) ** (-1.0 / mu - 1))
        phi_inv_u_prime = -mu * u ** (-mu - 1) * (1 - (x - 1) ** (-delta - 1) * T ** (-1.0 / delta - 1))
        return h1 * phi_inv_u_prime

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """
        Compute the partial derivative ∂C(u,v)/∂v of the copula CDF.

        Args:
            u (float or array-like): First margin in (0,1).
            v (float or array-like): Second margin in (0,1).
            param (Sequence[float], optional): Copula parameters (mu, delta). Defaults to self.parameters.

        Returns:
            float or np.ndarray: Value of ∂C/∂v at (u, v).
        """

        return self.partial_derivative_C_wrt_u(v, u, param)

    def conditional_cdf_u_given_v(self, u, v, param=None):
        """
        Compute the conditional CDF P(U ≤ u | V = v).

        Args:
            u (float or array-like): Value of U in (0,1).
            v (float or array-like): Conditioning value of V in (0,1).
            param (Sequence[float], optional): Copula parameters (mu, delta). Defaults to self.parameters.

        Returns:
            float or np.ndarray: Conditional CDF of U given V.
        """

        return self.partial_derivative_C_wrt_v(u, v, param) / self.partial_derivative_C_wrt_v(1.0, v, param)

    def conditional_cdf_v_given_u(self, u, v, param=None):
        """
        Compute the conditional CDF P(V ≤ v | U = u).

        Args:
            u (float or array-like): Conditioning value of U in (0,1).
            v (float or array-like): Value of V in (0,1).
            param (Sequence[float], optional): Copula parameters (mu, delta). Defaults to self.parameters.

        Returns:
            float or np.ndarray: Conditional CDF of V given U.
        """

        return self.partial_derivative_C_wrt_u(u, v, param) / self.partial_derivative_C_wrt_u(u, 1.0, param)

    def IAD(self, data):
        """
        Return NaN for the Integrated Anderson-Darling (IAD) statistic.

        Args:
            data (Sequence[array-like, array-like]): Ignored pseudo-observations.

        Returns:
            float: Always returns np.nan.
        """

        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        """
        Return NaN for the Anderson-Darling (AD) statistic.

        Args:
            data (Sequence[array-like, array-like]): Ignored pseudo-observations.

        Returns:
            float: Always returns np.nan.
        """

        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan
