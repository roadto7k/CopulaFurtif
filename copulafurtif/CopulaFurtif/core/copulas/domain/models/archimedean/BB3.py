import numpy as np
from numpy.random import default_rng
from scipy.optimize import brentq

from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel, CopulaParameters
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


class BB3Copula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """
    BB3 (Joe–Hu 1996) – Positive-stable stopped-gamma LT copula..

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
        # self.bounds_param = [(1e-6, 30), (1.0, 5.0)] # [d, q]
        # self.param_names = ["d", "q"] # d = δ, q = θ
        # self.get_parameters() = [2.0, 1.2] # safe
        self.default_optim_method = "Powell"
        self.init_parameters(CopulaParameters([2, 1.5], [(1.0, 10.0),(0.05, 10.0) ], ["theta", "delta"]))


    def _h(self, s, param=None):
        """
        Compute the generator function h(s) = (log1p(s) / theta)^(1/delta).

        Args:
            s (float or array-like): Input to the generator.
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float or np.ndarray: Value of h(s).
        """

        theta, delta = self.get_parameters() if param is None else param
        return (np.log1p(s) / delta) ** (1.0 / theta)

    def _h_prime(self, s, param=None):
        """
        Compute the first derivative hʼ(s) of the generator function.

        Args:
            s (float or array-like): Input to the generator.
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float or np.ndarray: Value of hʼ(s).
        """

        theta, delta = self.get_parameters() if param is None else param
        g = np.log1p(s) / delta
        return g ** (1.0 / theta - 1.0) / (theta * delta * (1.0 + s))

    def _h_double(self, s, param=None):
        """
        Compute the second derivative h″(s) of the generator function.

        Args:
            s (float or array-like): Input to the generator.
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float or np.ndarray: Value of h″(s).
        """

        theta, delta = self.get_parameters() if param is None else param
        inv_theta = 1.0 / theta
        g = np.log1p(s) / delta

        return (inv_theta * g ** (inv_theta - 2) /
                (delta ** 2 * (1.0 + s) ** 2) *
                ((inv_theta - 1.0) - delta * g))

    def get_cdf(self, u, v, param=None):
        """
        Evaluate the copula CDF at points (u, v) using the  BB3 positive-stable stopped-gamma.

        Args:
            u (float or array-like): First uniform margin in (0,1).
            v (float or array-like): Second uniform margin in (0,1).
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float or np.ndarray: CDF value C(u, v).
        """

        theta, delta = self.get_parameters() if param is None else param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        s_u = np.expm1(delta * (-np.log(u)) ** theta)
        s_v = np.expm1(delta * (-np.log(v)) ** theta)
        return np.exp(-self._h(s_u + s_v, param))

    def get_pdf(self, u, v, param=None):
        """
        Evaluate the copula PDF at points (u, v) using the  BB3 positive-stable stopped-gamma.

        Args:
            u (float or array-like): First uniform margin in (0,1).
            v (float or array-like): Second uniform margin in (0,1).
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float or np.ndarray: PDF value c(u, v).
        """

        theta, delta = self.get_parameters() if param is None else param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        s_u = np.expm1(delta * (-np.log(u)) ** theta)
        s_v = np.expm1(delta * (-np.log(v)) ** theta)
        s = s_u + s_v

        h = self._h(s, param)
        h1 = self._h_prime(s, param)
        h2 = self._h_double(s, param)
        phi_dd = np.exp(-h) * (h1 ** 2 - h2)

        inv_u_prime = -delta * theta * np.exp(delta * (-np.log(u)) ** theta) * \
                      (-np.log(u)) ** (theta - 1) / u
        inv_v_prime = -delta * theta * np.exp(delta * (-np.log(v)) ** theta) * \
                      (-np.log(v)) ** (theta - 1) / v

        pdf = phi_dd * inv_u_prime * inv_v_prime
        return np.nan_to_num(pdf, nan=0.0, neginf=0.0, posinf=np.finfo(float).max)

    def kendall_tau(self, param=None, m: int = 800) -> float:
        """
        Compute Kendall’s tau for the BB3 copula by numerical integration
        over a uniform m×m grid on [0,1]^2:

            τ = 4 * E[C(U,V)] − 1,   (U,V) ∼ Uniform[0,1]^2

        Parameters
        ----------
        param : sequence of two floats, optional
            Copula parameters [theta, delta]. If None, the model’s current
            parameters are used.
        m : int, default=800
            Number of grid points per dimension. A larger m increases
            accuracy at the cost of more computation.

        Returns
        -------
        float
            Theoretical Kendall’s tau in [−1, 1].
        """
        # 1) unpack parameters
        theta, delta = (self.get_parameters() if param is None else param)

        # 2) build a regular grid avoiding the exact boundaries 0 and 1
        #    by centering points in each cell: (i − 0.5) / m
        grid = (np.arange(1, m + 1, dtype=float) - 0.5) / m
        U, V = np.meshgrid(grid, grid)

        C = self.get_cdf(U, V, [theta, delta])
        pdf = self.get_pdf(U, V, [theta, delta])

        # 4) weight C by pdf, average and scale to get τ
        return float(4.0 * (C * pdf).mean() - 1.0)

    def sample(
            self,
            n: int,
            param=None,
            rng=None,
            eps: float = 1e-12,
            max_iter: int = 40,
    ) -> np.ndarray:
        """
        Generate n i.i.d. samples from the BB3 copula via conditional inversion.

        Parameters
        ----------
        n        : int
            Number of sample pairs.
        param    : sequence-like, optional
            Copula parameters [theta, delta].  If None, uses current parameters.
        rng      : numpy.random.Generator, optional
            Random generator for reproducibility.
        eps      : float
            Clip guard to keep values in (0,1).
        max_iter : int
            Max Newton/bisection iterations inside the inverter.

        Returns
        -------
        ndarray, shape (n, 2)
            Columns are U and V.
        """
        if rng is None:
            rng = default_rng()

        # unpack parameters
        theta, delta = (
            self.get_parameters() if param is None else param
        )

        # 1) draw uniforms
        u = rng.random(n)
        p = rng.random(n)

        # 2) invert ∂C/∂u(u,v) = p  →  v
        v = np.empty_like(u)
        for i in range(n):
            v[i] = brentq(
                lambda vv: self.conditional_cdf_v_given_u(u[i], vv, [theta, delta]) - p[i],
                eps,
                1.0 - eps,
                maxiter=max_iter,
            )

        # 3) clip & pack
        np.clip(u, eps, 1.0 - eps, out=u)
        np.clip(v, eps, 1.0 - eps, out=v)
        return np.column_stack((u, v))

    def LTDC(self, param=None):
        """
        Compute the lower tail dependence coefficient (LTDC) of the copula.

        Args:
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float: LTDC value
        """
        theta, delta = self.get_parameters() if param is None else param
        return 1.0 if theta > 1.0 else 2.0 ** (-1.0 / delta)

    def UTDC(self, param=None):
        """
        Compute the upper tail dependence coefficient (UTDC) of the copula.

        Args:
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float: UTDC value (2 − 2^(1/q)).
        """

        theta, delta = self.get_parameters() if param is None else param
        return 2.0 - 2.0 ** (1.0 / theta)

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """
        Compute the partial derivative ∂C(u,v)/∂u of the copula CDF.

        Args:
            u (float or array-like): First margin in (0,1).
            v (float or array-like): Second margin in (0,1).
            param (Sequence[float], optional): Copula parameters (theta, delta). Defaults to self.get_parameters().

        Returns:
            float or numpy.ndarray: Value of ∂C/∂u at (u,v).
        """

        theta, delta = self.get_parameters() if param is None else param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        s_u = np.expm1(delta * (-np.log(u)) ** theta)
        s_v = np.expm1(delta * (-np.log(v)) ** theta)
        s = s_u + s_v

        # φ_p = -h'(s) · exp(-h(s))
        h1 = self._h_prime(s, param)
        phi_p = -h1 * np.exp(-self._h(s, param))

        # d/du [exp(δ·(-log u)^θ)] = exp(...) * δ·θ·(-log u)^(θ-1)·(1/u)
        inv_u_prime = (
                -delta * theta
                * np.exp(delta * (-np.log(u)) ** theta)
                * (-np.log(u)) ** (theta - 1)
                / u
        )

        deriv = phi_p * inv_u_prime

        deriv = np.nan_to_num(deriv, nan=0.0, neginf=0.0, posinf=np.finfo(float).max)
        deriv = np.where(u <= 1e-9, 0.5 * deriv, deriv)
        return deriv

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """
        Compute the conditional CDF P(V ≤ v | U = u).

        Args:
            u (float or array-like): Conditioning value of U in (0,1).
            v (float or array-like): Value of V in (0,1).
            param (Sequence[float], optional): Copula parameters (d, q). Defaults to self.get_parameters().

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
            param (Sequence[float], optional): Copula parameters (d, q). Defaults to self.get_parameters().

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
            param (Sequence[float], optional): Copula parameters (d, q). Defaults to self.get_parameters().

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
