import numpy as np
from scipy.stats import uniform
from SaucissonPerime.Copulas.base import BaseCopula


class GumbelCopula(BaseCopula):
    """
    Gumbel Copula (Archimedean)

    Parameters
    ----------
    theta : float
        Copula parameter (θ ≥ 1) controlling upper-tail dependence.

    Attributes
    ----------
    type : str
        Identifier for the copula family ('gumbel').
    name : str
        Human-readable name.
    param_names : list of str
        Names of copula parameters in order.
    bounds_param : list of tuple
        Bounds for each parameter: [(1+eps, None)].
    _parameters : np.ndarray
        Internal parameter storage [theta].
    default_optim_method : str
        Default optimizer for parameter fitting.
    """

    def __init__(self):
        super().__init__()
        self.type = "gumbel"
        self.name = "Gumbel Copula"
        # Theta must be >= 1
        self.param_names = ["theta"]
        self.bounds_param = [(1.0 + 1e-6, None)]
        self._parameters = np.array([1.5])  # [theta]
        self.default_optim_method = "SLSQP"

    @property
    def parameters(self) -> np.ndarray:
        """Current copula parameters [theta]."""
        return self._parameters

    @parameters.setter
    def parameters(self, param: np.ndarray):
        """Validate and set copula parameters."""
        theta = float(param[0])
        lower, upper = self.bounds_param[0]
        if theta < lower or (upper is not None and theta > upper):
            raise ValueError(f"Parameter 'theta' must satisfy {self.bounds_param[0]}, got {theta}.")
        self._parameters = np.array([theta])

    def get_cdf(self, u: float, v: float, param: np.ndarray=None) -> float:
        """
        CDF of the Gumbel copula: C(u,v) = exp(-[(-log u)^theta + (-log v)^theta]^(1/theta)).

        Parameters
        ----------
        u : float
            First uniform input in (0,1).
        v : float
            Second uniform input in (0,1).
        param : ndarray, optional
            [theta], default to self.parameters.

        Returns
        -------
        float
            Copula CDF value.
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        s_u = -np.log(u)
        s_v = -np.log(v)
        S = s_u**theta + s_v**theta
        return np.exp(-S**(1.0/theta))

    def get_pdf(self, u: float, v: float, param: np.ndarray=None) -> float:
        """
        PDF of the Gumbel copula.

        c(u,v) = C(u,v)/(u v) * S^(2/theta - 2) * (s_u s_v)^(theta-1)
                 * [theta + (theta-1) S^(-1/theta)].

        Parameters
        ----------
        u, v : float
            Uniform inputs in (0,1).
        param : ndarray, optional
            [theta], default to self.parameters.

        Returns
        -------
        float
            Copula density value.
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        s_u = -np.log(u)
        s_v = -np.log(v)
        S = s_u**theta + s_v**theta
        C = np.exp(-S**(1.0/theta))

        term1 = C / (u * v)
        term2 = S**(-2.0 + 2.0/theta)
        term3 = (s_u * s_v)**(theta - 1.0)
        term4 = theta + (theta - 1.0) * S**(-1.0/theta)
        return term1 * term2 * term3 * term4

    def kendall_tau(self, param: np.ndarray=None) -> float:
        """
        Kendall's tau for Gumbel: 1 - 1/theta.
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        return 1.0 - 1.0/theta

    def sample(self, n: int, param: np.ndarray=None) -> np.ndarray:
        """
        Approximate sampling via Marshall–Olkin method.

        Parameters
        ----------
        n : int
            Number of samples.
        param : ndarray, optional
            [theta], default to self.parameters.

        Returns
        -------
        ndarray shape (n,2)
            Pseudo-observations.
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        if theta < 1.0:
            raise ValueError("Theta must be ≥ 1 for Gumbel copula.")
        if theta == 1.0:
            return np.random.rand(n, 2)

        alpha = 1.0/theta
        E = np.random.exponential(size=n)
        V = np.random.uniform(0, np.pi, size=n)
        W = np.random.exponential(size=n)

        S1 = (np.sin(alpha * V) / np.cos(V)**(1.0/alpha)) * \
             (np.cos((1 - alpha) * V) / W)**((1 - alpha)/alpha)
        S2 = (np.sin(alpha * V) / np.cos(V)**(1.0/alpha)) * \
             (np.cos((1 - alpha) * V) / np.random.exponential(size=n))**((1 - alpha)/alpha)

        u = np.exp(-S1**(1.0/theta) / E)
        v = np.exp(-S2**(1.0/theta) / E)
        return np.column_stack((u, v))

    def LTDC(self, param: np.ndarray=None) -> float:
        """Lower-tail dependence (0)."""
        return 0.0

    def UTDC(self, param: np.ndarray=None) -> float:
        """Upper-tail dependence: 2 - 2^(1/theta)."""
        if param is None:
            param = self.parameters
        theta = param[0]
        return 2.0 - 2.0**(1.0/theta)

    def partial_derivative_C_wrt_v(self, u: float, v: float, param: np.ndarray=None) -> float:
        """
        ∂C(u,v)/∂v for conditional distributions.
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        s_u = -np.log(u)
        s_v = -np.log(v)
        S = s_u**theta + s_v**theta
        C = np.exp(-S**(1.0/theta))
        return (C / v) * S**(1.0/theta - 1.0) * s_v**(theta - 1.0)

    def partial_derivative_C_wrt_u(self, u: float, v: float, param: np.ndarray=None) -> float:
        """∂C(u,v)/∂u (symmetric)."""
        if param is None:
            param = self.parameters
        theta = param[0]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        s_u = -np.log(u)
        s_v = -np.log(v)
        S = s_u**theta + s_v**theta
        C = np.exp(-S**(1.0/theta))
        return (C / u) * S**(1.0/theta - 1.0) * s_u**(theta - 1.0)

    def conditional_cdf_u_given_v(self, u: float, v: float, param: np.ndarray=None) -> float:
        """P(U ≤ u | V=v) = ∂C/∂v normalized."""
        if param is None:
            param = self.parameters
        num = self.partial_derivative_C_wrt_v(u, v, param)
        den = self.partial_derivative_C_wrt_v(1.0, v, param)
        return num / max(den, 1e-14)

    def conditional_cdf_v_given_u(self, u: float, v: float, param: np.ndarray=None) -> float:
        """P(V ≤ v | U=u) = ∂C/∂u normalized."""
        if param is None:
            param = self.parameters
        num = self.partial_derivative_C_wrt_u(u, v, param)
        den = self.partial_derivative_C_wrt_u(u, 1.0, param)
        return num / max(den, 1e-14)
