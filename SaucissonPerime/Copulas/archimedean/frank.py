import numpy as np
from scipy.stats import uniform
from scipy.integrate import quad

from SaucissonPerime.Copulas.base import BaseCopula


def debye1(t: float) -> float:
    """
    Debye function of order 1: D1(theta) = (1/theta) * \int_0^theta t/(e^t - 1) dt
    """
    return t / (np.exp(t) - 1)


class FrankCopula(BaseCopula):
    """
    Frank Copula class (Archimedean copula).

    Parameters
    ----------
    theta : float
        Copula parameter; real-valued except 0 (θ ≠ 0).

    Attributes
    ----------
    type : str
        Copula identifier ('frank').
    name : str
        Human-readable name.
    bounds_param : list of tuple
        Parameter bounds for optimization: θ ∈ (-∞, -ε] ∪ [ε, ∞).
    parameters : np.ndarray
        Copula parameter [θ]. Enforced |θ| ≥ ε.
    default_optim_method : str
        Recommended optimizer ('SLSQP').
    """

    def __init__(self):
        super().__init__()
        self.type = 'frank'
        self.name = 'Frank Copula'
        eps = 1e-6
        # Exclude zero: enforce |theta| >= eps
        self.bounds_param = [(-np.inf, -eps), (eps, np.inf)]  # two intervals: negative and positive domains
        self._parameters = np.array([2.0])  # initial guess for theta
        self.default_optim_method = 'SLSQP'

    @property
    def parameters(self) -> np.ndarray:
        """
        Return current copula parameter [θ].
        """
        return self._parameters

    @parameters.setter
    def parameters(self, param: np.ndarray):
        """
        Set copula parameter [θ], validating |θ| >= eps.
        """
        theta = float(param[0])
        eps = 1e-6
        if abs(theta) < eps:
            raise ValueError(f"Frank copula parameter theta must satisfy |theta| >= {eps} (non-zero).")
        self._parameters = np.array([theta])

    def get_cdf(self, u, v, param: np.ndarray = None):
        """
        Compute Frank copula CDF C(u,v).

        Parameters
        ----------
        u, v : float or array-like
            Pseudo-observations in (0,1).
        param : ndarray, optional
            Copula parameter [θ]. If None, uses self.parameters.

        Returns
        -------
        C : float or ndarray
            Copula CDF at (u, v).
        """
        if param is None:
            theta = self.parameters[0]
        else:
            theta = float(param[0])
        # independence case for small |theta|
        if abs(theta) < 1e-8:
            return u * v

        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        exp_neg_t = np.exp(-theta)
        a = (np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1)
        return -1.0 / theta * np.log(1 + a / (exp_neg_t - 1))

    def get_pdf(self, u, v, param: np.ndarray = None):
        """
        Compute Frank copula PDF c(u,v).

        Parameters
        ----------
        u, v : float or array-like
            Pseudo-observations in (0,1).
        param : ndarray, optional
            Copula parameter [θ]. If None, uses self.parameters.

        Returns
        -------
        c : float or ndarray
            Copula PDF at (u, v).
        """
        if param is None:
            theta = self.parameters[0]
        else:
            theta = float(param[0])
        # independence case
        if abs(theta) < 1e-8:
            return np.ones_like(u)

        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        exp_neg_t = np.exp(-theta)
        term1 = theta * (1 - exp_neg_t) * np.exp(-theta * (u + v))
        term2 = (1 - exp_neg_t - (1 - np.exp(-theta * u)) * (1 - np.exp(-theta * v)))**2
        return term1 / term2

    def kendall_tau(self, param: np.ndarray = None) -> float:
        """
        Compute Kendall's tau: τ = 1 - (4/θ)(1 - D1(θ)).
        """
        if param is None:
            theta = self.parameters[0]
        else:
            theta = float(param[0])
        if abs(theta) < 1e-8:
            return 0.0
        debye_val, _ = quad(debye1, 0.0, theta)
        D1 = debye_val / theta
        return 1.0 - (4.0 / theta) * (1.0 - D1)

    def sample(self, n: int, param: np.ndarray = None) -> np.ndarray:
        """
        Generate samples via conditional inversion.

        Returns
        -------
        samples : ndarray of shape (n,2)
        """
        if param is None:
            theta = self.parameters[0]
        else:
            theta = float(param[0])
        # independent case
        if abs(theta) < 1e-8:
            u = uniform.rvs(size=n)
            v = uniform.rvs(size=n)
            return np.column_stack((u, v))

        u = uniform.rvs(size=n)
        w = uniform.rvs(size=n)
        exp_neg_t = np.exp(-theta)
        exp_neg_t_u = np.exp(-theta * u)
        numerator = np.log(1 - w * (1 - exp_neg_t))
        denominator = exp_neg_t_u - 1
        denominator = np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
        v = -1.0 / theta * np.log(1 + numerator / denominator)
        return np.column_stack((u, v))

    def LTDC(self, param: np.ndarray = None) -> float:
        """
        Lower-tail dependence (Frank): 0.
        """
        return 0.0

    def UTDC(self, param: np.ndarray = None) -> float:
        """
        Upper-tail dependence (Frank): 0.
        """
        return 0.0

    def partial_derivative_C_wrt_v(self, u, v, param: np.ndarray = None):
        """
        Compute ∂C/∂v = exp(-θv)*(exp(-θu)-1)/[(exp(-θ)-1)+(exp(-θu)-1)(exp(-θv)-1)].
        """
        if param is None:
            theta = self.parameters[0]
        else:
            theta = float(param[0])
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        exp_neg_t = np.exp(-theta)
        numer = np.exp(-theta * v) * (np.exp(-theta * u) - 1)
        denom = (exp_neg_t - 1) + (np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1)
        return numer / denom

    def partial_derivative_C_wrt_u(self, u, v, param: np.ndarray = None):
        """
        Compute ∂C/∂u (symmetric to ∂C/∂v).
        """
        return self.partial_derivative_C_wrt_v(v, u, param)

    def conditional_cdf_u_given_v(self, u, v, param: np.ndarray = None):
        """
        P(U ≤ u | V = v) = ∂C/∂v.
        """
        return self.partial_derivative_C_wrt_v(u, v, param)

    def conditional_cdf_v_given_u(self, u, v, param: np.ndarray = None):
        """
        P(V ≤ v | U = u) = ∂C/∂u.
        """
        return self.partial_derivative_C_wrt_u(u, v, param)
