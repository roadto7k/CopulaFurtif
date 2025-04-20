import numpy as np
from scipy.optimize import root_scalar

from Service.Copulas.base import BaseCopula


class AMHCopula(BaseCopula):
    """
    Ali–Mikhail–Haq (AMH) Copula class.

    Attributes
    ----------
    family : str
        Identifier for the copula family, "amh".
    name : str
        Human-readable name for output/logging.
    bounds_param : list of tuple
        Bounds for the copula parameter theta in (-1,1).
    parameters : np.ndarray
        Copula parameter [theta].
    default_optim_method : str
        Optimizer method.
    """

    def __init__(self):
        super().__init__()
        self.type = "amh"
        self.name = "Ali–Mikhail–Haq Copula"
        self.bounds_param = [(-0.999999, 0.999999)]
        self._parameters = np.array([0.0])  # [theta]
        self.default_optim_method = "SLSQP"

    @property
    def parameters(self) -> np.ndarray:
        return self._parameters

    @parameters.setter
    def parameters(self, param: np.ndarray):
        """
        Set copula parameters with validation against bounds_param.

        Parameters
        ----------
        param : ndarray
            New copula parameter values.

        Raises
        ------
        ValueError
            If any parameter is outside its specified bounds.
        """
        param = np.asarray(param)
        # Validate each parameter against bounds_param
        for idx, bound in enumerate(self.bounds_param):
            lower, upper = bound
            value = param[idx]
            if lower is not None and value <= lower:
                raise ValueError(f"Parameter at index {idx} must be > {lower}, got {value}")
            if upper is not None and value >= upper:
                raise ValueError(f"Parameter at index {idx} must be < {upper}, got {value}")
        self._parameters = param

    def get_cdf(self, u, v, param: np.ndarray = None):
        """
        Copula CDF: C(u,v) = u*v / [1 - theta*(1-u)*(1-v)].

        Parameters
        ----------
        u : float or array-like
            Pseudo-observations in (0,1).
        v : float or array-like
            Pseudo-observations in (0,1).
        param : ndarray, optional
            Copula parameter [theta]. If None, uses self.parameters.

        Returns
        -------
        C : float or ndarray
            CDF value(s).
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        denom = 1.0 - theta * (1.0 - u) * (1.0 - v)
        return (u * v) / denom

    def get_pdf(self, u, v, param: np.ndarray = None):
        """
        Copula PDF: c(u,v) = [1 + theta*((1+u)*(1+v)-3) + theta^2*(1-u)*(1-v)]
                      / [1 - theta*(1-u)*(1-v)]^3.

        Parameters
        ----------
        u : float or array-like
            Pseudo-observations in (0,1).
        v : float or array-like
            Pseudo-observations in (0,1).
        param : ndarray, optional
            Copula parameter [theta]. If None, uses self.parameters.

        Returns
        -------
        c : float or ndarray
            PDF value(s).
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        one_mu = 1.0 - u
        one_mv = 1.0 - v
        D = 1.0 - theta * one_mu * one_mv
        num = (1.0
               + theta * ((1.0 + u) * (1.0 + v) - 3.0)
               + theta**2 * one_mu * one_mv)
        return num / (D ** 3)

    def kendall_tau(self, param: np.ndarray = None) -> float:
        """
        Kendall's tau: tau(theta) = 1 - 2*((1-theta)^2*ln(1-theta) + theta)/(3*theta^2).

        Parameters
        ----------
        param : ndarray, optional
            Copula parameter [theta]. If None, uses self.parameters.

        Returns
        -------
        tau : float
            Kendall's tau.
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        if abs(theta) < 1e-8:
            return 0.0
        return 1.0 - 2.0 * (((1 - theta)**2 * np.log(1 - theta) + theta) / (3 * theta**2))

    def partial_derivative_C_wrt_u(self, u, v, param: np.ndarray = None):
        """
        Partial derivative ∂C/∂u = v*(1 - theta*(1-v)) / [1 - theta*(1-u)*(1-v)]^2.

        Parameters
        ----------
        u : float or array-like
            Pseudo-observations.
        v : float or array-like
            Pseudo-observations.
        param : ndarray, optional
            Copula parameter [theta]. If None, uses self.parameters.

        Returns
        -------
        dCdu : float or ndarray
            Partial derivative values.
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        D = 1.0 - theta * (1.0 - u) * (1.0 - v)
        return v * (1.0 - theta * (1.0 - v)) / (D ** 2)

    def partial_derivative_C_wrt_v(self, u, v, param: np.ndarray = None):
        """
        Partial derivative ∂C/∂v = u*(1 - theta*(1-u)) / [1 - theta*(1-u)*(1-v)]^2.

        Parameters
        ----------
        u : float or array-like
            Pseudo-observations.
        v : float or array-like
            Pseudo-observations.
        param : ndarray, optional
            Copula parameter [theta]. If None, uses self.parameters.

        Returns
        -------
        dCdv : float or ndarray
            Partial derivative values.
        """
        if param is None:
            param = self.parameters
        theta = param[0]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        D = 1.0 - theta * (1.0 - u) * (1.0 - v)
        return u * (1.0 - theta * (1.0 - u)) / (D ** 2)

    def conditional_cdf_v_given_u(self, u, v, param: np.ndarray = None):
        """
        Conditional CDF P(V <= v | U = u) = ∂C/∂u(u,v).

        Parameters
        ----------
        u : float or array-like
        v : float or array-like
        param : ndarray, optional
            Copula parameter [theta]. If None, uses self.parameters.

        Returns
        -------
        P : float or ndarray
            Conditional CDF values.
        """
        if param is None:
            param = self.parameters
        return self.partial_derivative_C_wrt_u(u, v, param)

    def conditional_cdf_u_given_v(self, u, v, param: np.ndarray = None):
        """
        Conditional CDF P(U <= u | V = v) = ∂C/∂v(u,v).

        Parameters
        ----------
        u : float or array-like
        v : float or array-like
        param : ndarray, optional
            Copula parameter [theta]. If None, uses self.parameters.

        Returns
        -------
        P : float or ndarray
            Conditional CDF values.
        """
        if param is None:
            param = self.parameters
        return self.partial_derivative_C_wrt_v(u, v, param)

    def sample(self, n, param: np.ndarray = None) -> np.ndarray:
        """
        Generate samples via conditional inversion:
          1. Draw u ~ Uniform(0,1)
          2. Draw p ~ Uniform(0,1)
          3. Solve ∂C/∂u(u,v) = p for v by bisection.

        Parameters
        ----------
        n : int
            Number of samples.
        param : ndarray, optional
            Copula parameter [theta]. If None, uses self.parameters.

        Returns
        -------
        samples : ndarray, shape (n,2)
            (u,v) samples.
        """
        if param is None:
            param = self.parameters
        eps = 1e-6
        u = np.random.rand(n)
        v = np.empty(n)
        for i in range(n):
            p = np.random.rand()
            sol = root_scalar(
                lambda vv: self.partial_derivative_C_wrt_u(u[i], vv, param) - p,
                bracket=[eps, 1 - eps], method="bisect", xtol=1e-6
            )
            v[i] = sol.root
        return np.column_stack((u, v))

    def LTDC(self, param: np.ndarray = None) -> float:
        """
        Lower-tail dependence λ_L = 0 for AMH.

        Returns
        -------
        0.0
        """
        return 0.0

    def UTDC(self, param: np.ndarray = None) -> float:
        """
        Upper-tail dependence λ_U = 0 for AMH.

        Returns
        -------
        0.0
        """
        return 0.0
