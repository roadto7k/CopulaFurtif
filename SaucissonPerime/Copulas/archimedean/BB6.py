"""
BB6 Copula implementation following the project coding standard:

Norms:
 1. Use private `_parameters` with public `@property parameters` and validation in setter.
 2. All methods accept `param: np.ndarray = None` defaulting to `self.parameters`.
 3. Docstrings include **Parameters** and **Returns** with types.
 4. Parameter bounds in `bounds_param`; setter enforces them.
 5. Uniform boundary clipping with `eps=1e-12` and `np.clip`.
 6. Document parameter names (`theta`, `delta`) in `__init__` docstring.
"""
import numpy as np
from scipy.optimize import root_scalar
from scipy.integrate import trapz

from SaucissonPerime.Copulas.base import BaseCopula


class BB6Copula(BaseCopula):
    """
    BB6 Copula (Archimedean with generator φ(t) = [-ln(1−(1−t)^θ)]^(1/δ)).

    Parameters
    ----------
    theta : float
        Degree parameter (θ ≥ 1).
    delta : float
        Tail-shape parameter (δ ≥ 1).
    """
    def __init__(self):
        super().__init__()
        self.type = "bb6"
        self.name = "BB6 Copula"
        # θ ≥ 1, δ ≥ 1
        self.bounds_param = [(1.0, None), (1.0, None)]
        self._parameters = np.array([2.0, 2.0])  # [theta, delta]
        self.default_optim_method = "Powell"

    @property
    def parameters(self) -> np.ndarray:
        """
        Get the copula parameters.

        Returns
        -------
        np.ndarray
            Current parameters [theta, delta].
        """
        return self._parameters

    @parameters.setter
    def parameters(self, param: np.ndarray):
        """
        Set and validate copula parameters against bounds_param.

        Parameters
        ----------
        param : array-like
            New parameters [theta, delta].

        Raises
        ------
        ValueError
            If any value is outside its specified bound.
        """
        param = np.asarray(param)
        names = ['theta', 'delta']
        for idx, (lower, upper) in enumerate(self.bounds_param):
            val = param[idx]
            name = names[idx]
            if lower is not None and val < lower:
                raise ValueError(f"Parameter '{name}' must be >= {lower}, got {val}")
        self._parameters = param

    def _phi(self, t: np.ndarray, theta: float, delta: float) -> np.ndarray:
        """
        Archimedean generator φ(t) = [-ln(1-(1-t)^θ)]^(1/δ).

        Parameters
        ----------
        t : float or ndarray
        theta : float
        delta : float

        Returns
        -------
        float or ndarray
        """
        return (-np.log(1.0 - (1.0 - t)**theta))**(1.0 / delta)

    def _phi_prime(self, t: np.ndarray, theta: float, delta: float) -> np.ndarray:
        """
        Derivative φ'(t) of the generator.

        Parameters
        ----------
        t : float or ndarray
        theta : float
        delta : float

        Returns
        -------
        float or ndarray
        """
        g = 1.0 - (1.0 - t)**theta
        gp = theta * (1.0 - t)**(theta - 1)
        L = -np.log(g)
        Lp = -gp / g
        return (1.0 / delta) * L**(1.0 / delta - 1.0) * Lp

    def get_cdf(self, u, v, param: np.ndarray = None):
        """
        Compute BB6 copula CDF: C(u,v) = 1 - [1-w]^(1/θ).

        Parameters
        ----------
        u : float or ndarray
        v : float or ndarray
        param : ndarray, optional
            Copula parameters [theta, delta].

        Returns
        -------
        float or ndarray
            Copula CDF values.
        """
        if param is None:
            param = self.parameters
        theta, delta = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        ubar, vbar = 1.0 - u, 1.0 - v
        x = -np.log(1.0 - ubar**theta)
        y = -np.log(1.0 - vbar**theta)
        sm = x**delta + y**delta
        tem = sm**(1.0 / delta)
        w = np.exp(-tem)
        return 1.0 - (1.0 - w)**(1.0 / theta)

    def get_pdf(self, u, v, param: np.ndarray = None):
        """
        Compute BB6 copula PDF by differentiating C.

        Parameters
        ----------
        u : float or ndarray
        v : float or ndarray
        param : ndarray, optional
            Copula parameters [theta, delta].

        Returns
        -------
        float or ndarray
            Copula PDF values.
        """
        if param is None:
            param = self.parameters
        theta, delta = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        ubar, vbar = 1.0 - u, 1.0 - v
        zu = 1.0 - ubar**theta
        zv = 1.0 - vbar**theta
        x = -np.log(zu)
        y = -np.log(zv)
        xd, yd = x**delta, y**delta
        sm = xd + yd
        tem = sm**(1.0 / delta)
        w = np.exp(-tem)
        prefac = (1.0 - w)**(1.0/theta - 2.0) * w * (tem / sm**2) * (xd / x) * (yd / y)
        bracket = (theta - w) * tem + theta * (delta - 1.0) * (1.0 - w)
        jac = (1.0 - zu) * (1.0 - zv) / (zu * zv * ubar * vbar)
        return prefac * bracket * jac

    def partial_derivative_C_wrt_u(self, u, v, param: np.ndarray = None):
        """
        Compute ∂C/∂u via chain rule.

        Parameters
        ----------
        u : float or ndarray
        v : float or ndarray
        param : ndarray, optional
            Copula parameters [theta, delta].

        Returns
        -------
        float or ndarray
            ∂C/∂u values.
        """
        if param is None:
            param = self.parameters
        theta, delta = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        ubar, vbar = 1.0 - u, 1.0 - v
        x = -np.log(1.0 - ubar**theta)
        y = -np.log(1.0 - vbar**theta)
        xd, yd = x**delta, y**delta
        sm = xd + yd
        tem = sm**(1.0 / delta)
        w = np.exp(-tem)
        C = 1.0 - (1.0 - w)**(1.0 / theta)
        dC_dw = (1.0 / theta) * (1.0 - w)**(1.0 / theta - 1.0)
        dC_dtem = dC_dw * (-w)
        dtem_dsm = (1.0 / delta) * sm**(1.0 / delta - 1.0)
        dsm_dx = delta * x**(delta - 1.0)
        dC_dx = dC_dtem * dtem_dsm * dsm_dx
        dx_du = -theta * ubar**(theta - 1.0) / (1.0 - ubar**theta)
        return dC_dx * dx_du

    def partial_derivative_C_wrt_v(self, u, v, param: np.ndarray = None):
        """
        Compute ∂C/∂v by symmetry (swap u/v).

        Returns same shape/type as ∂C/∂u.
        """
        return self.partial_derivative_C_wrt_u(v, u, param)

    def conditional_cdf_v_given_u(self, u, v, param: np.ndarray = None):
        """
        Compute P(V ≤ v | U = u) = ∂C/∂u(u,v) / ∂C/∂u(u,1).

        Returns
        -------
        float or ndarray
        """
        num = self.partial_derivative_C_wrt_u(u, v, param)
        den = self.partial_derivative_C_wrt_u(u, 1.0, param)
        return num / den

    def conditional_cdf_u_given_v(self, u, v, param: np.ndarray = None):
        """
        Compute P(U ≤ u | V = v) = ∂C/∂v(u,v) / ∂C/∂v(1,v).

        Returns
        -------
        float or ndarray
        """
        num = self.partial_derivative_C_wrt_v(u, v, param)
        den = self.partial_derivative_C_wrt_v(1.0, v, param)
        return num / den

    def sample(self, n: int, param: np.ndarray = None) -> np.ndarray:
        """
        Generate samples via conditional inversion.

        Parameters
        ----------
        n : int
            Number of samples.
        param : ndarray, optional
            Copula parameters [theta, delta].

        Returns
        -------
        np.ndarray
            Shape (n,2) array of samples.
        """
        if param is None:
            param = self.parameters
        eps = 1e-6
        u = np.random.rand(n)
        v = np.empty(n)
        for i in range(n):
            p = np.random.rand()
            root = root_scalar(
                lambda vi: self.conditional_cdf_v_given_u(u[i], vi, param) - p,
                bracket=[eps, 1 - eps], method='bisect', xtol=1e-6
            )
            v[i] = root.root
        return np.column_stack((u, v))

    def kendall_tau(self, param: np.ndarray = None, n: int = 1001) -> float:
        """
        Estimate Kendall's tau by numerical integration of φ/φ'.

        Parameters
        ----------
        param : ndarray, optional
            Copula parameters [theta, delta].
        n : int
            Number of grid points for trapz.

        Returns
        -------
        float
            Estimated Kendall's tau.
        """
        if param is None:
            param = self.parameters
        theta, delta = param
        t = np.linspace(0.0, 1.0, n)
        t = t[1:-1]
        phi_vals = self._phi(t, theta, delta)
        phi_p_vals = self._phi_prime(t, theta, delta)
        integrand = phi_vals / phi_p_vals
        integral = trapz(integrand, t)
        return 1.0 + 4.0 * integral

    def LTDC(self, param: np.ndarray = None) -> float:
        """
        Lower-tail dependence λ_L = 0 for BB6.

        Returns
        -------
        float
            0.0
        """
        return 0.0

    def UTDC(self, param: np.ndarray = None) -> float:
        """
        Upper-tail dependence λ_U = 2 - 2^(1/δ).

        Returns
        -------
        float
            Upper-tail dependence.
        """
        if param is None:
            param = self.parameters
        delta = param[1]
        return 2.0 - 2.0 ** (1.0 / delta)
