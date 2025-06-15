"""
BB5 Copula implementation following the project coding standard:

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
from scipy.integrate import quad

from SaucissonPerime.Copulas.base import BaseCopula


class BB5Copula(BaseCopula):
    """
    BB5 Copula (Joe’s two-parameter extreme-value copula).

    Parameters
    ----------
    theta : float
        Dependence strength parameter (θ ≥ 1).
    delta : float
        Tail dependence parameter (δ > 0).
    """
    def __init__(self):
        super().__init__()
        self.type = "bb5"
        self.name = "BB5 Copula"
        # θ ≥ 1, δ > 0
        self.bounds_param = [(1.0, None), (1e-6, None)]
        self._parameters = np.array([1.0, 1.0])  # [theta, delta]
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

    def get_cdf(self, u, v, param: np.ndarray = None):
        """
        Compute the BB5 copula CDF:
            C(u,v) = exp(-g),
            where x=-ln u, y=-ln v,
                  w = x^θ + y^θ - (x^{-δθ} + y^{-δθ})^{-1/δ},
                  g = w^{1/θ}.

        Parameters
        ----------
        u : float or ndarray
            Pseudo-observations in (0,1).
        v : float or ndarray
            Pseudo-observations in (0,1).
        param : ndarray, optional
            Copula parameters [theta, delta].

        Returns
        -------
        float or ndarray
            Copula CDF value(s).
        """
        if param is None:
            param = self.parameters
        theta, delta = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        x = -np.log(u)
        y = -np.log(v)
        xt = x**theta
        yt = y**theta
        S = x**(-delta*theta) + y**(-delta*theta)
        xyp = S**(-1.0/delta)
        w = xt + yt - xyp
        g = w**(1.0/theta)
        return np.exp(-g)

    def get_pdf(self, u, v, param: np.ndarray = None):
        """
        Compute the BB5 copula PDF via exact mixed second derivative of CDF.

        Parameters
        ----------
        u : float or ndarray
            Pseudo-observations in (0,1).
        v : float or ndarray
            Pseudo-observations in (0,1).
        param : ndarray, optional
            Copula parameters [theta, delta].

        Returns
        -------
        float or ndarray
            Copula PDF value(s).
        """
        if param is None:
            param = self.parameters
        theta, delta = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        x = -np.log(u)
        y = -np.log(v)
        xt = x**theta
        yt = y**theta
        xdt = x**(-delta*theta)
        ydt = y**(-delta*theta)
        S = xdt + ydt
        xyp = S**(-1.0/delta)
        w = xt + yt - xyp
        g = w**(1.0/theta)
        C = np.exp(-g)
        # derivatives
        dx = -1.0/u
        dxt_du = theta * x**(theta-1) * dx
        dxdt_du = -delta*theta * x**(-delta*theta-1) * dx
        dw_du = dxt_du - (-(1.0/delta)*S**(-1.0/delta-1)*dxdt_du)
        dyt_dv = theta * y**(theta-1) * (-1.0/v)
        dydt_dv = -delta*theta * y**(-delta*theta-1) * (-1.0/v)
        dw_dv = dyt_dv - (-(1.0/delta)*S**(-1.0/delta-1)*dydt_dv)
        # cross-derivative
        w_uv = -( (1.0/delta)*(1.0/delta+1.0) * S**(-1.0/delta-2) * dxdt_du * dydt_dv )
        dg_dw = (1.0/theta) * w**(1.0/theta-1)
        d2g_dw2 = (1.0/theta)*(1.0/theta-1.0)*w**(1.0/theta-2)
        term1 = -C * dg_dw * w_uv
        term2 = -C * d2g_dw2 * dw_du * dw_dv
        term3 = C * (dg_dw**2) * dw_du * dw_dv
        return term1 + term2 + term3

    def kendall_tau(self, param: np.ndarray = None) -> float:
        """
        Compute Kendall's tau via Pickands function:
            τ = 1 - 4 ∫0^1 A(t) dt,
            A(t) = [t^θ + (1-t)^θ - (t^{-δθ}+(1-t)^{-δθ})^{-1/δ}]^{1/θ}.

        Parameters
        ----------
        param : ndarray, optional
            Copula parameters [theta, delta].

        Returns
        -------
        float
            Kendall's tau.
        """
        if param is None:
            param = self.parameters
        theta, delta = param
        def A(t):
            return (t**theta + (1-t)**theta - (t**(-delta*theta)+(1-t)**(-delta*theta))**(-1.0/delta))**(1.0/theta)
        integral, _ = quad(A, 0.0, 1.0)
        return 1.0 - 4.0 * integral

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
            Partial derivative ∂C/∂u.
        """
        if param is None:
            param = self.parameters
        theta, delta = param
        eps = 1e-12
        u = np.clip(u, eps, 1-eps)
        v = np.clip(v, eps, 1-eps)
        x = -np.log(u)
        y = -np.log(v)
        S = x**(-delta*theta) + y**(-delta*theta)
        w = x**theta + y**theta - S**(-1.0/delta)
        C = np.exp(-w**(1.0/theta))
        dg_dw = (1.0/theta)*w**(1.0/theta-1)
        dxt_du = theta*x**(theta-1)*(-1.0/u)
        dxdt_du = -delta*theta*x**(-delta*theta-1)*(-1.0/u)
        dw_du = dxt_du - (-(1.0/delta)*S**(-1.0/delta-1)*dxdt_du)
        return -C * dg_dw * dw_du

    def partial_derivative_C_wrt_v(self, u, v, param: np.ndarray = None):
        """
        Compute ∂C/∂v by symmetry (swap u/v).
        """
        if param is None:
            param = self.parameters
        return self.partial_derivative_C_wrt_u(v, u, param)

    def conditional_cdf_v_given_u(self, u, v, param: np.ndarray = None):
        """
        Compute P(V ≤ v | U = u) = ∂C/∂u(u,v)/∂C/∂u(u,1).
        """
        if param is None:
            param = self.parameters
        num = self.partial_derivative_C_wrt_u(u, v, param)
        den = self.partial_derivative_C_wrt_u(u, 1.0, param)
        return num / den

    def conditional_cdf_u_given_v(self, u, v, param: np.ndarray = None):
        """
        Compute P(U ≤ u | V = v) = ∂C/∂v(u,v)/∂C/∂v(1,v).
        """
        if param is None:
            param = self.parameters
        num = self.partial_derivative_C_wrt_v(u, v, param)
        den = self.partial_derivative_C_wrt_v(1.0, v, param)
        return num / den

    def sample(self, n: int, param: np.ndarray = None):
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
        u = np.random.rand(n)
        v = np.empty(n)
        for i in range(n):
            p = np.random.rand()
            sol = root_scalar(
                lambda vv: self.conditional_cdf_v_given_u(u[i], vv, param) - p,
                bracket=[1e-6,1-1e-6], method='bisect', xtol=1e-6
            )
            v[i] = sol.root
        return np.column_stack((u, v))
