import numpy as np
from scipy.optimize import root_scalar
from scipy.integrate import quad

from Service.Copulas.base import BaseCopula


class BB5Copula(BaseCopula):
    """
    BB5 (Joe’s two-parameter extreme-value) Copula class

    Attributes
    ----------
    family : str
        Identifier for the copula family. Here, "bb5".
    name : str
        Human-readable name for output/logging.
    bounds_param : list of tuple
        Bounds for the copula parameters (θ ≥ 1, δ > 0).
    parameters : np.ndarray
        Initial guess for the copula parameters [θ, δ].
    default_optim_method : str
        Default optimizer to use.
    """

    def __init__(self):
        super().__init__()
        self.type = "bb5"
        self.name = "BB5 Copula"
        # θ ≥ 1, δ > 0
        self.bounds_param = [(1.0, None), (0.0, None)]
        self.parameters = np.array([1.0, 1.0])
        self.default_optim_method = "SLSQP"

    def get_cdf(self, u, v, param):
        """
        C(u,v) = exp(-g), where
          x = -log(u), y = -log(v)
          w = x^θ + y^θ - ( x^{-δθ} + y^{-δθ} )^{-1/δ}
          g = w^{1/θ}

        Returns
        -------
        float or np.ndarray
            Copula CDF at (u, v).
        """
        θ, δ = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        x = -np.log(u)
        y = -np.log(v)
        xt = x**θ
        yt = y**θ
        xdt = x**(-δ * θ)
        ydt = y**(-δ * θ)
        S = xdt + ydt
        xyp = S**(-1.0/δ)
        w = xt + yt - xyp
        g = w**(1.0/θ)
        return np.exp(-g)

    def get_pdf(self, u, v, param):
        """
        PDF c(u,v) via direct differentiation of C:
          ∂C/∂u and ∂C/∂v combined into closed-form using intermediate vars.

        Returns
        -------
        float or np.ndarray
            Copula PDF at (u, v).
        """
        θ, δ = param
        # reuse get_cdf intermediates
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        x = -np.log(u)
        y = -np.log(v)
        xt = x**θ
        yt = y**θ
        xdt = x**(-δ * θ)
        ydt = y**(-δ * θ)
        S = xdt + ydt
        xyp = S**(-1.0/δ)
        w = xt + yt - xyp
        g = w**(1.0/θ)
        C = np.exp(-g)

        # derivatives of w
        dxt_dx = θ * x**(θ - 1)
        dx_dt = -1.0 / u
        dxt_du = dxt_dx * dx_dt

        dxdt_dx = -δ * θ * x**(-δ * θ - 1)
        dxdt_du = dxdt_dx * dx_dt

        dS_du = dxdt_du
        dxyp_dS = -(1.0/δ) * S**(-1.0/δ - 1)
        dxyp_du = dxyp_dS * dS_du

        dw_du = dxt_du - dxyp_du

        # symmetric for v
        dyt_dy = θ * y**(θ - 1)
        dy_dv = -1.0 / v
        dyt_dv = dyt_dy * dy_dv

        dydt_dy = -δ * θ * y**(-δ * θ - 1)
        dydt_dv = dydt_dy * dy_dv

        dS_dv = dydt_dv
        dxyp_dv = dxyp_dS * dS_dv

        dw_dv = dyt_dv - dxyp_dv

        # derivative of g = w^{1/θ}
        dg_dw = (1.0/θ) * w**(1.0/θ - 1)

        # chain rule
        dC_du = -C * dg_dw * dw_du
        dC_dv = -C * dg_dw * dw_dv

        # PDF is ∂²C / (∂u ∂v)
        # differentiate dC_du w.r.t v
        # here we approximate by mixed partial symmetry: ∂²C/∂u∂v = ∂/∂v(dC/du)
        # explicitly:
        d2C = -(
            dg_dw * dw_du * dC_dv / C  # from -C*d(g(w))/du * dC_dv piece
            + C * dg_dw * (
                (1.0/θ - 1) * w**(1.0/θ - 2) * dw_du * dw_dv
                + w**(1.0/θ - 1) * (dw_du * (-(1.0/δ+1)*dx_dt * dxdt_du / S) + dw_dv * 0 )
            )
        )
        return d2C

    def partial_derivative_C_wrt_u(self, u, v, param):
        """
        ∂C/∂u via chain rule:
          dC/du = -C * (1/θ) w^{1/θ -1} * dw/du
        """
        θ, δ = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        x = -np.log(u)
        y = -np.log(v)
        xt = x**θ
        xdt = x**(-δ * θ)
        ydt = y**(-δ * θ)
        S = xdt + ydt
        xyp = S**(-1.0/δ)
        w = xt + ydt - xyp

        # dw/du
        dxt_dx = θ * x**(θ - 1)
        dx_du = -1.0 / u
        dxt_du = dxt_dx * dx_du

        dxdt_dx = -δ * θ * x**(-δ * θ - 1)
        dxdt_du = dxdt_dx * dx_du

        dxyp_dS = -(1.0/δ) * S**(-1.0/δ - 1)
        dxyp_du = dxyp_dS * dxdt_du

        dw_du = dxt_du - dxyp_du

        g = w**(1.0/θ)
        C = np.exp(-g)
        dg_dw = (1.0/θ) * w**(1.0/θ - 1)

        return -C * dg_dw * dw_du

    def partial_derivative_C_wrt_v(self, u, v, param):
        """
        ∂C/∂v via symmetry of ∂C/∂u (swap u<->v).
        """
        return self.partial_derivative_C_wrt_u(v, u, param)

    def conditional_cdf_v_given_u(self, u, v, param):
        """
        P(V ≤ v | U = u) = ∂C/∂u(u, v) / ∂C/∂u(u, 1).
        """
        num = self.partial_derivative_C_wrt_u(u, v, param)
        denom = self.partial_derivative_C_wrt_u(u, 1.0, param)
        return num / denom

    def conditional_cdf_u_given_v(self, u, v, param):
        """
        P(U ≤ u | V = v) = ∂C/∂v(u, v) / ∂C/∂v(1, v).
        """
        num = self.partial_derivative_C_wrt_v(u, v, param)
        denom = self.partial_derivative_C_wrt_v(1.0, v, param)
        return num / denom

    def sample(self, n, param):
        """
        Generate n samples via conditional inversion:
          u ~ Uniform(0,1)
          v solves C_{2|1}(v|u) = U'.
        """
        θ, δ = param
        eps = 1e-6
        u = np.random.rand(n)
        v = np.empty(n)
        for i in range(n):
            p = np.random.rand()
            sol = root_scalar(
                lambda vi: self.conditional_cdf_v_given_u(u[i], vi, param) - p,
                bracket=[eps, 1 - eps],
                method='bisect', xtol=1e-6
            )
            v[i] = sol.root
        return np.column_stack((u, v))

    def kendall_tau(self, param):
        """
        τ = 1 − 4 ∫₀¹ A(t) dt, where A(t) is the Pickands function:
          A(t) = [t^θ + (1−t)^θ − (t^{−δθ} + (1−t)^{−δθ})^{−1/δ}]^{1/θ}
        """
        θ, δ = param

        def A(t):
            return (
                t**θ
                + (1 - t)**θ
                - (t**(-δ*θ) + (1 - t)**(-δ*θ))**(-1.0/δ)
            )**(1.0/θ)

        integral, _ = quad(A, 0.0, 1.0)
        return 1.0 - 4.0 * integral

    def LTDC(self, param):
        """
        Lower tail dependence λ_L = 0 for BB5.
        """
        return 0.0

    def UTDC(self, param):
        """
        Upper tail dependence λ_U = 2 − (2 − 2^{−1/δ})^{1/θ}.
        """
        θ, δ = param
        return 2.0 - (2.0 - 2.0**(-1.0/δ))**(1.0/θ)
