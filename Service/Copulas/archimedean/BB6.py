import numpy as np
from scipy.optimize import root_scalar
from scipy.integrate import quad

from Service.Copulas.base import BaseCopula


class BB6Copula(BaseCopula):
    """
    BB6 Copula class (Archimedean with generator φ(t) = [-ln(1 − (1−t)^θ)]^{1/δ})

    Attributes
    ----------
    family : str
        Identifier for the copula family. Here, "bb6".
    name : str
        Human-readable name for output/logging.
    bounds_param : list of tuple
        Bounds for the copula parameters (θ ≥ 1, δ ≥ 1).
    parameters : np.ndarray
        Initial guess for the copula parameters [θ, δ].
    default_optim_method : str
        Default optimizer to use.
    """

    def __init__(self):
        super().__init__()
        self.type = "bb6"
        self.name = "BB6 Copula"
        self.bounds_param = [(1.0, None), (1.0, None)]  # θ ≥ 1, δ ≥ 1
        self.parameters = np.array([2.0, 2.0])          # [θ, δ]
        self.default_optim_method = "SLSQP"

    def _phi(self, t, θ, δ):
        """Archimedean generator φ(t)."""
        return (-np.log(1 - (1 - t)**θ))**(1.0/δ)

    def _phi_prime(self, t, θ, δ):
        """Derivative φ'(t) of the generator."""
        # g(t) = 1 - (1 - t)^θ
        g = 1 - (1 - t)**θ
        # g'(t) = θ (1 - t)^(θ - 1)
        gp = θ * (1 - t)**(θ - 1)
        # L(t) = -ln(g(t)), L'(t) = -g'(t)/g(t)
        L = -np.log(g)
        Lp = -gp / g
        # φ(t) = [L(t)]^(1/δ) => φ'(t) = (1/δ) L^(1/δ - 1) · L'
        return (1.0/δ) * L**(1.0/δ - 1) * Lp

    def get_cdf(self, u, v, param):
        """
        C(u,v) = 1 − [1 − w]^{1/θ},  where
          w = exp( − [ x^δ + y^δ ]^{1/δ} )
          x = −log(1 − (1−u)^θ),  y = −log(1 − (1−v)^θ)
        """
        θ, δ = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        ubar, vbar = 1 - u, 1 - v
        x = -np.log(1 - ubar**θ)
        y = -np.log(1 - vbar**θ)
        sm = x**δ + y**δ
        tem = sm**(1.0/δ)
        w = np.exp(-tem)

        return 1.0 - (1.0 - w)**(1.0/θ)

    def get_pdf(self, u, v, param):
        """
        PDF c(u,v) by differentiating C:
          prefac = (1−w)^{1/θ−2} · w · (tem/sm^2) · (x^δ/x) · (y^δ/y)
          bracket = ( (θ−w)·tem + θ(δ−1)(1−w) )
          jac    = (1−zu)(1−zv)/(zu·zv·ū·v̄)
        """
        θ, δ = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        ubar, vbar = 1 - u, 1 - v
        zu = 1 - ubar**θ
        zv = 1 - vbar**θ
        x, y = -np.log(zu), -np.log(zv)

        xd, yd = x**δ, y**δ
        sm = xd + yd
        tem = sm**(1.0/δ)
        w = np.exp(-tem)

        prefac = (1 - w)**(1.0/θ - 2) * w * (tem / (sm**2)) * (xd / x) * (yd / y)
        bracket = ( (θ - w) * tem + θ * (δ - 1) * (1 - w) )
        jac = (1 - zu) * (1 - zv) / (zu * zv * ubar * vbar)

        return prefac * bracket * jac

    def partial_derivative_C_wrt_u(self, u, v, param):
        """
        ∂C/∂u via chain‑rule for BB6:
          C = 1 - (1-w)^(1/θ),
          w = exp(- [ x^δ + y^δ ]^(1/δ) ),
          x = -ln(1-(1-u)^θ), y = -ln(1-(1-v)^θ).
        """
        θ, δ = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        # build w
        ubar, vbar = 1 - u, 1 - v
        x = -np.log(1 - ubar ** θ)
        y = -np.log(1 - vbar ** θ)
        xd = x ** δ
        yd = y ** δ
        sm = xd + yd
        tem = sm ** (1.0 / δ)
        w = np.exp(-tem)

        # **correct** dC/dw for C=1-(1-w)^(1/θ)
        dC_dw = (1.0 / θ) * (1 - w) ** (1.0 / θ - 1)
        dw_dtem = -w
        dC_dtem = dC_dw * dw_dtem

        # dtem/dsm and dsm/dx
        dtem_dsm = (1.0 / δ) * sm ** (1.0 / δ - 1)
        dsm_dx = δ * x ** (δ - 1)
        dC_dx = dC_dtem * dtem_dsm * dsm_dx

        # dx/du for x = -ln(1-ubar^θ)
        dx_du = -θ * ubar ** (θ - 1) / (1 - ubar ** θ)

        # ∂C/∂u
        return dC_dx * dx_du

    def partial_derivative_C_wrt_v(self, u, v, param):
        """
        ∂C/∂v via symmetry: swap u<->v.
        """
        return self.partial_derivative_C_wrt_u(v, u, param)

    def conditional_cdf_v_given_u(self, u, v, param):
        """
        P(V ≤ v | U = u) = ∂C/∂u ÷ ∂C/∂u at v=1.
        """
        num = self.partial_derivative_C_wrt_u(u, v, param)
        denom = self.partial_derivative_C_wrt_u(u, 1.0, param)
        return num / denom

    def conditional_cdf_u_given_v(self, u, v, param):
        """
        P(U ≤ u | V = v) = ∂C/∂v ÷ ∂C/∂v at u=1.
        """
        num = self.partial_derivative_C_wrt_v(u, v, param)
        denom = self.partial_derivative_C_wrt_v(1.0, v, param)
        return num / denom

    def sample(self, n, param):
        """
        Generate n samples via conditional inversion:
          u ~ Uniform(0,1)
          For each u, solve C_{2|1}(v|u) = U' by bisection.
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
                method="bisect", xtol=1e-6
            )
            v[i] = sol.root
        return np.column_stack((u, v))

    def kendall_tau(self, param):
        """
        Compute Kendall's tau via
          τ = 1 + 4 ∫₀¹ φ(t) / φ'(t) dt,
        using numerical integration.
        """
        θ, δ = param
        integral, _ = quad(
            lambda t: self._phi(t, θ, δ) / self._phi_prime(t, θ, δ),
            0.0, 1.0
        )
        return 1.0 + 4.0 * integral