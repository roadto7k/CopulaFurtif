import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar

from Service.Copulas.base import BaseCopula


class TawnT1Copula(BaseCopula):
    """
    Tawn Type‑1 (asymmetric logistic) extreme‑value copula.

    Pickands dependence function:
      A(t) = (1 − β)·t + [ (1 − t)^θ + (β·t)^θ ]^(1/θ)
    where θ ≥ 1, β ∈ [0,1]. :contentReference[oaicite:0]{index=0}

    The copula itself is
      C(u,v) = exp{ − (x + y)·A( x/(x+y) ) },
    with x = −log(u), y = −log(v). :contentReference[oaicite:1]{index=1}
    """

    def __init__(self):
        super().__init__()
        self.type = "tawn1"
        self.name = "Tawn Type‑1 Copula"
        # θ ≥ 1, β ∈ [0,1]
        self.bounds_param = [(1.0, None), (0.0, 1.0)]
        self.parameters = np.array([2.0, 0.5])  # [θ, β]
        self.default_optim_method = "SLSQP"

    def _A(self, t, param):
        """
        Pickands dependence function A(t).
        """
        θ, β = param
        h = (1 - t)**θ + (β * t)**θ
        return (1 - β) * t + h**(1.0/θ)

    def _A_prime(self, t, param):
        """
        First derivative A'(t).
        """
        θ, β = param
        # h(t) = (1 − t)^θ + (β·t)^θ
        h = (1 - t)**θ + (β * t)**θ
        # h'(t)
        hp = -θ * (1 - t)**(θ - 1) + θ * β * (β * t)**(θ - 1)
        return (1 - β) + (1.0/θ) * h**(1.0/θ - 1) * hp

    def _A_double(self, t, param):
        """
        Second derivative A''(t).
        """
        θ, β = param
        h = (1 - t)**θ + (β * t)**θ
        hp = -θ * (1 - t)**(θ - 1) + θ * β * (β * t)**(θ - 1)
        # h''(t)
        hpp = θ * (θ - 1) * ((1 - t)**(θ - 2) + β**θ * t**(θ - 2))
        term1 = (1.0/θ) * (1.0/θ - 1) * h**(1.0/θ - 2) * hp**2
        term2 = (1.0/θ) * h**(1.0/θ - 1) * hpp
        return term1 + term2

    def get_cdf(self, u, v, param):
        """
        CDF C(u,v) = exp( − (x + y)·A(x/(x+y)) )
        """
        θ, β = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        x = -np.log(u)
        y = -np.log(v)
        s = x + y
        t = x / s

        return np.exp(-s * self._A(t, param))

    def get_pdf(self, u, v, param):
        """
        Density c(u,v) = C(u,v) · [ℓ_x·ℓ_y − ℓ_{xy}] / (u·v),
        where ℓ(x,y) = (x+y)·A(x/(x+y)) is the stable tail function.
        """
        θ, β = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        x = -np.log(u)
        y = -np.log(v)
        s = x + y
        t = x / s
        A = self._A(t, param)
        Ap = self._A_prime(t, param)
        App = self._A_double(t, param)

        # partials of ℓ = s*A(t)
        # ℓ_x = A + (y/s)*A'
        Lx = A + (y / s) * Ap
        # ℓ_y = A - (x/s)*A'
        Ly = A - (x / s) * Ap
        # ℓ_{xy} = - (x·y / s^3) · A''
        Lxy = - (x * y / s**3) * App

        C = np.exp(-s * A)
        return C * (Lx * Ly - Lxy) / (u * v)

    def partial_derivative_C_wrt_u(self, u, v, param):
        """
        ∂C/∂u = C(u,v) · [ A(t)/u + (y/(u·s))·A'(t) ].
        """
        C = self.get_cdf(u, v, param)
        x = -np.log(u)
        y = -np.log(v)
        s = x + y
        t = x / s
        A = self._A(t, param)
        Ap = self._A_prime(t, param)
        return C * (A / u + (y / (u * s)) * Ap)

    def partial_derivative_C_wrt_v(self, u, v, param):
        """
        ∂C/∂v = C(u,v) · [ A(t)/v - (x/(v·s))·A'(t) ].
        """
        C = self.get_cdf(u, v, param)
        x = -np.log(u)
        y = -np.log(v)
        s = x + y
        t = x / s
        A = self._A(t, param)
        Ap = self._A_prime(t, param)
        return C * (A / v - (x / (v * s)) * Ap)

    def conditional_cdf_v_given_u(self, u, v, param):
        """
        P(V ≤ v | U = u) = ∂C/∂u(u, v)
        (since U ~ Uniform(0,1)).
        """
        return self.partial_derivative_C_wrt_u(u, v, param)

    def conditional_cdf_u_given_v(self, u, v, param):
        """
        P(U ≤ u | V = v) = ∂C/∂v(u, v).
        """
        return self.partial_derivative_C_wrt_v(u, v, param)

    def sample(self, n, param):
        """
        Generate n samples via conditional inversion:
          1. Draw u ~ Uniform(0,1)
          2. Draw p ~ Uniform(0,1)
          3. Solve ∂C/∂u(u, v) = p for v by bisection.
        """
        θ, β = param
        eps = 1e-6
        u = np.random.rand(n)
        v = np.empty(n)
        for i in range(n):
            p = np.random.rand()
            sol = root_scalar(
                lambda vv: self.conditional_cdf_v_given_u(u[i], vv, param) - p,
                bracket=[eps, 1 - eps],
                method="bisect", xtol=1e-6
            )
            v[i] = sol.root
        return np.column_stack((u, v))

    def kendall_tau(self, param):
        """
        Kendall's tau for an EV copula:
          τ = 1 − 4 ∫₀¹ A(t) dt.
        Computed numerically.
        """
        integral, _ = quad(lambda t: self._A(t, param), 0.0, 1.0)
        return 1.0 - 4.0 * integral

    def LTDC(self, param):
        """
        Lower‑tail dependence coefficient λ_L = 0 for Tawn Type‑1.
        """
        return 0.0

    def UTDC(self, param):
        """
        Upper‑tail dependence coefficient λ_U = 2 − 2^(1/θ)
        (same as the Gumbel/Logistic EV copula).
        """
        θ, _ = param
        return 2.0 - 2.0**(1.0/θ)