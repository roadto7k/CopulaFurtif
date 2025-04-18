import numpy as np
from scipy.stats import norm
from scipy.optimize import root_scalar
from scipy.integrate import quad

from copulas.domain.copulas.base import BaseCopula


class HuslerReissCopula(BaseCopula):
    """
    Hüsler–Reiss (extreme‑value) Copula class

    Attributes
    ----------
    family : str
        Identifier for the copula family. Here, "husler-reiss".
    name : str
        Human-readable name for output/logging.
    bounds_param : list of tuple
        Bounds for the copula parameter Θ ≥ 0.
    parameters : np.ndarray
        Initial guess for the copula parameter [Θ].
    default_optim_method : str
        Default optimizer to use.
    """

    def __init__(self):
        super().__init__()
        self.type = "husler-reiss"
        self.name = "Hüsler–Reiss Copula"
        self.bounds_param = [(0.0, None)]      # Θ ≥ 0
        self.parameters = np.array([1.0])      # initial Θ
        self.default_optim_method = "SLSQP"

    def get_cdf(self, u, v, param):
        """
        C(u,v) = exp[ -x·Φ(X) - y·Φ(Y) ],  where
          x = -log(u),  y = -log(v),
          X = 1/Θ + (Θ/2)·log(x/y),
          Y = 1/Θ + (Θ/2)·log(y/x).

        Returns
        -------
        float or np.ndarray
            Copula CDF at (u, v). :contentReference[oaicite:0]{index=0}
        """
        Θ = param[0]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        x = -np.log(u)
        y = -np.log(v)
        X = 1.0/Θ + (Θ/2.0) * np.log(x / y)
        Y = 1.0/Θ + (Θ/2.0) * np.log(y / x)

        return np.exp(-x * norm.cdf(X) - y * norm.cdf(Y))

    def get_pdf(self, u, v, param):
        """
        PDF c(u,v) computed analytically:
          ℓ_x = Φ(X) + (Θ/2)·φ(X) - (Θ·y/(2x))·φ(Y)
          ℓ_y = Φ(Y) + (Θ/2)·φ(Y) - (Θ·x/(2y))·φ(X)
          ℓ_xy = (Θ/(4y))(Θ·X - 2)φ(X) - (Θ/(2x))(1 - Θ·Y/2)φ(Y)
          c = C(u,v)·[ℓ_x·ℓ_y - ℓ_xy] / (u·v)
        """
        Θ = param[0]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        x = -np.log(u)
        y = -np.log(v)
        X = 1.0/Θ + (Θ/2.0) * np.log(x / y)
        Y = 1.0/Θ + (Θ/2.0) * np.log(y / x)

        C = np.exp(-x * norm.cdf(X) - y * norm.cdf(Y))
        φX = norm.pdf(X)
        φY = norm.pdf(Y)
        ΦX = norm.cdf(X)
        ΦY = norm.cdf(Y)

        ℓ_x = ΦX + (Θ/2.0)*φX - (Θ*y/(2.0*x))*φY
        ℓ_y = ΦY + (Θ/2.0)*φY - (Θ*x/(2.0*y))*φX
        ℓ_xy = (Θ/(4.0*y))*(Θ*X - 2.0)*φX - (Θ/(2.0*x))*(1.0 - Θ*Y/2.0)*φY

        return C * (ℓ_x * ℓ_y - ℓ_xy) / (u * v)

    def partial_derivative_C_wrt_u(self, u, v, param):
        """
        ∂C/∂u = C(u,v)·ℓ_x(u,v) / u
        """
        C = self.get_cdf(u, v, param)
        Θ = param[0]
        x = -np.log(np.clip(u, 1e-12, 1-1e-12))
        y = -np.log(np.clip(v, 1e-12, 1-1e-12))
        X = 1.0/Θ + (Θ/2.0) * np.log(x / y)
        Y = 1.0/Θ + (Θ/2.0) * np.log(y / x)
        φX = norm.pdf(X)
        φY = norm.pdf(Y)
        ΦX = norm.cdf(X)
        # ℓ_x as in get_pdf:
        ℓ_x = (
            ΦX
            + (Θ/2.0) * φX
            - (Θ * y / (2.0 * x)) * φY
        )
        return C * ℓ_x / u

    def partial_derivative_C_wrt_v(self, u, v, param):
        """
        ∂C/∂v = C(u,v)·ℓ_y(u,v) / v
        """
        C = self.get_cdf(u, v, param)
        Θ = param[0]
        x = -np.log(np.clip(u, 1e-12, 1-1e-12))
        y = -np.log(np.clip(v, 1e-12, 1-1e-12))
        X = 1.0/Θ + (Θ/2.0) * np.log(x / y)
        Y = 1.0/Θ + (Θ/2.0) * np.log(y / x)
        φX = norm.pdf(X)
        φY = norm.pdf(Y)
        ΦY = norm.cdf(Y)
        # ℓ_y as in get_pdf:
        ℓ_y = (
            ΦY
            + (Θ/2.0) * φY
            - (Θ * x / (2.0 * y)) * φX
        )
        return C * ℓ_y / v

    def conditional_cdf_v_given_u(self, u, v, param):
        """
        P(V ≤ v | U = u) = ∂C/∂u(u,v) (since C(u,1)=u, ∂C/∂u|_{v=1}=1).
        """
        return self.partial_derivative_C_wrt_u(u, v, param)

    def conditional_cdf_u_given_v(self, u, v, param):
        """
        P(U ≤ u | V = v) = ∂C/∂v(u,v).
        """
        return self.partial_derivative_C_wrt_v(u, v, param)

    def sample(self, n, param):
        """
        Generate n samples via conditional inversion:
          1. Draw U ~ Uniform(0,1)
          2. For each U=u, draw V by solving C_{2|1}(v|u) = p with p~Uniform(0,1)
        """
        Θ = param[0]
        eps = 1e-6
        u = np.random.rand(n)
        v = np.empty(n)
        for i in range(n):
            p = np.random.rand()
            sol = root_scalar(
                lambda vv: self.conditional_cdf_v_given_u(u[i], vv, param) - p,
                bracket=[eps, 1-eps],
                method='bisect',
                xtol=1e-6
            )
            v[i] = sol.root
        return np.column_stack((u, v))

    def kendall_tau(self, param):
        """
        τ = 1 − 4 ∫₀¹ A(t) dt,
        where A(t) = t·Φ(X_t) + (1−t)·Φ(Y_t),
        X_t = 1/Θ + (Θ/2)·log[t/(1−t)],
        Y_t = 1/Θ + (Θ/2)·log[(1−t)/t].
        Computed by numerical integration. :contentReference[oaicite:1]{index=1}
        """
        Θ = param[0]

        def A(t):
            eps = 1e-12
            t = np.clip(t, eps, 1-eps)
            Xt = 1.0/Θ + (Θ/2.0) * np.log(t / (1-t))
            Yt = 1.0/Θ + (Θ/2.0) * np.log((1-t) / t)
            return t * norm.cdf(Xt) + (1-t) * norm.cdf(Yt)

        integral, _ = quad(A, 0.0, 1.0)
        return 1.0 - 4.0 * integral

    def LTDC(self, param):
        """
        Lower‑tail dependence λ_L = 0 (extreme-value copula).
        """
        return 0.0

    def UTDC(self, param):
        """
        Upper‑tail dependence λ_U = 2 − 2·Φ(a/2),
        where a = 1/Θ. :contentReference[oaicite:2]{index=2}
        """
        Θ = param[0]
        a = 1.0 / Θ
        return 2.0 - 2.0 * norm.cdf(a / 2.0)

    def IAD(self, data):
        """
        Integrated Anderson–Darling not implemented for Hüsler–Reiss.
        """
        print(f"[INFO] IAD not implemented for {self.name}.")
        return np.nan

    def AD(self, data):
        """
        Anderson–Darling not implemented for Hüsler–Reiss.
        """
        print(f"[INFO] AD not implemented for {self.name}.")
        return np.nan
