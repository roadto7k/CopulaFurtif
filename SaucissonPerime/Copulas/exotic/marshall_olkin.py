import numpy as np
from scipy.stats import expon

from SaucissonPerime.Copulas.base import BaseCopula


class MarshallOlkinCopula(BaseCopula):
    """
    Marshall–Olkin shock‐model copula.

    C(u,v) = min( u^(1−α)·v,  u·v^(1−β) ),
    α, β ∈ (0,1).

    Attributes
    ----------
    family : str
        Identifier for the copula family. Here, "marshall-olkin".
    name : str
        Human-readable name for output/logging.
    bounds_param : list of tuple
        Bounds for the copula parameters α, β ∈ (0,1).
    parameters : np.ndarray
        Initial guess for the copula parameters [α, β].
    default_optim_method : str
        Default optimizer to use.
    """

    def __init__(self):
        super().__init__()
        self.type = "marshall-olkin"
        self.name = "Marshall–Olkin Copula"
        self.bounds_param = [(0.0, 1.0), (0.0, 1.0)]
        self.parameters = np.array([0.5, 0.5])  # [α, β]
        self.default_optim_method = "SLSQP"

    def get_cdf(self, u, v, param):
        """
        Copula C(u,v) = min{ u^(1−α) v,  u v^(1−β) }.
        α, β in (0,1). :contentReference[oaicite:0]{index=0}
        """
        α, β = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        # region where u^(1−α) v ≤ u v^(1−β)  ⇔  v ≤ u^(α/β)
        cond = v <= u**(α/β)
        c1 = u**(1 - α) * v
        c2 = u * v**(1 - β)
        return np.where(cond, c1, c2)

    def get_pdf(self, u, v, param):
        """
        Density wrt Lebesgue (ignoring singular part):
          c(u,v) = (1−α) u^(−α)    if v < u^(α/β),
                 = (1−β) v^(−β)    if v > u^(α/β).
        """
        α, β = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        cond = v < u**(α/β)
        d1 = (1 - α) * u**(-α)
        d2 = (1 - β) * v**(-β)
        return np.where(cond, d1, d2)

    def kendall_tau(self, param):
        """
        Kendall's τ = α·β / (α + β − α·β). :contentReference[oaicite:1]{index=1}
        """
        α, β = param
        return (α * β) / (α + β - α * β)

    def partial_derivative_C_wrt_u(self, u, v, param):
        """
        ∂C/∂u:
          = (1−α) u^(−α) v    if v < u^(α/β),
          = v^(1−β)           otherwise.
        """
        α, β = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        cond = v < u**(α/β)
        p1 = (1 - α) * u**(-α) * v
        p2 = v**(1 - β)
        return np.where(cond, p1, p2)

    def partial_derivative_C_wrt_v(self, u, v, param):
        """
        ∂C/∂v:
          = u^(1−α)           if v < u^(α/β),
          = (1−β) u v^(−β)    otherwise.
        """
        α, β = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        cond = v < u**(α/β)
        q1 = u**(1 - α)
        q2 = (1 - β) * u * v**(-β)
        return np.where(cond, q1, q2)

    def conditional_cdf_v_given_u(self, u, v, param):
        """
        P(V ≤ v | U = u) = ∂C/∂u(u,v).
        """
        return self.partial_derivative_C_wrt_u(u, v, param)

    def conditional_cdf_u_given_v(self, u, v, param):
        """
        P(U ≤ u | V = v) = ∂C/∂v(u,v).
        """
        return self.partial_derivative_C_wrt_v(u, v, param)

    def sample(self, n, param):
        """
        Simulate via the MO shock model:
          E0 ~ Exp(1), E1 ~ Exp((1−α)/α), E2 ~ Exp((1−β)/β)
          X = min(E0, E1),  Y = min(E0, E2)
          U = 1 − exp(−X/α),  V = 1 − exp(−Y/β)
        """
        α, β = param
        # rates for E1, E2 to match marginal uniforms
        λ1 = (1 - α) / α
        λ2 = (1 - β) / β
        E0 = expon.rvs(scale=1.0, size=n)
        E1 = expon.rvs(scale=1/λ1, size=n)
        E2 = expon.rvs(scale=1/λ2, size=n)
        X = np.minimum(E0, E1)
        Y = np.minimum(E0, E2)
        U = 1 - np.exp(-X / α)
        V = 1 - np.exp(-Y / β)
        return np.column_stack((U, V))

    def LTDC(self, param):
        """
        Lower‐tail dependence λ_L = 0.
        """
        return 0.0

    def UTDC(self, param):
        """
        Upper‐tail dependence λ_U = min(α, β). :contentReference[oaicite:2]{index=2}
        """
        α, β = param
        return min(α, β)

    def IAD(self, data):
        """
        Integrated Anderson–Darling not implemented for Marshall–Olkin.
        """
        print(f"[INFO] IAD not implemented for {self.name}.")
        return np.nan

    def AD(self, data):
        """
        Anderson–Darling not implemented for Marshall–Olkin.
        """
        print(f"[INFO] AD not implemented for {self.name}.")
        return np.nan
