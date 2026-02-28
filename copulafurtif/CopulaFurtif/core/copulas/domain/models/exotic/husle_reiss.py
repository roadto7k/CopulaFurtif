import numpy as np
from numpy.random import default_rng
from scipy.stats import norm
from scipy.integrate import quad

from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel, CopulaParameters
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


class HuslerReissCopula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """
    Bivariate Hüsler–Reiss extreme-value copula.

    Parametrization (book):
      δ >= 0 (we use δ > 0 with exclusive bounds; # δ -> 0+ gives independence, δ -> +∞ gives comonotone dependence).
    """

    def __init__(self):
        super().__init__()
        self.name = "Hüsler–Reiss Copula"
        self.type = "husler-reiss"
        self.default_optim_method = "SLSQP"

        # Exclusive bounds in your framework: (lo < δ < hi).
        # Put a finite hi for numerical stability.
        self.init_parameters(CopulaParameters(
            np.array([1.0], dtype=float),
            [(0.0, 50.0)],
            ["delta"],
        ))

    # -------------------------
    # Helpers
    # -------------------------
    @staticmethod
    def _clip01(x, eps=1e-12):
        return np.clip(x, eps, 1.0 - eps)

    @staticmethod
    def _ensure_pairwise(u, v):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        if u.ndim == 0 and v.ndim == 0:
            return u, v, True
        if u.shape != v.shape:
            raise ValueError("Vectorized evaluation is pairwise: u and v must have the same shape.")
        return u, v, False

    def _xyXY(self, u, v, delta):
        u = self._clip01(u)
        v = self._clip01(v)
        x = -np.log(u)
        y = -np.log(v)

        # log(x/y) robust
        log_xy = np.log(x) - np.log(y)
        inv = 1.0 / delta

        X = inv + 0.5 * delta * log_xy
        Y = inv - 0.5 * delta * log_xy
        return x, y, X, Y

    # -------------------------
    # Core API
    # -------------------------
    def get_cdf(self, u, v, param=None):
        if param is None:
            param = self.get_parameters()
        delta = float(param[0])

        eps = 1e-12
        u = np.clip(np.asarray(u, float), eps, 1.0 - eps)
        v = np.clip(np.asarray(v, float), eps, 1.0 - eps)

        u, v, is_scalar = self._ensure_pairwise(u, v)
        x, y, X, Y = self._xyXY(u, v, delta)

        # C(u,v)=exp{-x Φ(X) - y Φ(Y)}
        logC = -(x * norm.cdf(X) + y * norm.cdf(Y))
        C = np.exp(logC)

        return float(C) if is_scalar else C

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """
        Book (4.21): C_{2|1}(v|u)=∂C/∂u = C(u,v) * u^{-1} * Φ(X),
        with X = δ^{-1} + (δ/2) log(x/y), x=-log u, y=-log v.
        """
        if param is None:
            param = self.get_parameters()
        delta = float(param[0])

        eps = 1e-12
        u = np.clip(np.asarray(u, float), eps, 1.0 - eps)
        v = np.clip(np.asarray(v, float), eps, 1.0 - eps)

        u, v, is_scalar = self._ensure_pairwise(u, v)
        u = self._clip01(u)
        v = self._clip01(v)

        x, y, X, Y = self._xyXY(u, v, delta)
        logC = -(x * norm.cdf(X) + y * norm.cdf(Y))
        C = np.exp(logC)

        h = C * norm.cdf(X) / u
        h = np.clip(h, 0.0, 1.0)

        return float(h) if is_scalar else h

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """
        Symmetric: ∂C/∂v = C(u,v) * v^{-1} * Φ(Y)
        """
        if param is None:
            param = self.get_parameters()
        delta = float(param[0])

        eps = 1e-12
        u = np.clip(np.asarray(u, float), eps, 1.0 - eps)
        v = np.clip(np.asarray(v, float), eps, 1.0 - eps)

        u, v, is_scalar = self._ensure_pairwise(u, v)
        u = self._clip01(u)
        v = self._clip01(v)

        x, y, X, Y = self._xyXY(u, v, delta)
        logC = -(x * norm.cdf(X) + y * norm.cdf(Y))
        C = np.exp(logC)

        h = C * norm.cdf(Y) / v
        h = np.clip(h, 0.0, 1.0)

        return float(h) if is_scalar else h

    def get_pdf(self, u, v, param=None):
        """
        Book (Hüsler–Reiss, Section 4.10.1, density after (4.21)):

          c(u,v;δ) = C(u,v;δ)/(u v) * [ Φ(Y)Φ(X) + (δ/2) * y^{-1} * φ(X) ],

        where x = -log u, y = -log v,
              X = δ^{-1} + (δ/2) log(x/y),
              Y = δ^{-1} + (δ/2) log(y/x).
        """
        if param is None:
            param = self.get_parameters()
        delta = float(param[0])

        u, v, is_scalar = self._ensure_pairwise(u, v)
        u = self._clip01(u)
        v = self._clip01(v)

        x, y, X, Y = self._xyXY(u, v, delta)

        # C(u,v)=exp{-x Φ(X) - y Φ(Y)}
        logC = -(x * norm.cdf(X) + y * norm.cdf(Y))
        C = np.exp(logC)

        PhiX = norm.cdf(X)
        PhiY = norm.cdf(Y)
        phiX = norm.pdf(X)

        y_safe = np.maximum(y, 1e-15)

        pdf = (C / (u * v)) * (PhiX * PhiY + 0.5 * delta * (phiX / y_safe))
        pdf = np.maximum(pdf, 0.0)

        return float(pdf) if is_scalar else pdf

    # -------------------------
    # Dependence summaries
    # -------------------------
    def UTDC(self, param=None) -> float:
        """
        Book: λ_U = 2[1 - Φ(δ^{-1})]
        δ → 0+ : λU → 0 (weak upper-tail dependence)
        δ → ∞ : λU → 1 (strong upper-tail dependence, comonotone limit)
        """
        if param is None:
            param = self.get_parameters()
        delta = float(param[0])
        return float(2.0 * (1.0 - norm.cdf(1.0 / delta)))

    def LTDC(self, param=None) -> float:
        return 0.0

    def blomqvist_beta(self, param=None) -> float:
        """
        β = 4 C(1/2,1/2) - 1.
        For HR: C(1/2,1/2) = 2^{-2 Φ(δ^{-1})} => β = 2^{2-2Φ(δ^{-1})}-1.
        """
        if param is None:
            param = self.get_parameters()
        delta = float(param[0])
        return float(2.0 ** (2.0 - 2.0 * norm.cdf(1.0 / delta)) - 1.0)

    def kendall_tau(self, param=None) -> float:
        """
        Kendall τ via EV formula (8.11) using Pickands B(w) and B'(w).

        Book (4.22):
          B(w;δ) = w Φ(δ^{-1} + (δ/2) log(w/(1-w))) + (1-w) Φ(δ^{-1} + (δ/2) log((1-w)/w))
          B'(w;δ)= Φ(δ^{-1} + (δ/2) log(w/(1-w))) - Φ(δ^{-1} + (δ/2) log((1-w)/w))
        """
        if param is None:
            param = self.get_parameters()
        delta = float(param[0])

        eps = 1e-10

        def B_and_Bp(w):
            w = np.clip(w, eps, 1.0 - eps)
            logit = np.log(w) - np.log(1.0 - w)
            a = (1.0 / delta) + 0.5 * delta * logit
            b = (1.0 / delta) - 0.5 * delta * logit
            Fa = norm.cdf(a)
            Fb = norm.cdf(b)
            B = w * Fa + (1.0 - w) * Fb
            Bp = Fa - Fb
            return B, Bp

        def integrand(w):
            B, Bp = B_and_Bp(w)
            num = (2.0 * w - 1.0) * Bp * B + w * (1.0 - w) * (Bp ** 2)
            den = B ** 2
            return num / den

        val, _ = quad(integrand, eps, 1.0 - eps, limit=300)
        return float(val)

    # -------------------------
    # init_from_data (β first)
    # -------------------------
    def init_from_data(self, u, v):
        """
        Prefer Blomqvist beta inversion (closed form), then clamp to bounds.

        Empirical:
          beta_hat = 4 * mean(1{u<=1/2, v<=1/2}) - 1

        Inversion from book:
          β = 2^{2-2Φ(1/δ)} - 1
          => Φ(1/δ) = 1 - (1/2) log_2(1+β)
          => 1/δ = Φ^{-1}(1 - log(1+β)/(2 log 2))
          => δ = 1 / Φ^{-1}(...)
        """
        u = np.asarray(u, float).ravel()
        v = np.asarray(v, float).ravel()
        mask = np.isfinite(u) & np.isfinite(v)
        u = u[mask]
        v = v[mask]
        if u.size < 30:
            return self.get_parameters()

        u = self._clip01(u)
        v = self._clip01(v)

        c_hat = float(np.mean((u <= 0.5) & (v <= 0.5)))
        beta_emp = 4.0 * c_hat - 1.0

        # HR is positive-dependence only => beta in [0,1). Clip for stability.
        beta_emp = float(np.clip(beta_emp, 0.0, 0.999999))

        # invert beta -> delta
        p = 1.0 - (np.log(1.0 + beta_emp) / (2.0 * np.log(2.0)))
        p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
        z = float(norm.ppf(p))

        low, high = self.get_bounds()[0]
        eps_th = 1e-6

        if z <= 0:
            delta0 = high - eps_th
        else:
            delta0 = 1.0 / z
            delta0 = float(np.clip(delta0, low + eps_th, high - eps_th))

        return np.array([delta0], dtype=float)

    # -------------------------
    # Sampling (vectorized bisection)
    # -------------------------
    def sample(self, n: int, param=None, rng=None, eps: float = 1e-12) -> np.ndarray:
        if param is None:
            param = self.get_parameters()
        delta = float(param[0])

        if rng is None:
            rng = default_rng()

        u = rng.random(int(n))
        p = rng.random(int(n))

        lo = np.full(int(n), eps)
        hi = np.full(int(n), 1.0 - eps)

        # monotone in v -> bisection
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            F = self.partial_derivative_C_wrt_u(u, mid, param=[delta])  # C_{2|1}(mid|u)
            mask = F < p
            lo[mask] = mid[mask]
            hi[~mask] = mid[~mask]

        v = 0.5 * (lo + hi)
        uv = np.column_stack([np.clip(u, eps, 1.0 - eps), np.clip(v, eps, 1.0 - eps)])
        return uv