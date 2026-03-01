import numpy as np
from scipy.special import beta
from scipy.stats import kendalltau

from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel, CopulaParameters
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence
from numpy.random import default_rng


class BB1Copula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """
    BB1 Copula (Two-parameter Archimedean copula).

    The BB1 copula extends Clayton and Gumbel copulas with two parameters: 
    one for dependence strength (theta > 0) and one for tail dependence (delta >= 1).

    Attributes:
        name (str): Human-readable name of the copula.
        type (str): Identifier for the copula family.
        bounds_param (list of tuple): Bounds for parameters [theta, delta].
        parameters (np.ndarray): Current copula parameters.
        default_optim_method (str): Default optimization method.
    """

    def __init__(self):
        """Initialize BB1 copula with default parameters."""
        super().__init__()
        self.name = "BB1 Copula"
        self.type = "bb1"
        self.default_optim_method = "Powell"
        self.init_parameters(CopulaParameters(np.array([0.5, 1.5]),[(0, np.inf), (1.0, np.inf)], ["theta", "delta"] ))

    def get_cdf(self, u, v, param=None):
        """
        Compute the BB1 copula CDF C(u, v).

        Args:
            u (float or np.ndarray): First input in (0,1).
            v (float or np.ndarray): Second input in (0,1).
            param (np.ndarray, optional): Copula parameters [theta, delta].

        Returns:
            float or np.ndarray: Copula CDF values.
        """
        if param is None:
            param = self.get_parameters()
        theta, delta = map(float, param)

        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)

        # --- exact boundaries (Fréchet)
        # (works for scalar & arrays)
        out = np.empty(np.broadcast(u, v).shape, dtype=float)
        uu, vv = np.broadcast_arrays(u, v)

        out[(uu <= 0.0) | (vv <= 0.0)] = 0.0
        out[uu >= 1.0] = np.clip(vv[uu >= 1.0], 0.0, 1.0)
        out[vv >= 1.0] = np.clip(uu[vv >= 1.0], 0.0, 1.0)

        mask = (uu > 0.0) & (uu < 1.0) & (vv > 0.0) & (vv < 1.0)
        if np.any(mask):
            eps = 1e-12
            um = np.clip(uu[mask], eps, 1.0 - eps)
            vm = np.clip(vv[mask], eps, 1.0 - eps)

            term1 = (um ** (-theta) - 1.0) ** delta
            term2 = (vm ** (-theta) - 1.0) ** delta
            inner = term1 + term2
            out[mask] = (1.0 + inner ** (1.0 / delta)) ** (-1.0 / theta)

        # return scalar if scalar input
        return float(out) if out.shape == () else out

    def get_pdf(self, u, v, param=None):
        """
        Compute the BB1 copula PDF c(u, v).

        Args:
            u (float or np.ndarray): First input in (0,1).
            v (float or np.ndarray): Second input in (0,1).
            param (np.ndarray, optional): Copula parameters [theta, delta].

        Returns:
            float or np.ndarray: Copula PDF values.
        """
        if param is None:
            param = self.get_parameters()
        theta, delta = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        x = (u ** (-theta) - 1) ** delta
        y = (v ** (-theta) - 1) ** delta
        S = x + y
        term1 = (1 + S ** (1.0 / delta)) ** (-1.0 / theta - 2)
        term2 = S ** (1.0 / delta - 2)
        term3 = theta * (delta - 1) + (theta * delta + 1) * S ** (1.0 / delta)
        term4 = (x * y) ** (1 - 1.0 / delta) * (u * v) ** (-theta - 1)
        return term1 * term2 * term3 * term4

    def kendall_tau(self, param=None):
        """
        Closed-form Kendall's tau for the BB1 copula:

        τ(θ, δ) = 1 − 2 / [ δ (θ + 2) ]      (Joe 1997, Eq. 4.32)
        """
        if param is None:
            param = self.get_parameters()
        theta, delta = map(float, param)
        return 1.0 - 2.0 / (delta * (theta + 2.0))

    def _invert_conditional_v(self, u, target_cdf, theta, delta,
                              eps=1e-12, max_iter=40):
        """
        Solve for v in (0,1) such that

            F_{V|U}(v | u) = ∂C(u,v)/∂u = target_cdf

        by vectorized bisection.
        """
        lo = np.full_like(target_cdf, eps)
        hi = np.full_like(target_cdf, 1.0 - eps)

        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            cdf_mid = self.partial_derivative_C_wrt_u(u, mid, [theta, delta])
            # when CDF(mid) > target, decrease upper bound
            hi = np.where(cdf_mid > target_cdf, mid, hi)
            # when CDF(mid) <= target, increase lower bound
            lo = np.where(cdf_mid <= target_cdf, mid, lo)

        return 0.5 * (lo + hi)

    def sample(self,
               n: int,
               param=None,
               rng=None,
               eps: float = 1e-12,
               max_iter: int = 40) -> np.ndarray:
        """
        Generate n i.i.d. pairs (U, V) from the BB1 copula by conditional inversion.

        Parameters
        ----------
        n        : int
            Number of samples.
        param    : sequence-like, optional
            Copula parameters [theta, delta].  If None, uses current parameters.
        rng      : numpy.random.Generator, optional
            Random number generator for reproducibility.
        eps      : float
            Small epsilon to keep values in (0,1).
        max_iter : int
            Maximum number of bisection iterations.

        Returns
        -------
        ndarray of shape (n, 2)
            Columns are U and V.
        """
        if rng is None:
            rng = default_rng()

        if param is None:
            theta, delta = map(float, self.get_parameters())
        else:
            theta, delta = map(float, param)

        # 1) draw U uniform and target probabilities P
        u = rng.random(n)
        p = rng.random(n)

        # 2) invert conditional CDF ∂C/∂u to obtain V
        v = self._invert_conditional_v(u, p, theta, delta,
                                       eps=eps, max_iter=max_iter)

        # 3) final clipping and pack
        np.clip(u, eps, 1.0 - eps, out=u)
        np.clip(v, eps, 1.0 - eps, out=v)
        return np.column_stack((u, v))

    def LTDC(self, param=None):
        """
        Compute lower tail dependence coefficient.

        Args:
            param (np.ndarray, optional): Copula parameters [theta, delta].

        Returns:
            float: Lower tail dependence.
        """
        if param is None:
            param = self.get_parameters()
        theta, delta = param
        return 2.0 ** (-1.0 / (delta * theta))

    def UTDC(self, param=None):
        """
        Compute upper tail dependence coefficient.

        Args:
            param (np.ndarray, optional): Copula parameters [theta, delta].

        Returns:
            float: Upper tail dependence.
        """
        if param is None:
            param = self.get_parameters()
        delta = param[1]
        return 2.0 - 2.0 ** (1.0 / delta)

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """
        Compute partial derivative ∂C/∂u.

        Args:
            u (float or np.ndarray): U values.
            v (float or np.ndarray): V values.
            param (np.ndarray, optional): Copula parameters [theta, delta].

        Returns:
            float or np.ndarray: Partial derivative values.
        """
        if param is None:
            param = self.get_parameters()
        theta, delta = map(float, param)

        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        uu, vv = np.broadcast_arrays(u, v)

        out = np.empty(uu.shape, dtype=float)

        # boundaries in v: h(u,0)=0 ; h(u,1)=1 (for u in (0,1))
        out[vv <= 0.0] = 0.0
        out[vv >= 1.0] = 1.0

        # if u is at boundary, conditional CDF is degenerate but we keep it safe:
        out[uu <= 0.0] = 0.0
        out[uu >= 1.0] = 1.0  # convention, not used by your unit_interval anyway

        mask = (uu > 0.0) & (uu < 1.0) & (vv > 0.0) & (vv < 1.0)
        if np.any(mask):
            eps = 1e-12
            um = np.clip(uu[mask], eps, 1.0 - eps)
            vm = np.clip(vv[mask], eps, 1.0 - eps)

            # reuse stable powers
            u_pow = np.exp(np.minimum(-theta * np.log(um), 700.0))  # u^{-theta}
            v_pow = np.exp(np.minimum(-theta * np.log(vm), 700.0))  # v^{-theta}

            Tu = u_pow - 1.0
            Tv = v_pow - 1.0
            T = Tu ** delta + Tv ** delta

            factor = (1.0 + T ** (1.0 / delta)) ** (-1.0 / theta - 1.0)

            h = factor * T ** (1.0 / delta - 1.0) * (Tu ** (delta - 1.0)) * (u_pow / um)

            out[mask] = np.clip(h, 0.0, 1.0)

        return float(out) if out.shape == () else out

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """
        Compute partial derivative ∂C/∂v.

        Args:
            u (float or np.ndarray): U values.
            v (float or np.ndarray): V values.
            param (np.ndarray, optional): Copula parameters [theta, delta].

        Returns:
            float or np.ndarray: Partial derivative values.
        """
        return self.partial_derivative_C_wrt_u(v, u, param)

    def IAD(self, data):
        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan

    def blomqvist_beta(self, param=None) -> float:
        """
        Blomqvist's beta for BB1 (Joe & Hu, 1996; Eq. 4.56):

            beta = 4*beta_star - 1
            beta_star = (1 + 2^(1/delta) * (2^theta - 1))^(-1/theta)
        """
        if param is None:
            param = self.get_parameters()
        theta, delta = map(float, param)

        log2 = np.log(2.0)

        # a = 2^(1/delta)
        a = np.exp(log2 / delta)

        # b = 2^theta - 1 (stable)
        b = np.expm1(theta * log2)

        # beta_star = exp( -(1/theta) * log(1 + a*b) )
        beta_star = np.exp(-(1.0 / theta) * np.log1p(a * b))

        return float(4.0 * beta_star - 1.0)

    def init_from_data(self, u, v):
        """
        Initialize (theta, delta) from pseudo-observations using:
          - empirical Kendall tau (stable),
          - empirical upper tail dependence (for delta),
          - then theta from tau (closed-form), fallback from lower tail dep.

        This is a *starting guess* for optimization, not a perfect estimator.
        """

        u = np.asarray(u, dtype=float).ravel()
        v = np.asarray(v, dtype=float).ravel()

        mask = np.isfinite(u) & np.isfinite(v)
        u = u[mask]
        v = v[mask]
        if u.size < 50:
            return self.get_parameters()

        eps_uv = 1e-12
        u = np.clip(u, eps_uv, 1.0 - eps_uv)
        v = np.clip(v, eps_uv, 1.0 - eps_uv)

        # --- 1) empirical Kendall tau
        tau_emp = float(kendalltau(u, v).correlation)
        if not np.isfinite(tau_emp):
            tau_emp = 0.0

        # BB1 (theta>0, delta>=1) is positive dependence => tau should be >= 0
        tau_emp = max(0.0, min(tau_emp, 0.999999))

        # --- 2) empirical upper tail dependence (symmetric estimate)
        qU = 0.95
        Vu = (u > qU)
        Vv = (v > qU)
        bothU = Vu & Vv
        den_u = max(1, int(Vu.sum()))
        den_v = max(1, int(Vv.sum()))
        lam_u = bothU.sum() / den_v  # P(U>q | V>q)
        lam_v = bothU.sum() / den_u  # P(V>q | U>q)
        lamU_emp = float(0.5 * (lam_u + lam_v))
        lamU_emp = float(np.clip(lamU_emp, 1e-6, 0.999999))

        # invert lambda_U = 2 - 2^(1/delta)  ->  delta = log(2)/log(2 - lambda_U)
        log2 = np.log(2.0)
        denom = np.log(2.0 - lamU_emp)
        # denom in (log1, log2) = (0, log2)
        delta0 = float(log2 / denom)
        delta0 = max(delta0, 1.0 + 1e-6)

        # --- 3) theta from tau: tau = 1 - 2/(delta*(theta+2))
        if tau_emp < 1.0 - 1e-8:
            theta0 = 2.0 / (delta0 * (1.0 - tau_emp)) - 2.0
        else:
            theta0 = np.nan

        # fallback via lower tail dependence if theta0 invalid
        if (not np.isfinite(theta0)) or (theta0 <= 1e-8):
            qL = 0.05
            Lu = (u < qL)
            Lv = (v < qL)
            bothL = Lu & Lv
            den_uL = max(1, int(Lu.sum()))
            den_vL = max(1, int(Lv.sum()))
            lam_uL = bothL.sum() / den_vL
            lam_vL = bothL.sum() / den_uL
            lamL_emp = float(0.5 * (lam_uL + lam_vL))
            lamL_emp = float(np.clip(lamL_emp, 1e-6, 0.999999))

            # lambda_L = 2^(-1/(delta*theta)) -> theta = -log(2)/(delta*log(lambda_L))
            theta0 = float(-log2 / (delta0 * np.log(lamL_emp)))

        # final clamps for numerical stability
        theta0 = float(np.clip(theta0, 1e-6, 1e3))
        delta0 = float(np.clip(delta0, 1.0 + 1e-6, 1e3))

        self.set_parameters([theta0, delta0])
        return self.get_parameters()
