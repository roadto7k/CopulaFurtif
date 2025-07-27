import math
from typing import Union

import numpy as np
from numpy.random import default_rng
from scipy import integrate

# ─────────── BB2 JAX implementation ─────────────────────
import math
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as jtu
from functools import partial

from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel, CopulaParameters
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence
from CopulaFurtif.core.copulas.domain.models.archimedean.BB1 import BB1Copula

# --- helpers ------------------------------------------------------------
MAX_EXP = 700.0                       # e^700 ≈ 1e304, still finite
_LOG_MAX = 700.0          # safe upper bound for exp() on 64-bit floats

def _safe_exp(x):
    """exp(x) but hard‑clipped to avoid overflow"""
    return np.exp(np.minimum(x, MAX_EXP))

def _safe_expm1(x):
    """expm1(x) sans overflow pour x très grand."""
    return jnp.where(x < _LOG_MAX, jnp.expm1(x), jnp.exp(jnp.clip(x, None, _LOG_MAX)))

def _logsumexp_minus1(a, b):
    """
    log( e^a + e^b - 1 ) de façon stable
    (les deux a,b ≥ 0 dans BB2 ⇒ pas de signe négatif).
    """
    m = jnp.maximum(a, b)
    return m + jnp.log(jnp.exp(a - m) + jnp.exp(b - m) - jnp.exp(-m))

def _log_expm1(x):
    """log(expm1(x))  stable jusqu’à x≈1e4."""
    return jnp.where(x < 1e-2, jnp.log(jnp.expm1(x)), x + jnp.log1p(-jnp.exp(-x)))

def _logsumexp_two(a, b):
    """log(e^a + e^b)  sans nan même si a ou b = ±inf"""
    m = jnp.maximum(a, b)
    return jnp.where(jnp.isfinite(m), m + jnp.log(jnp.exp(a - m) + jnp.exp(b - m)), m)


@partial(jax.jit, static_argnums=(0, 4, 5))
def _bisect_root(f, lo, hi, args, max_iter: int = 40, eps: float = 1e-12):
    """Batched bisection root finder.
    Solves f(x, *args) = 0 for x in (lo, hi).  `lo`,`hi` and the result
    are arrays of the same length (vectorised).
    """

    def body(val):
        lo, hi = val
        mid = 0.5 * (lo + hi)
        sign = jnp.sign(f(mid, *args))  # >0 ? move hi, else move lo
        lo = jnp.where(sign > 0, lo, mid)
        hi = jnp.where(sign > 0, mid, hi)
        return lo, hi

    lo, hi = jax.lax.fori_loop(
        0, max_iter, lambda _, val: body(val), (lo, hi)
    )
    return 0.5 * (lo + hi)


class BB2Copula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """
    BB2 Copula (Survival version of BB1 Copula).

    This copula is the survival (180-degree rotated) version of BB1, used to
    model upper tail dependence with two parameters.

    Attributes:
        name (str): Human-readable name of the copula.
        type (str): Identifier for the copula family.
        bounds_param (list of tuple): Bounds for copula parameters [theta, delta].
        parameters (np.ndarray): Current copula parameters.
        default_optim_method (str): Optimization method for parameter fitting.
    """

    def __init__(self):
        """Initialize BB2 copula with default parameters."""
        super().__init__(use_jax=True)                  # backend JAX
        self.name = "BB2 Copula"
        self.type = "bb2"
        self.default_optim_method = "Powell"
        self.bounds_param = [(0.05, 10.0), (1.0, 10.0)]
        self.init_parameters(CopulaParameters([2, 1.5],self.bounds_param, ["theta", "delta"] ))

    # ----- PyTree plumbing -----
    def tree_flatten(self):
        """
        Flatten this BB2Copula instance into JAX‐traceable leaves and static auxiliary data.

        The returned leaves are the copula parameters as a JAX array, and the auxiliary
        data is the list of parameter bounds.

        Returns:
            tuple:
                children (tuple[jnp.ndarray]): A one‐element tuple containing the JAX array
                    of copula parameters.
                aux (tuple[list[tuple[float, float]]]): A one‐element tuple containing the
                    parameter bounds.
        """
        params = jnp.asarray(self.get_parameters())  # leaf
        return (params,), (self.get_bounds(),)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """
        Reconstruct a BB2Copula instance from flattened leaves and auxiliary data.

        Args:
            aux (tuple[list[tuple[float, float]]]): A one‐element tuple containing the
                parameter bounds.
            children (tuple[jnp.ndarray]): A one‐element tuple containing the JAX array
                of copula parameters.

        Returns:
            BB2Copula: A new instance whose parameters and bounds have been restored.
        """
        bounds, = aux
        params, = children

        obj = cls()
        obj.set_bounds(bounds)  # setter -> CopulaParameters
        obj.set_parameters(params)  # setter -> jnp array
        return obj

    # ----- Archimedean generator and safe inverse -----
    @staticmethod
    def _phi_inv(u: jnp.ndarray, theta: float, delta: float) -> jnp.ndarray:
        """
        Compute the inverse Archimedean generator φ⁻¹(u) for the BB2 copula.

        Implements φ⁻¹(u) = δ·(expm1(−θ·log(u))) in a numerically stable way, clipping
        large exponents to avoid overflow.

        Args:
            u (jnp.ndarray): Values in the unit interval (0, 1).
            theta (float): Copula parameter θ > 0.
            delta (float): Copula parameter δ > 0.

        Returns:
            jnp.ndarray: The generator inverse values, φ⁻¹(u) ≥ 0.
        """
        # s = -θ log u  ≥ 0
        s = -theta * jnp.log(u)

        # pour s > 50, exp(s) >> 1 expm1(s) ≈ exp(s)
        small = s < 50.0
        expm1_s = jnp.where(small, jnp.expm1(s), jnp.exp(jnp.clip(s, None, _LOG_MAX)))

        return delta * expm1_s

    @partial(jax.jit, static_argnums=0)
    def get_cdf(self, u, v, param=None):
        """
        Evaluate the BB2 copula cumulative distribution function C(u, v).

        Computes

            C(u,v;θ,δ) = [1 + δ⁻¹ · log(e^{w_u} + e^{w_v} − 1)]^{−1/θ}

        where w_u = δ(u^{−θ}−1), w_v = δ(v^{−θ}−1), implemented fully in log‐space
        for numerical stability.

        Args:
            u (array_like): First marginal values in (0, 1).
            v (array_like): Second marginal values in (0, 1).
            param (Optional[Sequence[float]]): Sequence [θ, δ] of copula parameters;
                if None, uses this instance’s stored parameters.

        Returns:
            jnp.ndarray: The CDF values C(u, v), same shape as inputs.
        """

        if param is None:
            theta, delta = self.get_parameters()
        else:
            theta, delta = param

        eps = 1e-15
        u = jnp.clip(self._to_backend(u), eps, 1 - eps)
        v = jnp.clip(self._to_backend(v), eps, 1 - eps)

        # 3) A = δ (u^{-θ} - 1), B = δ (v^{-θ} - 1)
        gu = jnp.expm1(-theta * jnp.log(u))  # = u^{-θ} - 1
        gv = jnp.expm1(-theta * jnp.log(v))  # = v^{-θ} - 1
        A = delta * gu
        B = delta * gv

        # 4) logS = log( e^A + e^B - 1 )
        logS = _logsumexp_minus1(A, B)

        # 5) L1 = δ⁻¹·logS  →  logL1 = logS - log(delta)
        logL1 = logS - jnp.log(delta)

        # 6) log(1 + L1) stable
        logA = jnp.log1p(jnp.exp(logL1))

        # 7) CDF
        return jnp.exp(-logA / theta)

    @partial(jax.jit, static_argnums=0)
    def get_pdf(self, u, v, param=None):
        """
        Compute the BB2 copula density c(u, v).

        This method accepts scalars or arrays; if arrays are provided,
        standard NumPy broadcasting rules apply and the result has
        the broadcasted shape.

        Args:
            u (array_like): First margin values in (0, 1).
            v (array_like): Second margin values in (0, 1).
            param (Sequence[float] or None): Optional [theta, delta].
                If None, uses this instance’s stored parameters.

        Returns:
            jnp.ndarray: Copula density values with the same
                         broadcasted shape as u and v.
        """
        # 1. Extract parameters ------------------------------------------------
        if param is None:
            theta, delta = self.get_parameters()
        else:
            theta, delta = param

        eps = 1e-15
        u = jnp.clip(self._to_backend(u), eps, 1 - eps)
        v = jnp.clip(self._to_backend(v), eps, 1 - eps)

        A = delta * (u ** (-theta) - 1.0)
        B = delta * (v ** (-theta) - 1.0)

        logS = _logsumexp_minus1(A, B)  # log E
        T = 1.0 + logS / delta  # >0

        # softmax-like stable weights
        p_u = jnp.exp(A - logS)
        p_v = jnp.exp(B - logS)

        # powers that appear repeatedly
        u_pow = u ** (-theta - 1.0)
        v_pow = v ** (-theta - 1.0)

        # assemble the density
        prefactor = jnp.exp(-(1.0 / theta + 2.0) * jnp.log(T))  # T^{-1/θ-2}
        bracket = (1.0 + theta) + theta * delta * T  # [...]
        pdf = prefactor * p_u * p_v * u_pow * v_pow * bracket

        return pdf

    # @partial(jax.jit, static_argnums=0)
    # def get_pdf(self, u, v, param=None):
    #     """
    #     Auto‑diff BB2 copula density c(u,v) via reverse‑over‑reverse.
    #
    #     Returns a JAX array of the same broadcasted shape as u, v.
    #     """
    #     # 1. parameters
    #     if param is None:
    #         theta, delta = self.get_parameters()
    #     else:
    #         theta, delta = param
    #
    #     # 2. clip & broadcast
    #     eps = 1e-6
    #     u_b, v_b = jnp.broadcast_arrays(
    #         jnp.clip(self._to_backend(u), eps, 1 - eps),
    #         jnp.clip(self._to_backend(v), eps, 1 - eps),
    #     )
    #     u_flat = u_b.ravel()
    #     v_flat = v_b.ravel()
    #
    #     # 3. scalar CDF wrapper
    #     def _cdf_scalar(uu, vv):
    #         return self.get_cdf(uu, vv, (theta, delta))
    #
    #     # 4. mixed second derivative via reverse‑over‑reverse
    #     scalar_pdf = jax.grad(jax.grad(_cdf_scalar, argnums=0), argnums=1)
    #
    #     # 5. vectorize & reshape
    #     pdf_flat = jax.vmap(scalar_pdf)(u_flat, v_flat)
    #     return pdf_flat.reshape(u_b.shape)

    @partial(jax.jit, static_argnums=(0, 2))
    def kendall_tau(self, param=None, n_grid: int = 400):
        """Return Kendall's tau computed numerically via the identity
            τ = 4 ∬_{[0,1]²} C(u,v) · c(u,v) du dv − 1

        The double integral is approximated on an ``n_grid × n_grid`` tensor
        product grid (Riemann midpoint rule).  Works for scalars or arrays
        thanks to JAX vectorisation; compiled with XLA for speed.

        Args
        ----
        param : Sequence[float] | None
            Copula parameters.  If *None*, uses the instance’s current values.
        n_grid : int
            Number of quadrature points per axis (default 400 ⇒ 1.6 e5 evals).

        Returns
        -------
        float
            Kendall’s τ estimate, high‑precision (error ~ O(n_grid⁻²)).
        """
        # ── parameters -------------------------------------------------------
        if param is None:
            theta, delta = self.get_parameters()
        else:
            theta, delta = param

        eps = 1e-6  # keep away from log(0)
        u = jnp.linspace(eps, 1.0 - eps, n_grid)
        v = jnp.linspace(eps, 1.0 - eps, n_grid)
        U, V = jnp.meshgrid(u, v, indexing="ij")  # shape (n, n)

        # ── evaluation (vectorised, no Python loop) -------------------------
        def _eval(f):  # helper for C & pdf
            flat = jax.vmap(lambda uu, vv: f(uu, vv, (theta, delta)))(
                U.ravel(), V.ravel()
            )
            return flat.reshape((n_grid, n_grid))

        C_vals = _eval(self.get_cdf)
        pdf_vals = _eval(self.get_pdf)

        # ── midpoint rule ---------------------------------------------------
        delta_uv = (1.0 - 2 * eps) / n_grid  # step in each dimension
        integral = jnp.sum(C_vals * pdf_vals) * delta_uv ** 2

        return 4.0 * integral - 1.0

    @partial(jax.jit, static_argnums=(0, 1))
    def sample(self, n: int, key=None, param=None,
               eps: float = 1e-12, max_iter: int = 40):
        """Draw `n` iid pairs (U,V) ~ BB2 using JAX.

        Parameters
        ----------
        n        : int
        key      : jax.random.PRNGKey
        param    : None | (theta, delta)
        eps      : float     (clipping to keep away from 0,1)
        max_iter : int       (bisection depth ~ 2⁻ᵏ accuracy)
        """
        # parameters ----------------------------------------------------------
        if param is None:
            theta, delta = self.get_parameters()
        else:
            theta, delta = param
        theta, delta = map(float, (theta, delta))

        if key is None:
            key = jax.random.PRNGKey(0)

        # split RNG
        key_u, key_p = jax.random.split(key)

        # 1) U ~ Unif,   P ~ Unif
        u = jax.random.uniform(key_u, (n,), minval=eps, maxval=1 - eps)
        p = jax.random.uniform(key_p, (n,), minval=eps, maxval=1 - eps)

        # 2) conditional CDF F_{V|U}(v)  (normalised ∂C/∂u)
        def cond_cdf(v, u, theta, delta):
            num = self.partial_derivative_C_wrt_u(u, v, (theta, delta))
            den = self.partial_derivative_C_wrt_u(u, 1.0, (theta, delta))
            return num / den

        # root function  g(v) = F(v) - p  (want zero)
        f_root = lambda v, u, p, theta, delta: cond_cdf(v, u, theta, delta) - p

        # 3) batched bisection ------------------------------------------------
        lo = jnp.full(n, eps)
        hi = jnp.full(n, 1.0 - eps)
        v = _bisect_root(f_root, lo, hi, (u, p, theta, delta),
                         max_iter=max_iter, eps=eps)

        # 4) pack & return
        return jnp.stack((u, v), axis=1)

    def LTDC(self, param=None):
        """
        Compute lower tail dependence coefficient.

        Args:
            param (np.ndarray, optional): Parameters [theta, delta].

        Returns:
            float: Lower tail dependence.
        """
        return 1.0

    def UTDC(self, param=None):
        """
        Compute upper tail dependence coefficient.

        Args:
            param (np.ndarray, optional): Parameters [theta, delta].

        Returns:
            float: Upper tail dependence.
        """
        return 0.0

    @partial(jax.jit, static_argnums=0)
    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """
        Return ∂C/∂u for the BB2 copula (closed‑form, NaN‑free).

        Args
        ----
        u, v : array‑like in (0,1)
        param: optional (theta, delta)

        Returns
        -------
        jnp.ndarray with the broadcasted shape of (u, v)
        """
        # 1) parameters -------------------------------------------------------
        theta, delta = param if param is not None else self.get_parameters()

        # 2) clip & broadcast -------------------------------------------------
        eps = 1e-15
        u_b, v_b = jnp.broadcast_arrays(
            jnp.clip(self._to_backend(u), eps, 1.0 - eps),
            jnp.clip(self._to_backend(v), eps, 1.0 - eps),
        )

        # 3) core quantities --------------------------------------------------
        A = delta * (u_b ** (-theta) - 1.0)
        B = delta * (v_b ** (-theta) - 1.0)

        logS = _logsumexp_minus1(A, B)  # log(e^A + e^B − 1)
        T = 1.0 + logS / delta  # > 0
        Cuv = jnp.exp(-jnp.log(T) / theta)  # C(u,v)

        p_u = jnp.exp(A - logS)  # weight e^A / (e^A+e^B−1)
        u_pow = u_b ** (-theta - 1.0)

        # 4) derivative -------------------------------------------------------
        dC_du = Cuv * p_u * u_pow / T
        return dC_du



    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """Compute the partial derivative ∂C_BB2/∂v at (u, v).

        By exchangeability of the Archimedean base copula, we can
        reuse ∂C/∂u by swapping arguments:
        ∂C_BB2/∂v(u,v) = ∂C_BB2/∂u(v,u).

        Args:
            u (float or np.ndarray): U-coordinates in [0, 1].
            v (float or np.ndarray): V-coordinates in [0, 1].
            param (Sequence[float], optional):
                [theta, delta] copula parameters. If None, uses
                the model’s current parameters.

        Returns:
            float or np.ndarray: The value(s) of ∂C/∂v at the given points.
        """
        return self.partial_derivative_C_wrt_u(v, u, param)

    def IAD(self, data):
        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan

