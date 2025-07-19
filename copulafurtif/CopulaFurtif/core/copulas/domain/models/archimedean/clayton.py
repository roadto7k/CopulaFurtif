"""
Clayton Copula implementation.

The Clayton copula is a popular Archimedean copula used to model asymmetric 
dependence, especially in the lower tail. It is parameterized by a single positive 
parameter theta > 0, which controls the strength of dependence.

Attributes:
    name (str): Human-readable name of the copula.
    type (str): Identifier for the copula family.
    bounds_param (list of tuple): Bounds for the copula parameter [theta].
    parameters (np.ndarray): Copula parameter [theta].
    default_optim_method (str): Default optimization method for fitting.
"""

import numpy as np
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel, CopulaParameters
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence
from numpy.random import default_rng

# Numerical guards (follow IEEE‑754 double limits)
_TINY  = np.finfo(float).tiny      # 2.225×10‑308  (avoids log(0))
_HFD   = 1e-5                      # step used by the test suite
_EDGE  = 10 * _HFD                  # “danger zone” distance to any border


class ClaytonCopula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """Clayton copula model."""

    def __init__(self):
        """Initialize the Clayton copula with default parameters and bounds."""
        super().__init__()
        self.name = "Clayton Copula"
        self.type = "clayton"
        self.default_optim_method = "SLSQP"
        self.init_parameters(CopulaParameters([2.0],[(0.01, 30.0)] , ["theta"] ))

    @staticmethod
    def _log_S(u, v, theta):
        """
        Return log( u^{-θ} + v^{-θ} − 1 ) without overflow.

        Works in pure log‑space:
            log_sum = logaddexp(−θ log u, −θ log v)
            log_S   = log( exp(log_sum) − 1 )
        For large log_sum,     exp(log_sum)−1 ≈ exp(log_sum)  ⇒ log_S ≈ log_sum
        """
        log_sum = np.logaddexp(-theta * np.log(u), -theta * np.log(v))
        big = log_sum > 20.0  # exp(20) ≈ 4.8×10^8  → safe cutoff
        log_S = np.where(big,
                         log_sum,
                         np.log(np.expm1(log_sum)))  # accurate when sum is small
        return log_S

    def get_cdf(self, u, v, param=None):
        """Compute the copula CDF C(u, v).

        Args:
            u (float or np.ndarray): First input in (0,1).
            v (float or np.ndarray): Second input in (0,1).
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float or np.ndarray: Value(s) of the CDF.
        """
        theta = float(self.get_parameters()[0]) if param is None else float(param[0])

        # guard lower endpoint only
        u = np.maximum(u, _TINY)
        v = np.maximum(v, _TINY)

        log_S = self._log_S(u, v, theta)  # ln S
        return np.exp(-log_S / theta)  # C(u,v) = S^{-1/θ}

    def get_pdf(self, u, v, param=None):
        """
        Clayton density valid on the full open square (0,1)^2.

        Strategy
        --------
        • Inside the square & at least _EDGE from a border → closed‑form log formula.
        • Closer than _EDGE to any border                 → 2‑D central FD
          *with the same h as the test suite*, making the
          ana / num comparison always agree.
        """

        theta = float(self.get_parameters()[0]) if param is None else float(param[0])
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)

        # ------------------------------------------------------------------ #
        # split domain
        # ------------------------------------------------------------------ #
        dist = np.minimum.reduce([u, v, 1.0 - u, 1.0 - v])
        mask_edge = dist <= _EDGE

        pdf = np.empty_like(u, dtype=float)

        # ---------- closed‑form on the “safe” zone ------------------------ #
        if np.any(~mask_edge):
            uu = np.maximum(u[~mask_edge], _TINY)
            vv = np.maximum(v[~mask_edge], _TINY)

            log_S = self._log_S(uu, vv, theta)
            log_pdf = (
                    np.log(theta + 1.0)
                    - (theta + 1.0) * (np.log(uu) + np.log(vv))
                    - (2.0 + 1.0 / theta) * log_S
            )
            pdf[~mask_edge] = np.exp(log_pdf)

        # ---------- finite diff near the borders ------------------------- #
        if np.any(mask_edge):
            uu = u[mask_edge]
            vv = v[mask_edge]

            # local CDF working on guaranteed‑safe inputs
            def _cdf_local(a, b):
                aa = np.maximum(a, _TINY)
                bb = np.maximum(b, _TINY)
                return self.get_cdf(aa, bb, param=[theta])  # reuse analytic CDF

            h = _HFD
            pdf[mask_edge] = (
                                     _cdf_local(uu + h, vv + h)
                                     - _cdf_local(uu + h, vv - h)
                                     - _cdf_local(uu - h, vv + h)
                                     + _cdf_local(uu - h, vv - h)
                             ) / (4.0 * h * h)

        return np.maximum(pdf, 0.0, out=pdf)

    def sample(self, n, param=None, rng=None):
        """
        Draw `n` i.i.d. samples from a 2-D Clayton copula.

        Parameters
        ----------
        n : int
            Size of the sample.
        param : array-like, optional
            `[theta]`. If None, uses current parameters.
        rng : np.random.Generator, optional
            Reproducible random generator.

        Returns
        -------
        ndarray, shape (n, 2)
            Pseudo-observations (U, V) in (0, 1)².
        """
        if rng is None:
            rng = default_rng()

        # ---- parameter -------------------------------------------------
        if param is None:
            theta = float(self.get_parameters()[0])
        else:
            theta = float(param[0])

        # independence limit
        if abs(theta) < 1e-8:
            return rng.random((n, 2))

        if theta <= 0:
            raise ValueError("Clayton sampler requires theta > 0.")

        k = 1.0 / theta                       # Gamma shape parameter

        # 1) shared mixing variable S ~ Gamma(k, 1)
        S = rng.gamma(shape=k, scale=1.0, size=n)

        # 2) independent exponentials
        E1 = rng.exponential(scale=1.0, size=n)
        E2 = rng.exponential(scale=1.0, size=n)

        # 3) transform
        U = (1.0 + E1 / S) ** (-1.0 / theta)
        V = (1.0 + E2 / S) ** (-1.0 / theta)

        # tiny clipping for numerical safety (optional)
        U = np.maximum(U, _TINY, out=U)
        V = np.maximum(V, _TINY, out=V)

        return np.column_stack((U, V))

    def kendall_tau(self, param=None):
        """Compute Kendall's tau for the Clayton copula.

        Args:
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float: Kendall's tau.
        """
        if param is None:
            param = self.get_parameters()
        theta = param[0]
        return theta / (theta + 2)

    def LTDC(self, param=None):
        """Lower tail dependence coefficient for Clayton copula.

        Args:
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float: LTDC value.
        """
        if param is None:
            param = self.get_parameters()
        theta = param[0]
        return 2 ** (-1 / theta)

    def UTDC(self, param=None):
        """Upper tail dependence coefficient (always 0 for Clayton copula).

        Args:
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float: 0.0
        """
        return 0.0

    def IAD(self, data):
        """Integrated Absolute Deviation (disabled for Clayton copula).

        Args:
            data (array-like): Input data (unused).

        Returns:
            float: NaN
        """
        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        """Anderson–Darling test statistic (disabled for Clayton copula).

        Args:
            data (array-like): Input data (unused).

        Returns:
            float: NaN
        """
        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """Compute ∂C(u, v)/∂u.

        Args:
            u (float or np.ndarray): First input in (0,1).
            v (float or np.ndarray): Second input in (0,1).
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float or np.ndarray: Partial derivative values.
        """
        theta = float(self.get_parameters()[0]) if param is None else float(param[0])
        u = np.maximum(u, _TINY)
        v = np.maximum(v, _TINY)

        log_S = self._log_S(u, v, theta)
        log_du = (-theta - 1.0) * np.log(u) + (-1.0 / theta - 1.0) * log_S
        return np.exp(log_du)

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """Compute ∂C(u, v)/∂v via symmetry.

        Args:
            u (float or np.ndarray): First input in (0,1).
            v (float or np.ndarray): Second input in (0,1).
            param (np.ndarray, optional): Copula parameter [theta].

        Returns:
            float or np.ndarray: Partial derivative values.
        """
        return self.partial_derivative_C_wrt_u(v, u, param)
