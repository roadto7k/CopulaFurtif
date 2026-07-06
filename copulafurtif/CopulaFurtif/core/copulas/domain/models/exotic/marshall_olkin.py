"""
Bivariate Marshall–Olkin copula (Joe, Sec. 4.14.1).

Stochastic representation (Exp(1) shocks), with 0 ≤ pi1, pi2 ≤ 1:
    X = min{ Z1/(1-pi1), Z12/pi1 }
    Y = min{ Z2/(1-pi2), Z12/pi2 }
and with unit-mean exponential margins, the (survival) copula CDF is (eq. 4.36):

    C(u, v; pi1, pi2) = min{ u^pi1, v^pi2 } * u^(1-pi1) * v^(1-pi2)
                      = { u * v^(1-pi2),   if u^pi1 ≤ v^pi2
                        { v * u^(1-pi1),   if u^pi1 > v^pi2

This copula has a singular component (mass on the curve u^pi1 = v^pi2).
Therefore a classical Lebesgue density does not exist everywhere; `get_pdf`
returns ONLY the absolutely-continuous part (a.e.), ignoring the singular mass.

Conditional copulas (eq. just after 4.36):
    C_{2|1}(v|u) = ∂C/∂u =
        { v^(1-pi2),                    if u^pi1 ≤ v^pi2
        { (1-pi1) * v * u^(-pi1),       if u^pi1 > v^pi2

    C_{1|2}(u|v) = ∂C/∂v =
        { (1-pi2) * u * v^(-pi2),       if u^pi1 ≤ v^pi2
        { u^(1-pi1),                    if u^pi1 > v^pi2
"""

from __future__ import annotations

import numpy as np
from numpy.random import default_rng
from scipy.stats import kendalltau

from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel, CopulaParameters
from CopulaFurtif.core.copulas.domain.models.mixins import ModelSelectionMixin, SupportsTailDependence


class MarshallOlkinCopula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    """Bivariate Marshall–Olkin copula (Joe, Sec. 4.14.1)."""

    def __init__(self):
        super().__init__()
        self.name = "Marshall–Olkin Copula"
        self.type = "marshall-olkin"
        self.default_optim_method  = None # to find best compromise
        # Marshall-Olkin contains a singular probability component concentrated
        # on the curve u**pi1 = v**pi2.
        self.is_absolutely_continuous = False
        self.has_singular_component = True

        # get_pdf returns only the absolutely-continuous component. Therefore it
        # cannot be used as a complete density likelihood.
        self.supports_density_likelihood = False

        # Book parameters: 0 ≤ pi1, pi2 ≤ 1.
        # In CopulaFurtif, bounds are *exclusive* (lo < val < hi), so we use (0,1) open interval
        # for numerical stability and to avoid degenerate limits (independence/comonotone).
        # Book parameters: 0 ≤ pi1, pi2 ≤ 1 (your archi handles open bounds if needed)
        self.init_parameters(
            CopulaParameters(
                values=np.array([0.5, 0.5], dtype=float),
                bounds=[(0.0, 1.0), (0.0, 1.0)],
                names=["pi1", "pi2"],
            )
        )

    # ---------------------------------------------------------------------
    # Core API
    # ---------------------------------------------------------------------
    def get_cdf(self, u, v, param=None):
        """Copula CDF C(u, v; pi1, pi2), eq. (4.36)."""
        if param is None:
            param = self.get_parameters()
        pi1, pi2 = float(param[0]), float(param[1])

        # IMPORTANT: allow exact boundaries 0 and 1 (CDF must satisfy C(u,0)=0, C(u,1)=u, etc.)
        u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        v = np.clip(np.asarray(v, dtype=float), 0.0, 1.0)

        # region split: u^pi1 ≤ v^pi2
        cond = (u ** pi1) <= (v ** pi2)

        c_cond = u * (v ** (1.0 - pi2))
        c_else = v * (u ** (1.0 - pi1))
        return np.where(cond, c_cond, c_else)

    def get_pdf(self, u, v, param=None):
        """
        Compute the absolutely-continuous component of the Marshall-Olkin
        copula density.

        Important
        ---------
        The Marshall-Olkin copula is not absolutely continuous. It contains a
        singular probability component concentrated on the curve

            u**pi1 = v**pi2.

        Consequently, this method does NOT represent the complete copula law.
        Its integral over the unit square is

            1 - singular_mass(param),

        rather than 1.

        The returned function is the two-dimensional Lebesgue density of the
        absolutely-continuous component only:

            c_ac(u, v) = (1-pi2) * v**(-pi2),
                if u**pi1 < v**pi2,

            c_ac(u, v) = (1-pi1) * u**(-pi1),
                if u**pi1 > v**pi2.

        On the singular curve u**pi1 = v**pi2, no ordinary two-dimensional
        density represents the singular probability mass. This method returns
        zero there.

        This density must not be used as a complete CMLE, MLE, IFM, AIC, or BIC
        likelihood contribution.

        Parameters
        ----------
        u : float or array-like
            First copula coordinate.

        v : float or array-like
            Second copula coordinate.

        param : array-like, optional
            Marshall-Olkin parameters [pi1, pi2]. If omitted, the current
            model parameters are used.

        Returns
        -------
        float or numpy.ndarray
            Absolutely-continuous density component evaluated pairwise at
            (u, v).
        """

    def singular_mass(self, param=None) -> float:
        """
        Return the total probability mass of the singular component.

        For the bivariate Marshall-Olkin copula,

            p_sing = pi1 * pi2 / (pi1 + pi2 - pi1 * pi2).

        The singular mass is concentrated on

            u**pi1 = v**pi2.

        For this copula, the singular mass is also equal to Kendall's tau.

        Parameters
        ----------
        param : array-like, optional
            Marshall-Olkin parameters [pi1, pi2]. If omitted, the current
            model parameters are used.

        Returns
        -------
        float
            Total probability mass of the singular component.
        """
        if param is None:
            param = self.get_parameters()

        pi1, pi2 = map(float, param)

        denominator = (
                pi1
                + pi2
                - pi1 * pi2
        )

        if denominator <= 0.0:
            return 0.0

        return float(
            pi1 * pi2 / denominator
        )

    def continuous_mass(self, param=None) -> float:
        """
        Return the total probability mass of the absolutely-continuous component.

        The integral of get_pdf over the unit square is

            1 - singular_mass.
        """
        return float(
            1.0 - self.singular_mass(param)
        )

    def sample(self, n: int, param=None, rng=None) -> np.ndarray:
        """
        Sample from the Marshall–Olkin copula using the common-shock Exp(1) representation
        (Joe, Sec. 4.14.1):

            X = min{ Z1/(1-pi1),  Z12/pi1 }
            Y = min{ Z2/(1-pi2),  Z12/pi2 }

        With Exp(1) margins, U = exp(-X), V = exp(-Y) are uniform(0,1) and follow the copula.
        """
        if param is None:
            param = self.get_parameters()
        pi1, pi2 = float(param[0]), float(param[1])

        if rng is None:
            rng = default_rng()

        n = int(n)
        # independent Exp(1)
        z1 = rng.exponential(scale=1.0, size=n)
        z2 = rng.exponential(scale=1.0, size=n)
        z12 = rng.exponential(scale=1.0, size=n)

        # bounds are exclusive, so pi1,pi2 in (0,1); still guard against numerical issues:
        eps = 1e-12
        pi1 = float(np.clip(pi1, eps, 1.0 - eps))
        pi2 = float(np.clip(pi2, eps, 1.0 - eps))

        x = np.minimum(z1 / (1.0 - pi1), z12 / pi1)
        y = np.minimum(z2 / (1.0 - pi2), z12 / pi2)

        u = np.exp(-x)
        v = np.exp(-y)

        # ensure open interval for downstream logs/tests
        u = np.clip(u, 1e-12, 1.0 - 1e-12)
        v = np.clip(v, 1e-12, 1.0 - 1e-12)
        return np.column_stack([u, v])

    # ---------------------------------------------------------------------
    # Dependence measures
    # ---------------------------------------------------------------------
    def kendall_tau(self, param=None):
        """
        Kendall's tau.

        For the bivariate Marshall-Olkin copula, Kendall's tau equals the
        probability mass of the singular component:

            tau = pi1 * pi2 / (pi1 + pi2 - pi1 * pi2).
        """
        return self.singular_mass(param)

    def blomqvist_beta(self, param=None) -> float:
        """
        Closed-form Blomqvist beta for Marshall–Olkin:
            β = 2^{min(pi1, pi2)} - 1
        """
        if param is None:
            param = self.get_parameters()
        pi1, pi2 = float(param[0]), float(param[1])
        beta = 2.0 ** (min(pi1, pi2)) - 1.0
        return float(np.clip(beta, -1.0, 1.0))

    def LTDC(self, param=None):
        """Lower tail dependence λ_L = 0 (for this MO copula)."""
        return 0.0

    def UTDC(self, param=None):
        """
        Upper tail dependence coefficient.

        For this Marshall–Olkin common-shock copula, a standard convention is:
            λ_U = min(pi1, pi2)
        """
        if param is None:
            param = self.get_parameters()
        pi1, pi2 = float(param[0]), float(param[1])
        return float(min(pi1, pi2))
    # ---------------------------------------------------------------------
    # Conditional distributions (partials of C)
    # ---------------------------------------------------------------------
    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """
        ∂C/∂u = C_{2|1}(v|u), eq. after (4.36).

        If u^pi1 ≤ v^pi2:  ∂C/∂u = v^(1-pi2)
        If u^pi1 >  v^pi2: ∂C/∂u = (1-pi1) v u^(-pi1)
        """
        if param is None:
            param = self.get_parameters()
        pi1, pi2 = float(param[0]), float(param[1])

        eps = 1e-12
        u = np.clip(np.asarray(u, dtype=float), eps, 1.0 - eps)
        v = np.clip(np.asarray(v, dtype=float), eps, 1.0 - eps)

        cond = (u ** pi1) <= (v ** pi2)
        p_cond = v ** (1.0 - pi2)
        p_else = (1.0 - pi1) * v * (u ** (-pi1))
        return np.where(cond, p_cond, p_else)

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """
        ∂C/∂v = C_{1|2}(u|v), eq. after (4.36).

        If u^pi1 ≤ v^pi2:  ∂C/∂v = (1-pi2) u v^(-pi2)
        If u^pi1 >  v^pi2: ∂C/∂v = u^(1-pi1)
        """
        if param is None:
            param = self.get_parameters()
        pi1, pi2 = float(param[0]), float(param[1])

        eps = 1e-12
        u = np.clip(np.asarray(u, dtype=float), eps, 1.0 - eps)
        v = np.clip(np.asarray(v, dtype=float), eps, 1.0 - eps)

        cond = (u ** pi1) <= (v ** pi2)
        q_cond = (1.0 - pi2) * u * (v ** (-pi2))
        q_else = u ** (1.0 - pi1)
        return np.where(cond, q_cond, q_else)

    def conditional_cdf_v_given_u(self, u, v, param=None):
        """P(V ≤ v | U = u) = ∂C/∂u(u,v), a.e."""
        return self.partial_derivative_C_wrt_u(u, v, param)

    def conditional_cdf_u_given_v(self, u, v, param=None):
        """P(U ≤ u | V = v) = ∂C/∂v(u,v), a.e."""
        return self.partial_derivative_C_wrt_v(u, v, param)

    # ---------------------------------------------------------------------
    # Initialization from data
    # ---------------------------------------------------------------------
    def init_from_data(self, u, v):
        """
        Conservative init from empirical Kendall's tau.

        Uses symmetric guess pi1=pi2=a by inverting:
            τ = a/(2-a)  =>  a = 2τ/(1+τ)
        """
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)

        tau_emp, _ = kendalltau(u, v)
        if not np.isfinite(tau_emp):
            tau_emp = 0.0
        tau_emp = float(np.clip(tau_emp, 0.0, 0.999))

        a = (2.0 * tau_emp) / (1.0 + tau_emp) if tau_emp > 1e-12 else 0.05
        a = float(np.clip(a, 1e-3, 1.0 - 1e-3))
        return np.array([a, a], dtype=float)

    # ---------------------------------------------------------------------
    # Diagnostics not implemented (match Frank behavior)
    # ---------------------------------------------------------------------
    def IAD(self, data):
        print(f"[INFO] IAD not implemented for {self.name}.")
        return np.nan

    def AD(self, data):
        print(f"[INFO] AD not implemented for {self.name}.")
        return np.nan