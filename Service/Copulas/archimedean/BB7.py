import numpy as np
from scipy.optimize import brentq

from Service.Copulas.base import BaseCopula

class BB7Copula(BaseCopula):
    """
    BB7 Copula class (Joe-Clayton)

    Parameters
    ----------
    theta : float
        Joe parameter (θ > 0)
    delta : float
        Clayton parameter (δ > 0)

    The BB7 copula is defined via an Archimedean generator:

        φ(t) = φ_C(φ_J(t)),
    where
        φ_J(t) = -ln(1 - (1 - t)**θ)  (Joe generator),
        φ_C(s) = (s**(-δ) - 1)/δ       (Clayton generator).

    Thus:
        φ(t) = ((-ln(1 - (1 - t)**θ))**(-δ) - 1)/δ,
    and its pseudo-inverse:
        φ^{-1}(s) = 1 - (1 - exp(-(1 + δ s)**(-1/δ)))**(1/θ).

    Methods
    -------
    get_cdf(u, v, param)
        CDF via φ^{-1}(φ(u) + φ(v)).
    get_pdf(u, v, param)
        PDF via finite-difference approximation of ∂²C/∂u∂v.
    sample(n, param)
        Conditional inversion sampling.
    LTDC(param), UTDC(param)
        Numeric tail-dependence estimates.
    """
    def __init__(self):
        super().__init__()
        self.type = "bb7"
        self.name = "BB7 Copula"
        # theta > 0, delta > 0
        self.bounds_param = [(1e-6, None), (1e-6, None)]
        self.parameters = np.array([1.0, 1.0])
        self.default_optim_method = "SLSQP"

    def _phi(self, t, param):
        """
        φ(t) = φ_C(φ_J(t)), where
          φ_J(t) = 1 - (1 - t)**θ      (non‑strict Joe generator)
          φ_C(s) = (s**(-δ) - 1)/δ     (strict Clayton generator)
        """
        theta, delta = param
        eps = 1e-12
        t = np.clip(t, eps, 1 - eps)
        phiJ = 1 - (1 - t)**theta
        return (phiJ**(-delta) - 1) / delta

    def _phi_inv(self, s, param):
        """
        φ⁻¹(s) = φ_J⁻¹(φ_C⁻¹(s)), where
          φ_C⁻¹(s) = (1 + δ s)**(-1/δ)
          φ_J⁻¹(u) = 1 - (1 - u)**(1/θ)
        """
        theta, delta = param
        s = np.maximum(s, 0)
        phiC_inv = (1 + delta * s)**(-1.0 / delta)
        return 1 - (1 - phiC_inv)**(1.0 / theta)

    def get_cdf(self, u, v, param):
        """
        Compute C(u,v) = φ^{-1}(φ(u) + φ(v)).
        """
        phi_u = self._phi(u, param)
        phi_v = self._phi(v, param)
        return self._phi_inv(phi_u + phi_v, param)

    def get_pdf(self, u, v, param):
        """
        PDF via finite-difference approximation of the mixed derivative ∂²C/∂u∂v.
        """
        eps = 1e-6
        c = self.get_cdf
        # central difference approximation
        return (
            c(u + eps, v + eps, param)
            - c(u + eps, v - eps, param)
            - c(u - eps, v + eps, param)
            + c(u - eps, v - eps, param)
        ) / (4 * eps**2)

    def partial_derivative_C_wrt_u(self, u, v, param):
        """
        ∂C/∂u via central difference.
        """
        eps = 1e-6
        c = self.get_cdf
        return (c(u + eps, v, param) - c(u - eps, v, param)) / (2 * eps)

    def partial_derivative_C_wrt_v(self, u, v, param):
        """
        ∂C/∂v via central difference.
        """
        eps = 1e-6
        c = self.get_cdf
        return (c(u, v + eps, param) - c(u, v - eps, param)) / (2 * eps)

    def conditional_cdf_u_given_v(self, u, v, param):
        """
        P(U ≤ u | V = v) = ∂C/∂v / ∂C(1,v)/∂v.
        """
        num = self.partial_derivative_C_wrt_v(u, v, param)
        den = self.partial_derivative_C_wrt_v(1.0, v, param)
        return num / den

    def conditional_cdf_v_given_u(self, u, v, param):
        """
        P(V ≤ v | U = u) = ∂C/∂u / ∂C(u,1)/∂u.
        """
        num = self.partial_derivative_C_wrt_u(u, v, param)
        den = self.partial_derivative_C_wrt_u(u, 1.0, param)
        return num / den

    def sample(self, n, param):
        """
        Generate samples by conditional inversion: for each u ~ U(0,1),
        find v s.t. P(V ≤ v | U = u) = p.
        """
        samples = np.empty((n, 2))
        for i in range(n):
            u = np.random.rand()
            p = np.random.rand()
            root = brentq(
                lambda v: self.conditional_cdf_v_given_u(u, v, param) - p,
                1e-6,
                1 - 1e-6,
            )
            samples[i, 0] = u
            samples[i, 1] = root
        return samples

    def LTDC(self, param):
        """
        Approximate lower tail dependence λ_L = lim_{u→0} C(u,u)/u.
        """
        eps = 1e-6
        return self.get_cdf(eps, eps, param) / eps

    def UTDC(self, param):
        """
        Approximate upper tail dependence λ_U = 2 - lim_{u→1} (1 - 2u + C(u,u))/(1 - u).
        """
        eps = 1e-6
        u = 1 - eps
        return 2 - (1 - 2 * u + self.get_cdf(u, u, param)) / eps

    def kendall_tau(self, param):
        """
        Kendall's tau not implemented for BB7.
        """
        raise NotImplementedError("Kendall's tau not implemented for BB7.")
