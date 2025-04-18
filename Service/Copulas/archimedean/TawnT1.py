# Service/Copulas/archimedean/TawnT1.py

import numpy as np
from Service.Copulas.archimedean.Tawn import TawnCopula

class TawnT1Copula(TawnCopula):
    """
    Tawn Type‑1 (asymmetric logistic).
    Paramètres libres : [theta, alpha], psi2=1 fixe.
    Swap (u,v) pour que t = -ln(u)/( - ln(u)-ln(v) ).
    """
    def __init__(self):
        super().__init__()
        self.type = "tawn1"
        self.name = "Tawn Type-1 Copula"
        # theta>=1, alpha∈[0,1]
        self.bounds_param = [(1.0, None), (0.0, 1.0)]
        # paramètres par défaut [θ=2.0, α=0.5]
        self.parameters = np.array([2.0, 0.5])

    @property
    def parameters(self) -> np.ndarray:
        # on expose seulement [theta, alpha]
        return np.array([self._parameters[0], self._psi1])

    @parameters.setter
    def parameters(self, param: np.ndarray):
        if param is None:
            return
        theta, alpha = param
        self._psi1 = alpha
        self._psi2 = 1.0
        self._parameters = np.array([theta, self._psi1, self._psi2])

    def get_cdf(self, u: np.ndarray, v: np.ndarray, param: np.ndarray) -> np.ndarray:
        """
        Compute the Type‑1 CDF C(u, v) by expanding to the full 3‑parameter Tawn form
        and swapping inputs so that the Pickands argument t = -ln(u)/(-ln(u)-ln(v)).

        If `param` already has length 3, assume this is an *internal* call from the
        base class (e.g. for derivatives), so forward it directly without swapping.

        Args:
            u: array of U values in [0,1]
            v: array of V values in [0,1]
            param: either
                   - length‑2 array [theta, alpha] for external use, or
                   - length‑3 array [theta, psi1, psi2] for internal recursion.

        Returns:
            CDF value(s) C(u,v) for the asymmetric logistic (Type‑1) copula.
        """
        # Internal call: bypass the wrapper logic
        if len(param) == 3:
            return super().get_cdf(u, v, param)

        # External call: unpack [theta, alpha], fix psi2=1, swap (u,v)
        theta, alpha = param
        full = np.array([theta, alpha, 1.0])
        return super().get_cdf(v, u, full)


    def get_pdf(self, u: np.ndarray, v: np.ndarray, param: np.ndarray) -> np.ndarray:
        """
        Compute the Type‑1 joint density c(u, v) by expanding to full parameters
        and swapping inputs.  Same bypass logic as get_cdf.

        - If param length == 3: direct call.
        - Else: unpack [theta, alpha], build [theta, alpha, 1], swap.
        """
        if len(param) == 3:
            return super().get_pdf(u, v, param)

        theta, alpha = param
        full = np.array([theta, alpha, 1.0])
        return super().get_pdf(v, u, full)


    def partial_derivative_C_wrt_u(
        self, u: np.ndarray, v: np.ndarray, param: np.ndarray
    ) -> np.ndarray:
        """
        Analytical ∂C/∂u for Type‑1 via wrapper:

        - Internal: if param length == 3, call base class ∂C/∂u directly.
        - External: unpack [θ,α], fix [θ,α,1], then note that
          ∂/∂u of wrapper equals ∂/∂v of base on swapped inputs.
        """
        if len(param) == 3:
            return super().partial_derivative_C_wrt_u(u, v, param)

        theta, alpha = param
        full = np.array([theta, alpha, 1.0])
        # External ∂/∂u = base ∂/∂v evaluated at (v,u)
        return super().partial_derivative_C_wrt_v(v, u, full)


    def partial_derivative_C_wrt_v(
        self, u: np.ndarray, v: np.ndarray, param: np.ndarray
    ) -> np.ndarray:
        """
        Analytical ∂C/∂v for Type‑1 via wrapper:

        - Internal: pass through.
        - External: unpack [θ,α], build full, then
          ∂/∂v of wrapper = ∂/∂u of base on swapped inputs.
        """
        if len(param) == 3:
            return super().partial_derivative_C_wrt_v(u, v, param)

        theta, alpha = param
        full = np.array([theta, alpha, 1.0])
        # External ∂/∂v = base ∂/∂u evaluated at (v,u)
        return super().partial_derivative_C_wrt_u(v, u, full)
