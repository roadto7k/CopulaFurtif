# Service/Copulas/archimedean/TawnT2.py

import numpy as np
from Service.Copulas.archimedean.Tawn import TawnCopula

class TawnT2Copula(TawnCopula):
    """
    Tawn Type‑2 (asymmetric mixed).
    Paramètres libres : [theta, beta], psi1=1 fixe.
    Swap (u,v) pour que t = -ln(u)/(…).
    """
    def __init__(self):
        super().__init__()
        self.type = "tawn2"
        self.name = "Tawn Type-2 Copula"
        self.bounds_param = [(1.0, None), (0.0, 1.0)]
        self.parameters = np.array([2.0, 0.5])

    @property
    def parameters(self) -> np.ndarray:
        # on expose seulement [theta, beta=psi2]
        return np.array([self._parameters[0], self._psi2])

    @parameters.setter
    def parameters(self, param: np.ndarray):
        if param is None:
            return
        theta, beta = param
        # on fixe psi1=1.0, psi2=beta
        self._psi1 = 1.0
        self._psi2 = beta
        self._parameters = np.array([theta, self._psi1, self._psi2])

    def get_cdf(self, u: np.ndarray, v: np.ndarray, param: np.ndarray) -> np.ndarray:
        """
        Compute the Type‑2 CDF C(u, v) by expanding to the full 3‑parameter form
        and swapping inputs so that t = -ln(u)/(-ln(u)-ln(v)) implements the
        mixed asymmetry on the (1‑t) margin.

        - If param length == 3: direct pass‑through.
        - Else: unpack [theta, beta], set psi1=1, psi2=beta, swap (u,v).
        """
        if len(param) == 3:
            return super().get_cdf(u, v, param)

        theta, beta = param
        full = np.array([theta, 1.0, beta])
        return super().get_cdf(v, u, full)


    def get_pdf(self, u: np.ndarray, v: np.ndarray, param: np.ndarray) -> np.ndarray:
        """
        Compute the Type‑2 density c(u, v).  Bypass if full parameters,
        otherwise expand [θ,1,β] and swap.
        """
        if len(param) == 3:
            return super().get_pdf(u, v, param)

        theta, beta = param
        full = np.array([theta, 1.0, beta])
        return super().get_pdf(v, u, full)


    def partial_derivative_C_wrt_u(
        self, u: np.ndarray, v: np.ndarray, param: np.ndarray
    ) -> np.ndarray:
        """
        ∂C/∂u wrapper for Type‑2:

        - Internal call (param length 3): direct base method.
        - External: unpack [θ,β], build full, and swap inputs so that
          ∂/∂u wrapper = ∂/∂v base at (v,u).
        """
        if len(param) == 3:
            return super().partial_derivative_C_wrt_u(u, v, param)

        theta, beta = param
        full = np.array([theta, 1.0, beta])
        return super().partial_derivative_C_wrt_v(v, u, full)


    def partial_derivative_C_wrt_v(
        self, u: np.ndarray, v: np.ndarray, param: np.ndarray
    ) -> np.ndarray:
        """
        ∂C/∂v wrapper for Type‑2:

        - Internal: pass through.
        - External: unpack, expand, swap → ∂/∂v wrapper = ∂/∂u base at (v,u).
        """
        if len(param) == 3:
            return super().partial_derivative_C_wrt_v(u, v, param)

        theta, beta = param
        full = np.array([theta, 1.0, beta])
        return super().partial_derivative_C_wrt_u(v, u, full)
