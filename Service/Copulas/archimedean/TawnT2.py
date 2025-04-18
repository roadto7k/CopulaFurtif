import numpy as np

from Service.Copulas.archimedean.Tawn import TawnCopula


class TawnT2Copula(TawnCopula):
    """
    Tawn Type-2 (asymmetric mixed) extreme-value copula as a wrapper
    around generic TawnCopula. Accepts two parameters [theta, beta] via
    the .parameters setter (no args in __init__).

    Parameters
    ----------
    theta : float
        Dependence strength (>=1).
    beta : float
        Asymmetry on the (1-t) margin, in [0,1].

    Internally fixes psi2 = 1.0 and sets psi1 = beta, then swaps (u, v)
    so that t = -ln(u)/(-ln(u)-ln(v)) matches the native Type-2 argument x/(x+y).
    """
    def __init__(self):
        super().__init__()
        self.type = "tawn2"
        self.name = "Tawn Type-2 Copula"
        self.bounds_param = [(1.0, None), (0.0, 1.0)]
        # initialize default parameters [theta=2.0, beta=0.5]
        self.parameters = np.array([2.0, 0.5])

    @property
    def parameters(self) -> np.ndarray:
        """
        Return the two free parameters [theta, beta].
        """
        return np.array([self._parameters[0], self._psi1])

    @parameters.setter
    def parameters(self, param: np.ndarray):
        """
        Set the two free parameters [theta, beta].
        Updates internal psi1 and theta, keeps psi2=1.
        Handles None silently.
        """
        if param is None:
            return
        theta, beta = param
        self._psi1 = beta
        self._psi2 = 1.0
        self._parameters = np.array([theta, self._psi1, self._psi2])

    def get_cdf(self, u: np.ndarray, v: np.ndarray, param: np.ndarray) -> np.ndarray:
        """
        Compute CDF C(u,v) for Type-2 by expanding parameters and swapping inputs.

        - Expand [theta, beta] to [theta, psi1, psi2]
        - Swap (u, v) so that base class computes t = x/(x+y)
        """
        theta, beta = param
        full = np.array([theta, beta, 1.0])
        return super().get_cdf(v, u, full)

    def get_pdf(self, u: np.ndarray, v: np.ndarray, param: np.ndarray) -> np.ndarray:
        """
        Compute joint density c(u,v) for Type-2 by swapping inputs.
        """
        theta, beta = param
        full = np.array([theta, beta, 1.0])
        return super().get_pdf(v, u, full)

    def partial_derivative_C_wrt_u(
        self, u: np.ndarray, v: np.ndarray, param: np.ndarray
    ) -> np.ndarray:
        """
        Conditional CDF P(V ≤ v | U = u), via ∂C/∂u wrapper.
        Swaps inputs so wrapper ∂/∂u corresponds to base ∂/∂v.
        """
        theta, beta = param
        full = np.array([theta, beta, 1.0])
        return super().partial_derivative_C_wrt_v(v, u, full)

    def partial_derivative_C_wrt_v(
        self, u: np.ndarray, v: np.ndarray, param: np.ndarray
    ) -> np.ndarray:
        """
        Conditional CDF P(U ≤ u | V = v), via ∂C/∂v wrapper.
        Swaps inputs so wrapper ∂/∂v corresponds to base ∂/∂u.
        """
        theta, beta = param
        full = np.array([theta, beta, 1.0])
        return super().partial_derivative_C_wrt_u(v, u, full)