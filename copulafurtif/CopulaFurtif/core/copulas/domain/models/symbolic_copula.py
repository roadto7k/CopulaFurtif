import importlib
from sympy.utilities.lambdify import lambdify
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel, CopulaParameters

class SymbolicCopula(CopulaModel):
    def __init__(self, parameters: CopulaParameters):
        super().__init__()
        self._parameters = parameters

    def kendall_tau(self):
        raise NotImplementedError("Must implement kendall_tau specifically.")

    def sample(self, n):
        raise NotImplementedError("Must implement sampling method specifically.")
