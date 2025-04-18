from domain.copulas.factory import CopulaFactory
from domain.copulas.elliptical.gaussian import GaussianCopula
from domain.copulas.elliptical.student import StudentCopula
from domain.copulas.archimedean.gumbel import GumbelCopula
from domain.copulas.archimedean.BB1 import BB1Copula

def register_all_copulas():
    CopulaFactory.register("gaussian", GaussianCopula)
    CopulaFactory.register("student", StudentCopula)
    CopulaFactory.register("gumbel", GumbelCopula)
    CopulaFactory.register("bb1", BB1Copula)
    # TODO: Ajouter les autres
