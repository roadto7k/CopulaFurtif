from CopulaFurtif.core.copulas.domain.factories.copula_factory import CopulaFactory
from CopulaFurtif.core.copulas.domain.models.elliptical.gaussian import GaussianCopula

CopulaFactory.register("gaussian", GaussianCopula)