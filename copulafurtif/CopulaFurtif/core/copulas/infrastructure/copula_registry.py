from CopulaFurtif.core.copulas.domain.factories.copula_factory import CopulaFactory

from CopulaFurtif.core.copulas.domain.models.elliptical.gaussian import GaussianCopula
from CopulaFurtif.core.copulas.domain.models.elliptical.student import StudentCopula

from CopulaFurtif.core.copulas.domain.models.archimedean.frank import FrankCopula
from CopulaFurtif.core.copulas.domain.models.archimedean.joe import JoeCopula
from CopulaFurtif.core.copulas.domain.models.archimedean.AMH import AMHCopula
from CopulaFurtif.core.copulas.domain.models.archimedean.clayton import ClaytonCopula
from CopulaFurtif.core.copulas.domain.models.archimedean.fgm import FGMCopula
from CopulaFurtif.core.copulas.domain.models.archimedean.galambos import GalambosCopula
from CopulaFurtif.core.copulas.domain.models.archimedean.gumbel import GumbelCopula
from CopulaFurtif.core.copulas.domain.models.archimedean.plackett import PlackettCopula
from CopulaFurtif.core.copulas.domain.models.archimedean.Tawn import TawnCopula

CopulaFactory.register("gaussian", GaussianCopula)
CopulaFactory.register("student", StudentCopula)

CopulaFactory.register("frank", FrankCopula)
CopulaFactory.register("joe", JoeCopula)
CopulaFactory.register("amh", AMHCopula)
CopulaFactory.register("clayton", ClaytonCopula)
CopulaFactory.register("fgm", FGMCopula)
CopulaFactory.register("galambos", GalambosCopula)
CopulaFactory.register("gumbel", GumbelCopula)
CopulaFactory.register("plackett", PlackettCopula)
CopulaFactory.register("tawn", TawnCopula)