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

from CopulaFurtif.copulas import CopulaType

CopulaFactory.register(CopulaType.GAUSSIAN, GaussianCopula)
CopulaFactory.register(CopulaType.STUDENT, StudentCopula)

CopulaFactory.register(CopulaType.FRANK, FrankCopula)
CopulaFactory.register(CopulaType.JOE, JoeCopula)
CopulaFactory.register(CopulaType.AMH, AMHCopula)
CopulaFactory.register(CopulaType.CLAYTON, ClaytonCopula)
CopulaFactory.register(CopulaType.FGM, FGMCopula)
CopulaFactory.register(CopulaType.GALAMBOS, GalambosCopula)
CopulaFactory.register(CopulaType.GUMBEL, GumbelCopula)
CopulaFactory.register(CopulaType.PLACKETT, PlackettCopula)
CopulaFactory.register(CopulaType.TAWN, TawnCopula)