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
from CopulaFurtif.core.copulas.domain.models.archimedean.TawnT1 import TawnT1Copula
from CopulaFurtif.core.copulas.domain.models.archimedean.TawnT2 import TawnT2Copula
from CopulaFurtif.core.copulas.domain.models.archimedean.BB1 import BB1Copula
from CopulaFurtif.core.copulas.domain.models.archimedean.BB2 import BB2Copula
from CopulaFurtif.core.copulas.domain.models.archimedean.BB3 import BB3Copula
from CopulaFurtif.core.copulas.domain.models.archimedean.BB4 import BB4Copula
from CopulaFurtif.core.copulas.domain.models.archimedean.BB5 import BB5Copula
from CopulaFurtif.core.copulas.domain.models.archimedean.BB6 import BB6Copula
from CopulaFurtif.core.copulas.domain.models.archimedean.BB7 import BB7Copula
from CopulaFurtif.core.copulas.domain.models.archimedean.BB8 import BB8Copula

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
CopulaFactory.register(CopulaType.TAWNT1, TawnT1Copula)
CopulaFactory.register(CopulaType.TAWNT2, TawnT1Copula)
CopulaFactory.register(CopulaType.BB1, BB1Copula)
CopulaFactory.register(CopulaType.BB2, BB2Copula)
CopulaFactory.register(CopulaType.BB3, BB3Copula)
CopulaFactory.register(CopulaType.BB4, BB4Copula)
CopulaFactory.register(CopulaType.BB5, BB5Copula)
CopulaFactory.register(CopulaType.BB6, BB6Copula)
CopulaFactory.register(CopulaType.BB7, BB7Copula)
CopulaFactory.register(CopulaType.BB8, BB8Copula)