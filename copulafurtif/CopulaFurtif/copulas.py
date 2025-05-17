from enum import Enum

class CopulaType(Enum):
    GAUSSIAN = "gaussian"
    STUDENT  = "student"
    FRANK = "frank"
    JOE = "joe"
    AMH = "amh"
    CLAYTON = "clayton"
    FGM = "fgm"
    GALAMBOS = "galambos"
    GUMBEL = "gumbel"
    PLACKETT = "plackett"
    TAWN = "tawn"
    