from CopulaFurtif.copulas import CopulaType

class CopulaFactory:
    """
    Factory centralisée pour créer des instances de copules par nom.
    """
    registry = {}

    @classmethod
    def register(cls, name: CopulaType, constructor):
        cls.registry[name] = constructor

    @classmethod
    def create(cls, name: CopulaType):
        key = name
        if key not in cls.registry:
            raise ValueError(f"Unknown copula type: {name}")
        return cls.registry[key]()
    
    
from CopulaFurtif.core.copulas.infrastructure import copula_registry