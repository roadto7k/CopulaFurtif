class CopulaFactory:
    """
    Factory centralisée pour créer des instances de copules par nom.
    """
    registry = {}

    @classmethod
    def register(cls, name: str, constructor):
        cls.registry[name.lower()] = constructor

    @classmethod
    def create(cls, name: str):
        key = name.lower()
        if key not in cls.registry:
            raise ValueError(f"Unknown copula type: {name}")
        return cls.registry[key]()
    
    
from CopulaFurtif.core.copulas.infrastructure import copula_registry