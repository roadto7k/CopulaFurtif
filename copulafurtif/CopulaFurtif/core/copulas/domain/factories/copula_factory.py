from CopulaFurtif.core.copulas.domain.copula_type import CopulaType
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel

class CopulaFactory:
    """
    Centralized factory for creating copula instances by name.
    """
    registry = {}

    @classmethod
    def register(cls, name: CopulaType, constructor):
        """
        Register a copula constructor under a given type name.

        Args:
            name (CopulaType): Identifier for the copula type.
            constructor (Callable[[], CopulaModel]): Factory function that returns a new copula instance.

        Returns:
            None
        """

        cls.registry[name] = constructor

    @classmethod
    def create(cls, name: CopulaType) -> CopulaModel:
        """
        Create a copula instance by its registered type name.

        Args:
            name (CopulaType): Identifier of the copula to instantiate.

        Returns:
            CopulaModel: New instance of the requested copula.

        Raises:
            ValueError: If no constructor is registered for the given name.
        """

        key = name
        if key not in cls.registry:
            raise ValueError(f"Unknown copula type: {name}")
        return cls.registry[key]()
    
    
from CopulaFurtif.core.copulas.infrastructure import copula_registry