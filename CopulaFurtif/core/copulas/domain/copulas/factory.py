from typing import Type, Dict
from domain.copulas.base import BaseCopula


class CopulaFactory:
    """
    Factory pour instancier dynamiquement des copules à partir d’un nom (chaîne de caractères).
    """

    _registry: Dict[str, Type[BaseCopula]] = {}

    @classmethod
    def register(cls, name: str, copula_cls: Type[BaseCopula]):
        """
        Enregistre une classe de copule sous un nom.
        """
        key = name.strip().lower()
        if key in cls._registry:
            raise ValueError(f"Copula '{name}' is already registered.")
        cls._registry[key] = copula_cls

    @classmethod
    def create(cls, name: str) -> BaseCopula:
        """
        Crée une instance de copule à partir de son nom.
        """
        key = name.strip().lower()
        if key not in cls._registry:
            raise ValueError(f"Copula '{name}' is not registered. Available: {list(cls._registry.keys())}")
        return cls._registry[key]()

    @classmethod
    def available(cls):
        """
        Liste les copules enregistrées.
        """
        return list(cls._registry.keys())
