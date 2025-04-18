from abc import ABC, abstractmethod


class CopulaFitter(ABC):
    """
    Port d’entrée métier pour ajuster une copule à des données.
    """
    
    @abstractmethod
    def fit_cmle(self, data, copula):
        pass

    @abstractmethod
    def fit_mle(self, data, copula, marginals, known_parameters=True):
        pass

    @abstractmethod
    def fit_ifm(self, data, copula, marginals):
        pass
