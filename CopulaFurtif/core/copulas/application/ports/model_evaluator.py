from abc import ABC, abstractmethod


class ModelEvaluator(ABC):
    """
    Port pour effectuer une évaluation complète du modèle (AIC, BIC, AD, etc.)
    """

    @abstractmethod
    def evaluate(self, data, copulas, quick=True):
        pass
