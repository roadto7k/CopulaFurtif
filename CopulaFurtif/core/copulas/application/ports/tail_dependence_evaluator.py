from abc import ABC, abstractmethod

class TailDependenceEvaluator(ABC):
    @abstractmethod
    def compare(self, data, copulas, q_low=0.05, q_high=0.95):
        pass
