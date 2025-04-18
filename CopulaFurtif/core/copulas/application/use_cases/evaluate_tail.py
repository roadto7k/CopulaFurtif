import pandas as pd
from application.ports.tail_dependence_evaluator import TailDependenceEvaluator
from domain.metrics.tail_dependence import compare_tail_dependence


class EvaluateTailDependenceUseCase(TailDependenceEvaluator):
    """
    Implémentation concrète de la comparaison de dépendance en queue.
    """

    def compare(self, data, copulas, q_low=0.05, q_high=0.95):
        return compare_tail_dependence(data, copulas, q_low, q_high, verbose=False)
