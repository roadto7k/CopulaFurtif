import pandas as pd
from application.ports.model_evaluator import ModelEvaluator
from domain.metrics.model_selection import copula_diagnostics


class EvaluateModelDiagnosticsUseCase(ModelEvaluator):
    """
    Implémente les diagnostics complets (logLik, AIC, BIC, AD, IAD…).
    """

    def evaluate(self, data, copulas, quick=True):
        return pd.DataFrame(copula_diagnostics(data, copulas, quick=quick))
