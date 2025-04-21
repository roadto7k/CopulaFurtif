from CopulaFurtif.core.copulas.domain.estimation.estimation import pseudo_obs
import numpy as np

class DiagnosticsService:
    def evaluate(self, data, copula):
        u, v = pseudo_obs(data)
        cdf = [u, v]
        result = {
            "Copula": copula.name,
            "LogLik": copula.log_likelihood_,
            "Params": len(copula.parameters),
            "Obs": copula.n_obs,
            "AIC": copula.AIC() if hasattr(copula, 'AIC') else np.nan,
            "BIC": copula.BIC() if hasattr(copula, 'BIC') else np.nan,
            "Kendall Tau Error": copula.kendall_tau_error(data) if hasattr(copula, 'kendall_tau_error') else np.nan
        }
        return result