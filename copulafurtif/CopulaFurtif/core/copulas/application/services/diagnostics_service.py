from CopulaFurtif.core.copulas.domain.estimation.estimation import pseudo_obs
import numpy as np
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel

class CopulaDiagnostics:
    def evaluate(self, data, copula : CopulaModel):
        """Evaluate diagnostics for a given copula model using pseudo-observations.

        Args:
            data (array-like): Observed data used to compute pseudo-observations.
            copula (CopulaModel): Fitted copula model with attributes:
                name (str),
                log_likelihood_ (float),
                parameters (Sequence),
                n_obs (int),
                and optional methods AIC(), BIC(), kendall_tau_error().

        Returns:
            dict[str, Any]: Diagnostic metrics:
                Copula (str): Copula name.
                LogLik (float): Log-likelihood.
                Params (int): Number of parameters.
                Obs (int): Number of observations.
                AIC (float): Akaike Information Criterion or NaN.
                BIC (float): Bayesian Information Criterion or NaN.
                Kendall Tau Error (float): Kendall tau error or NaN.
        """
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