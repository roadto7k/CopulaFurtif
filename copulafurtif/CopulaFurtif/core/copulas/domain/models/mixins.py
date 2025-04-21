from CopulaFurtif.core.copulas.domain.estimation.gof import compute_aic, compute_bic, kendall_tau_distance

class SupportsTailDependence:
    def LTDC(self, param):
        raise NotImplementedError

    def UTDC(self, param):
        raise NotImplementedError
    


class ModelSelectionMixin:
    def AIC(self):
        if self.log_likelihood_ is None:
            raise RuntimeError("Copula must be fitted before computing AIC.")
        return compute_aic(self)

    def BIC(self):
        if self.log_likelihood_ is None or self.n_obs is None:
            raise RuntimeError("Missing log-likelihood or n_obs.")
        return compute_bic(self)

    def kendall_tau_error(self, data):
        return kendall_tau_distance(self, data)
