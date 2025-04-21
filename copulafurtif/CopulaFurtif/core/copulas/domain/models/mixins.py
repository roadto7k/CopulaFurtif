from CopulaFurtif.core.copulas.domain.estimation.gof import compute_aic, compute_bic, kendall_tau_distance
import numpy as np

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


class AdvancedCopulaFeatures:
    def partial_derivative_C_wrt_u(self, u, v, param=None):
        raise NotImplementedError

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        raise NotImplementedError

    def conditional_cdf_u_given_v(self, u, v, param=None):
        return self.partial_derivative_C_wrt_v(u, v, param)

    def conditional_cdf_v_given_u(self, u, v, param=None):
        return self.partial_derivative_C_wrt_u(u, v, param)

    def IAD(self, data):
        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan