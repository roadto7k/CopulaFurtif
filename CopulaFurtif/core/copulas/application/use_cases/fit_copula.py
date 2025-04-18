from application.ports.copula_fitter import CopulaFitter
from domain.estimation.cmle import cmle
from domain.estimation.mle import fit_mle
from domain.estimation.ifm import fit_ifm


class FitCopulaUseCase(CopulaFitter):
    """
    Implémentation concrète du port CopulaFitter
    """

    def fit_cmle(self, data, copula):
        result = cmle(copula, data)
        if result:
            copula.parameters = result[0]
            copula.log_likelihood_ = result[1]
        return result

    def fit_mle(self, data, copula, marginals, known_parameters=True):
        result = fit_mle(data, copula, marginals, known_parameters=known_parameters)
        if result:
            copula.parameters = result[0][:len(copula.parameters)]
            copula.log_likelihood_ = result[1]
        return result

    def fit_ifm(self, data, copula, marginals):
        result = fit_ifm(data, copula, marginals)
        if result:
            copula.parameters = result[0]
            copula.log_likelihood_ = result[1]
        return result
