from CopulaFurtif.core.copulas.domain.estimation.estimation import cmle, fit_mle, fit_ifm

class CopulaFitter:
    def fit_cmle(self, data, copula):
        res = cmle(copula, data)
        if res:
            copula.parameters, copula.log_likelihood_ = res
        return res

    def fit_mle(self, data, copula, marginals, known_parameters=True):
        res = fit_mle(data, copula, marginals, known_parameters=known_parameters)
        if res:
            copula.parameters, copula.log_likelihood_ = res[0][:len(copula.parameters)], res[1]
        return res

    def fit_ifm(self, data, copula, marginals):
        res = fit_ifm(data, copula, marginals)
        if res:
            copula.parameters, copula.log_likelihood_ = res
        return res