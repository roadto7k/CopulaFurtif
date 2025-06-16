from CopulaFurtif.core.copulas.domain.estimation.estimation import cmle, fit_mle, fit_ifm

class CopulaFitter:
    def fit_cmle(self, data, copula):
        """
        Fit copula parameters via canonical maximum likelihood estimation.

        Args:
            data (array-like): Observed data for fitting.
            copula (CopulaModel): Copula instance to fit. Must allow assignment to `parameters` and `log_likelihood_`.

        Returns:
            tuple[list, float] or None: Fitted parameters and log-likelihood, or None if estimation failed.
        """

        res = cmle(copula, data)
        if res:
            copula.parameters, copula.log_likelihood_ = res
        return res

    def fit_mle(self, data, copula, marginals, known_parameters=True):
        """
        Fit copula parameters via full maximum likelihood estimation.

        Args:
            data (array-like): Observed data for fitting.
            copula (CopulaModel): Copula instance to fit.
            marginals (Sequence): Marginal models corresponding to the data.
            known_parameters (bool, optional): Whether some parameters are known. Defaults to True.

        Returns:
            tuple[list, float] or None: Fitted parameters and log-likelihood, or None if estimation failed.
        """

        res = fit_mle(data, copula, marginals, known_parameters=known_parameters)
        if res:
            copula.parameters, copula.log_likelihood_ = res[0][:len(copula.parameters)], res[1]
        return res

    def fit_ifm(self, data, copula, marginals):
        """Fit copula parameters via inference functions for margins (IFM) method.

        Args:
            data (array-like): Observed data for fitting.
            copula (CopulaModel): Copula instance to fit. Must allow assignment to `parameters` and `log_likelihood_`.
            marginals (Sequence): Marginal models corresponding to the data.

        Returns:
            tuple[list, float] or None: Fitted parameters and log-likelihood, or None if estimation failed.
        """

        res = fit_ifm(data, copula, marginals)
        if res:
            copula.parameters, copula.log_likelihood_ = res
        return res