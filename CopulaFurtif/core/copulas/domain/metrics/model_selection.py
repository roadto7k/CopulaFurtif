from domain.estimation.utils import pseudo_obs


def copula_diagnostics(data, copulas, quick=True):
    u, v = pseudo_obs(data)
    cdf = [u, v]
    results = []

    for cop in copulas:
        loglik = cop.log_likelihood_
        n_param = len(cop.bounds_param)
        n_obs = cop.n_obs

        aic = cop.AIC()
        bic = cop.BIC()

        iad = np.nan
        ad = np.nan
        tau_error = np.nan

        if not quick:
            try:
                iad = cop.IAD(cdf)
            except:
                pass
            try:
                ad = cop.AD(cdf)
            except:
                pass

        try:
            tau_error = cop.kendall_tau_error(data)
        except:
            pass

        results.append({
            "Copula": cop.name,
            "Family": cop.type,
            "LogLik": loglik,
            "Params": n_param,
            "Obs": n_obs,
            "AIC": aic,
            "BIC": bic,
            "IAD": iad,
            "AD": ad,
            "Kendall Tau Error": tau_error
        })

    return results
