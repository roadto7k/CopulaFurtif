import numpy as np

from CopulaFurtif.core.copulas.domain.estimation.estimation import _cmle, _fit_mle, _fit_ifm, quick_fit
from CopulaFurtif.core.copulas.domain.estimation.estimation import _fit_tau_core


def _safe_write_back_result(copula, params, ll):
    """Write fitted parameters back once, with open-bound clipping if needed."""
    cur_len = len(copula.get_parameters())
    theta = np.asarray(params, dtype=float).ravel()[:cur_len]

    try:
        copula.set_parameters(theta.tolist())
    except ValueError:
        bounds = copula.get_bounds() if hasattr(copula, "get_bounds") else [(None, None)] * cur_len
        clipped = []
        for value, (low, high) in zip(theta, bounds):
            lo = -1e10 if low is None else float(low)
            hi = 1e10 if high is None else float(high)
            val = float(value)
            if val <= lo:
                val = lo + 1e-8
            if val >= hi:
                val = hi - 1e-8
            clipped.append(val)
        copula.set_parameters(clipped)

    if hasattr(copula, "set_log_likelihood"):
        copula.set_log_likelihood(float(ll))
    else:
        copula.log_likelihood_ = float(ll)


class CopulaFitter:
    def fit_cmle(
            self,
            data,
            copula,
            opti_method=None,
            options=None,
            use_init=True,
            quick=False,
            return_metrics=False,
            verbose=True,
            inputs_are_uniform=False,
    ):
        """
        Fit copula parameters by maximizing the copula log-likelihood.

        Parameters
        ----------
        data : sequence
            Pair of raw observations or already-uniform PIT values.

        copula : CopulaModel
            Copula model to fit.

        opti_method : str, optional
            Optimization method.

        options : dict, optional
            Optimizer options.

        use_init : bool, optional
            Use robust data-driven initialization.

        quick : bool, optional
            Use reduced optimizer iterations.

        return_metrics : bool, optional
            Return optional fitting diagnostics.

        verbose : bool, optional
            Enable fitting diagnostics.

        inputs_are_uniform : bool, optional
            False:
                Raw data are rank-transformed internally (CML).

            True:
                Inputs are already PIT/uniform values and are used directly
                (IFM copula stage).

        Returns
        -------
        tuple or None
            (params, loglik) or
            (params, loglik, extras).
        """
        res = _cmle(
            copula=copula,
            data=data,
            opti_method=opti_method,
            options=options,
            verbose=verbose,
            use_init=use_init,
            quick=quick,
            return_metrics=return_metrics,
            inputs_are_uniform=inputs_are_uniform,
        )

        if not res:
            return None

        if (
                isinstance(res, (list, tuple))
                and len(res) == 3
        ):
            params, ll, extras = res

        else:
            params, ll = res
            extras = None

        _safe_write_back_result(
            copula,
            params,
            ll,
        )

        if return_metrics:
            return params, ll, extras

        return params, ll

    def fit_mle(self, data, copula, marginals,
                opti_method=None, known_parameters=True,
                options=None, use_init=True, quick=False,
                return_metrics=False, verbose=True):
        """
        Full MLE wrapper (copula + optionally marginals).
        Passes robust init/quick flags and handles 2- or 3-tuples.
        """
        try:
            res = _fit_mle(data=data, copula=copula, marginals=marginals,
                           opti_method=opti_method, known_parameters=known_parameters,
                           options=options, verbose=verbose,
                           use_init=use_init, quick=quick, return_metrics=return_metrics)
        except TypeError:
            # fallback old signature
            res = _fit_mle(data, copula, marginals, known_parameters=known_parameters)

        if not res:
            return None

        if isinstance(res, (list, tuple)) and len(res) == 3:
            params, ll, extras = res
        else:
            params, ll = res
            extras = None

        _safe_write_back_result(copula, params, ll)

        return (params, ll, extras) if return_metrics else (params, ll)

    def fit_ifm(self, data, copula, marginals,
                opti_method=None, options=None,
                use_init=True, quick=False, return_metrics=False, verbose=True):
        """
        IFM wrapper: fit marginals, transform to (U,V), then MLE for copula.
        """
        try:
            res = _fit_ifm(data=data, copula=copula, marginals=marginals,
                           opti_method=opti_method, options=options,
                           verbose=verbose, use_init=use_init,
                           quick=quick, return_metrics=return_metrics)
        except TypeError:
            # fallback old signature
            res = _fit_ifm(data, copula, marginals)

        if not res:
            return None

        if isinstance(res, (list, tuple)) and len(res) == 3:
            params, ll, extras = res
        else:
            params, ll = res
            extras = None

        _safe_write_back_result(copula, params, ll)

        return (params, ll, extras) if return_metrics else (params, ll)

    def fit_quick(self, data, copula, mode="cmle", marginals=None, maxiter=60,
                  optimizer="L-BFGS-B", return_metrics=True):
        """
        Wrapper mince autour de estimation.quick_fit pour éviter les imports croisés.
        """
        res = quick_fit(data=data, copula=copula, mode=mode, marginals=marginals,
                        maxiter=maxiter, optimizer=optimizer, return_metrics=return_metrics)

        return res

    def fit_tau(self, data, copula):
        """
        Init-only fit that calls copula.init_from_data(u, v).
        No optimization. No CMLE/MLE/IFM.
        """
        return _fit_tau_core(data=data, copula=copula)
