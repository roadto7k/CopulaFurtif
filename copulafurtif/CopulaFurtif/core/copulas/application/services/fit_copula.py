# import numpy as np

# from CopulaFurtif.core.copulas.domain.estimation.estimation import _cmle, _fit_mle, _fit_ifm, quick_fit
# from CopulaFurtif.core.copulas.domain.estimation.estimation import _fit_tau_core



# class CopulaFitter:
#     def fit_cmle(self, data, copula,
#                  opti_method='SLSQP', options=None,
#                  use_init=True, quick=False, return_metrics=False, verbose=True):
#         """
#         CMLE wrapper with robust init/quick mode passthrough and safe result handling.
#         """
#         try:
#             res = _cmle(copula=copula, data=data,
#                         opti_method=opti_method, options=options,
#                         verbose=verbose, use_init=use_init,
#                         quick=quick, return_metrics=return_metrics)
#         except TypeError:
#             # fallback old signature
#             res = _cmle(copula, data)

#         if not res:
#             return None

#         # unpack (params, ll[, extras])
#         if isinstance(res, (list, tuple)) and len(res) == 3:
#             params, ll, extras = res
#         else:
#             params, ll = res
#             extras = None

#         # write back to model (crop to expected length)
#         cur_len = len(copula.get_parameters())
#         copula.set_parameters(np.asarray(params, dtype=float)[:cur_len])
#         if hasattr(copula, "set_log_likelihood"):
#             copula.set_log_likelihood(float(ll))
#         else:
#             copula.log_likelihood_ = float(ll)

#         return (params, ll, extras) if return_metrics else (params, ll)

#     def fit_mle(self, data, copula, marginals,
#                 opti_method='SLSQP', known_parameters=True,
#                 options=None, use_init=True, quick=False,
#                 return_metrics=False, verbose=True):
#         """
#         Full MLE wrapper (copula + optionally marginals).
#         Passes robust init/quick flags and handles 2- or 3-tuples.
#         """
#         try:
#             res = _fit_mle(data=data, copula=copula, marginals=marginals,
#                            opti_method=opti_method, known_parameters=known_parameters,
#                            options=options, verbose=verbose,
#                            use_init=use_init, quick=quick, return_metrics=return_metrics)
#         except TypeError:
#             # fallback old signature
#             res = _fit_mle(data, copula, marginals, known_parameters=known_parameters)

#         if not res:
#             return None

#         if isinstance(res, (list, tuple)) and len(res) == 3:
#             params, ll, extras = res
#         else:
#             params, ll = res
#             extras = None

#         cur_len = len(copula.get_parameters())
#         copula.set_parameters(np.asarray(params, dtype=float)[:cur_len])
#         if hasattr(copula, "set_log_likelihood"):
#             copula.set_log_likelihood(float(ll))
#         else:
#             copula.log_likelihood_ = float(ll)

#         return (params, ll, extras) if return_metrics else (params, ll)

#     def fit_ifm(self, data, copula, marginals,
#                 opti_method='SLSQP', options=None,
#                 use_init=True, quick=False, return_metrics=False, verbose=True):
#         """
#         IFM wrapper: fit marginals, transform to (U,V), then MLE for copula.
#         """
#         try:
#             res = _fit_ifm(data=data, copula=copula, marginals=marginals,
#                            opti_method=opti_method, options=options,
#                            verbose=verbose, use_init=use_init,
#                            quick=quick, return_metrics=return_metrics)
#         except TypeError:
#             # fallback old signature
#             res = _fit_ifm(data, copula, marginals)

#         if not res:
#             return None

#         if isinstance(res, (list, tuple)) and len(res) == 3:
#             params, ll, extras = res
#         else:
#             params, ll = res
#             extras = None

#         cur_len = len(copula.get_parameters())
#         copula.set_parameters(np.asarray(params, dtype=float)[:cur_len])
#         if hasattr(copula, "set_log_likelihood"):
#             copula.set_log_likelihood(float(ll))
#         else:
#             copula.log_likelihood_ = float(ll)

#         return (params, ll, extras) if return_metrics else (params, ll)

#     def fit_quick(self, data, copula, mode="cmle", marginals=None, maxiter=60,
#                   optimizer="L-BFGS-B", return_metrics=True):
#         """
#         Wrapper mince autour de estimation.quick_fit pour éviter les imports croisés.
#         """
#         res = quick_fit(data=data, copula=copula, mode=mode, marginals=marginals,
#                         maxiter=maxiter, optimizer=optimizer, return_metrics=return_metrics)

#         return res

#     def fit_tau(self, data, copula):
#         """
#         Init-only fit that calls copula.init_from_data(u, v).
#         No optimization. No CMLE/MLE/IFM.
#         """
#         return _fit_tau_core(data=data, copula=copula)

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
    def fit_cmle(self, data, copula,
                 opti_method='SLSQP', options=None,
                 use_init=True, quick=False, return_metrics=False, verbose=True):
        """
        CMLE wrapper with robust init/quick mode passthrough and safe result handling.
        """
        try:
            res = _cmle(copula=copula, data=data,
                        opti_method=opti_method, options=options,
                        verbose=verbose, use_init=use_init,
                        quick=quick, return_metrics=return_metrics)
        except TypeError:
            # fallback old signature
            res = _cmle(copula, data)

        if not res:
            return None

        # unpack (params, ll[, extras])
        if isinstance(res, (list, tuple)) and len(res) == 3:
            params, ll, extras = res
        else:
            params, ll = res
            extras = None

        _safe_write_back_result(copula, params, ll)

        return (params, ll, extras) if return_metrics else (params, ll)

    def fit_mle(self, data, copula, marginals,
                opti_method='SLSQP', known_parameters=True,
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
                opti_method='SLSQP', options=None,
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
