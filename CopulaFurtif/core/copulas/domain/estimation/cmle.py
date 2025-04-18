from scipy.optimize import minimize
from domain.estimation.utils import pseudo_obs
import numpy as np


def cmle(copula, data, opti_method='SLSQP', options=None, verbose=True):
    if options is None:
        options = {}

    try:
        u, v = pseudo_obs(data)
        copula.n_obs = len(u)
    except Exception as e:
        print("[CMLE ERROR] Failed to compute pseudo-observations:", e)
        return None

    x0 = np.array(copula.parameters, dtype=float)
    bounds = copula.bounds_param if hasattr(copula, "bounds_param") else [(None, None)] * len(x0)
    bounds = [(low if low is not None else -1e10, high if high is not None else 1e10)
              for (low, high) in bounds]

    def log_likelihood(params):
        try:
            pdf_vals = copula.get_pdf(u, v, params)
            if np.any(pdf_vals <= 0):
                return np.inf
            return -np.sum(np.log(pdf_vals))
        except Exception:
            return np.inf

    try:
        result = minimize(log_likelihood, x0, method=opti_method, bounds=bounds, options=options)
    except Exception as e:
        print("[CMLE ERROR] Optimizer crashed:", e)
        return None

    if result.success:
        copula.parameters = result.x
        copula.log_likelihood_ = -result.fun
        return result.x, -result.fun
    else:
        if verbose:
            print("Optimization failed:", result.message)
        return None
