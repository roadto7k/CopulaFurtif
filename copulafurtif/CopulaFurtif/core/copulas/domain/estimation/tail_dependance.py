from __future__ import annotations

import numpy as np
import pandas as pd

from CopulaFurtif.core.copulas.domain.estimation.estimation import huang_lambda, pseudo_obs


def empirical_lambda_L(u, v, q=0.05):
    """
    Empirical estimate of the lower tail dependence coefficient λ_L.

    Parameters
    ----------
    u : array-like, shape (n,)
        First marginal (must be uniform on [0,1]).
    v : array-like, shape (n,)
        Second marginal (must be uniform on [0,1]).
    q : float, optional (default=0.05)
        Quantile threshold (e.g. 0.05 means bottom 5%).

    Returns
    -------
    float
        Empirical lower tail dependence coefficient λ_L.
    """
    u = np.asarray(u)
    v = np.asarray(v)
    joint = np.sum((u <= q) & (v <= q))
    return joint / np.sum(u <= q) if np.sum(u <= q) > 0 else 0.0

def empirical_lambda_U(u, v, q=0.95):
    """
    Empirical estimate of the upper tail dependence coefficient λ_U.

    Parameters
    ----------
    u : array-like, shape (n,)
        First marginal (must be uniform on [0,1]).
    v : array-like, shape (n,)
        Second marginal (must be uniform on [0,1]).
    q : float, optional (default=0.95)
        Quantile threshold (e.g. 0.95 means top 5%).

    Returns
    -------
    float
        Empirical upper tail dependence coefficient λ_U.
    """
    u = np.asarray(u)
    v = np.asarray(v)
    joint = np.sum((u > q) & (v > q))
    return joint / np.sum(u > q) if np.sum(u > q) > 0 else 0.0

def compare_tail_dependence(
    data,
    copula_list,
    *,
    method: str = "huang",
    q_low: float = 0.05,
    q_high: float = 0.95,
    k: int | None = None,
    verbose: bool = True,
):
    """
    Compare empirical tail dependence with model tail dependence.

    Parameters
    ----------
    data : (X, Y)
        Raw samples.
    copula_list : list[CopulaModel]
        Each copula must support:
          - get_parameters()
          - LTDC(params)  # lower tail dependence coefficient
          - UTDC(params)  # upper tail dependence coefficient
          - get_name() or .name
    method : {"huang","quantile"}
        "huang" uses Huang(1992) (recommended). "quantile" uses fixed thresholds q_low/q_high.
    q_low, q_high : float
        Only used if method="quantile".
    k : int or None
        Only used for method="huang". Defaults to floor(sqrt(n)).
    verbose : bool
        If True, print the best tail fit.

    Returns
    -------
    pd.DataFrame
        Sorted by Tail Error (Euclidean error in (λ_L, λ_U) space).
    """
    # pseudo-obs using your existing helper
    u, v = pseudo_obs(data)

    # empirical tails
    if method.lower() == "huang":
        lambda_L_emp = huang_lambda(u, v, side="lower", k=k)
        lambda_U_emp = huang_lambda(u, v, side="upper", k=k)
    else:
        # fallback to fixed-quantile estimators if you still have them
        # replace these two with your actual functions if names differ
        lambda_L_emp = empirical_lambda_L(u, v, q=q_low)
        lambda_U_emp = empirical_lambda_U(u, v, q=q_high)

    rows = []
    for copula in copula_list:
        params = copula.get_parameters()
        # model TD (theoretical) from the copula
        try:
            lambda_L_model = float(copula.LTDC(params))
        except Exception:
            lambda_L_model = np.nan
        try:
            lambda_U_model = float(copula.UTDC(params))
        except Exception:
            lambda_U_model = np.nan

        # Euclidean error in tail space (handle NaNs safely)
        diff_L = (lambda_L_model - lambda_L_emp) if np.isfinite(lambda_L_model) else np.nan
        diff_U = (lambda_U_model - lambda_U_emp) if np.isfinite(lambda_U_model) else np.nan
        tail_error = np.sqrt(diff_L**2 + diff_U**2) if np.isfinite(diff_L) and np.isfinite(diff_U) else np.inf

        # name accessor: prefer get_name(), fallback to .name, else class name
        name = getattr(copula, "get_name", None)
        if callable(name):
            name = copula.get_name()
        else:
            name = getattr(copula, "name", copula.__class__.__name__)

        rows.append({
            "Copula": name,
            "Empirical λ_L": lambda_L_emp,
            "Empirical λ_U": lambda_U_emp,
            "Model λ_L": lambda_L_model,
            "Model λ_U": lambda_U_model,
            "Tail Error": tail_error,
        })

    result_df = pd.DataFrame(rows).sort_values("Tail Error", kind="mergesort").reset_index(drop=True)
    if verbose and len(result_df) > 0 and np.isfinite(result_df.loc[0, "Tail Error"]):
        print(f"The best fitting copula in the tails is: {result_df.loc[0, 'Copula']}")

    return result_df

