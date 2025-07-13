import numpy as np
import pandas as pd

from CopulaFurtif.copula_utils import pseudo_obs

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

def compare_tail_dependence(data, copula_list, q_low=0.05, q_high=0.95, verbose=True):
    """
    Compare empirical tail dependence with theoretical tail dependence using the existing pseudo_obs function.

    Parameters
    ----------
    data : list or tuple of two arrays
        Raw data samples [X, Y].
    copula_list : list
        List of copula objects with attributes:
            - .name (string)
            - .parameters (array)
            - .LTDC(param) and .UTDC(param) methods.
    q_low : float, optional
        Quantile for estimating lower tail dependence (default 0.05).
    q_high : float, optional
        Quantile for estimating upper tail dependence (default 0.95).
    verbose : bool, optional
        If True, returns a summary message along with the DataFrame.

    Returns
    -------
    result_df : pd.DataFrame
        DataFrame containing empirical and theoretical tail dependence estimates for each copula.
    summary : str (if verbose is True)
        A summary message indicating the best fitting copula based on tail dependence.
    """

    u, v = pseudo_obs(data)
    lambda_L_emp = empirical_lambda_L(u, v, q=q_low)
    lambda_U_emp = empirical_lambda_U(u, v, q=q_high)

    rows = []
    for copula in copula_list:
        param = copula.get_parameters()
        ltdc = copula.LTDC(param)
        utdc = copula.UTDC(param)
        error = np.sqrt((ltdc - lambda_L_emp) ** 2 + (utdc - lambda_U_emp) ** 2)
        rows.append({
            "Copula": copula.get_name(),
            "Empirical λ_L": lambda_L_emp,
            "Empirical λ_U": lambda_U_emp,
            "Model λ_L": ltdc,
            "Model λ_U": utdc,
            "Tail Error": error
        })

    result_df = pd.DataFrame(rows).sort_values("Tail Error")
    best_fit = result_df.iloc[0]["Copula"]

    if verbose:
        print(f"The best fitting copula in the tails is: {best_fit}")

    return result_df
