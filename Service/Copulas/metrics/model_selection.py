import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from Service.Copulas.fitting.estimation import pseudo_obs


def compute_aic(copula):
    """
    Compute the Akaike Information Criterion (AIC) for a fitted copula.

    Parameters
    ----------
    copula : object
        The copula object with attributes:
        - log_likelihood_ : float
        - bounds_param : list of parameter bounds

    Returns
    -------
    float
        AIC score
    """
    if copula.log_likelihood_ is None:
        raise ValueError("The copula must be fitted before computing AIC.")
    k = len(copula.bounds_param)
    return 2 * k - 2 * copula.log_likelihood_


def compute_bic(copula):
    """
    Compute the Bayesian Information Criterion (BIC) for a fitted copula.

    Returns
    -------
    float
        BIC score
    """
    if copula.log_likelihood_ is None:
        raise ValueError("The copula must be fitted before computing BIC.")
    if not hasattr(copula, "n_obs") or copula.n_obs is None:
        raise ValueError("n_obs (number of observations) is missing in copula object.")

    k = len(copula.bounds_param)
    return k * np.log(copula.n_obs) - 2 * copula.log_likelihood_

def compute_iad_score(copula, data, params=None, fitted=False):
    """
    Compute the Integrated Anderson-Darling (IAD) statistic between the empirical
    copula and a parametric copula model.

    Parameters
    ----------
    copula : object
        A copula object with a `.get_cdf(u, v, params)` method, and optionally `copula.parameters`.
    data : array-like
        List or array [u, v], each of length n, with pseudo-observations ∈ (0, 1).
    params : array-like, optional
        Parameters to use. If `fitted=True`, this is ignored and `copula.parameters` is used.
    fitted : bool
        If True, use copula.parameters directly (assumes the copula has already been fitted).

    Returns
    -------
    float
        The IAD goodness-of-fit score. Lower values indicate a better fit.
    """
    u, v = data
    n = len(u)

    if len(v) != n:
        raise ValueError("Mismatch: len(u) != len(v)")

    if fitted:
        if not hasattr(copula, "parameters") or copula.parameters is None:
            raise ValueError("Copula must have a valid 'parameters' attribute if fitted=True.")
        params = copula.parameters

    if params is None:
        raise ValueError("You must provide 'params', or set fitted=True to use copula.parameters.")

    # Sort pseudo-observations
    u_sorted = np.sort(u)
    v_sorted = np.sort(v)

    # Build empirical copula grid
    U, V = np.meshgrid(u_sorted, v_sorted, indexing='ij')
    below_u = u[:, None, None] <= U
    below_v = v[:, None, None] <= V
    C_empirical = np.sum(below_u & below_v, axis=0) / n

    # Build parametric copula grid
    step = np.linspace(1 / n, 1 - 1 / n, n)
    uu, vv = np.meshgrid(step, step, indexing='ij')
    u_flat = uu.ravel()
    v_flat = vv.ravel()

    C_model = copula.get_cdf(u_flat, v_flat, params).reshape(n, n)

    # Anderson-Darling denominator
    eps = 1e-10
    denom = np.clip(C_model * (1 - C_model), eps, None)

    iad_score = np.sum(((C_empirical - C_model) ** 2) / denom)

    return iad_score

def AD_score(copula, data, fitted=False, param=None):
    """
    Compute the Anderson-Darling (AD) goodness-of-fit statistic between
    the empirical copula and the parametric copula.

    The AD score is the maximum weighted squared deviation between
    the empirical and parametric copula CDFs, with weights emphasizing
    tail behavior.

    Parameters
    ----------
    copula : object
        Copula object with method `.get_cdf(u, v, theta)` and attribute `.parameters`.
    data : array-like
        A list or array of shape (2, n) or (n, 2) with observations.
    fitted : bool
        If True, uses copula.parameters as the fitted parameter.
        If False, `param` must be provided.
    param : list or array, optional
        Copula parameter(s) to use if `fitted=False`.

    Returns
    -------
    float
        The Anderson-Darling (AD) score.
    """

    if fitted:
        if not hasattr(copula, "parameters") or copula.parameters is None:
            raise ValueError("copula.parameters is not set. Either fit the copula or pass param explicitly.")
        param = copula.parameters
    elif param is None:
        raise ValueError("param must be provided if fitted=False.")

    u, v = data
    n = len(u)

    # Sort data
    u_sorted = np.sort(u)
    v_sorted = np.sort(v)

    # Build empirical grid
    u_grid, v_grid = np.meshgrid(u_sorted, v_sorted, indexing="ij")

    # Empirical copula
    counts = np.sum((u[:, None, None] <= u_grid) & (v[:, None, None] <= v_grid), axis=0)
    C_empirical = counts / n

    # Theoretical copula
    grid_u_flat = u_grid.flatten()
    grid_v_flat = v_grid.flatten()
    C_model = copula.get_cdf(grid_u_flat, grid_v_flat, param).reshape((n, n))

    # Avoid division by zero (clip model CDF to avoid C*(1-C)=0)
    eps = 1e-10
    C_model = np.clip(C_model, eps, 1 - eps)

    # Anderson-Darling score: max of the weighted squared diff
    weights = C_model * (1 - C_model)
    score = np.max(((C_empirical - C_model) ** 2) / weights)

    return score

def kendall_tau_distance(copula, data, fitted=False):
    """
    Compute the absolute difference between the empirical Kendall's tau
    and the theoretical Kendall's tau implied by the copula.

    Parameters
    ----------
    copula : object
        Copula object. Must optionally have a method `.kendall_tau(param)`
    data : list of arrays
        [X, Y] sample
    fitted : bool
        If True, use `copula.parameters` as the fitted param.
        Otherwise, raise an error.

    Returns
    -------
    float
        Absolute distance between empirical and theoretical Kendall's tau,
        or np.nan if the copula does not implement `kendall_tau`.
    """

    if not fitted:
        raise ValueError("Set `fitted=True` to use fitted copula parameters.")

    if not hasattr(copula, "kendall_tau"):
        print(f"[WARNING] Kendall's tau formula not implemented for copula '{copula.name}'. Returning np.nan.")
        return np.nan

    # Empirical tau from data
    X, Y = data
    tau_empirical, _ = kendalltau(X, Y)

    try:
        tau_theoretical = copula.kendall_tau(copula.parameters)
    except Exception as e:
        print(f"[WARNING] Failed to compute theoretical Kendall's tau: {e}")
        return np.nan

    return abs(tau_empirical - tau_theoretical)


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



def copula_diagnostics(data, copulas, verbose=True):
    """
    Perform a comprehensive diagnostic evaluation of one or several fitted copulas.
    Computes model selection criteria and goodness-of-fit metrics.

    Parameters
    ----------
    data : list of arrays
        [X, Y] observations
    copulas : list
        List of fitted copula objects (must have .parameters and .n_obs set).
        Can also be a single copula (not in a list).
    verbose : bool
        If True, print informative logs to guide interpretation.

    Returns
    -------
    pd.DataFrame
        Summary table with AIC, BIC, IAD, AD, Kendall's tau error, etc.
    """
    # Ensure list
    if not isinstance(copulas, list):
        copulas = [copulas]

    from scipy.stats import kendalltau

    results = []

    for cop in copulas:
        if verbose:
            print(f"\nEvaluating: {cop.name}")

        # Extract required data
        try:
            loglik = cop.log_likelihood
            n_param = len(cop.bounds_param)
            n_obs = cop.n_obs
        except Exception as e:
            print(f"[ERROR] Missing attributes in copula '{cop.name}': {e}")
            continue

        # Model selection criteria
        aic = -2 * loglik + 2 * n_param
        bic = -2 * loglik + n_param * np.log(n_obs)

        # IAD
        try:
            iad = cop.IAD(data, fitted=True)
        except:
            iad = np.nan

        # AD
        try:
            ad = cop.AD(data, fitted=True)
        except:
            ad = np.nan

        # Kendall's tau
        try:
            tau_emp, _ = kendalltau(data[0], data[1])
            tau_model = cop.kendall_tau(cop.parameters) if hasattr(cop, 'kendall_tau') else np.nan
            tau_error = abs(tau_model - tau_emp) if not np.isnan(tau_model) else np.nan
        except:
            tau_model, tau_error = np.nan, np.nan

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

        if verbose:
            print(f"  Log-Likelihood: {loglik:.4f}")
            print(f"  AIC: {aic:.4f} | BIC: {bic:.4f}")
            print(f"  IAD Distance: {iad:.6f} | AD Distance: {ad:.6f}")
            print(f"  Kendall's Tau Error: {tau_error:.6f}")

    df = pd.DataFrame(results)
    df = df.sort_values(by="AIC").reset_index(drop=True)

    return df

def interpret_copula_results(df):
    """
    Interpret the copula comparison results from the summary DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output from copula_diagnostics(), expected to contain columns like:
        ['Copula', 'logLik', 'AIC', 'BIC', 'IAD', 'AD', 'Kendall_tau_error']

    Returns
    -------
    str
        Interpretation message with guidance on copula selection.
    """
    if len(df) == 0:
        return "No copula was evaluated."

    if len(df) == 1:
        row = df.iloc[0]
        msg = f"Only one copula ({row['Copula']}) was evaluated.\n"
        msg += f"- AIC: {row['AIC']:.3f}, BIC: {row['BIC']:.3f}\n"
        msg += f"- IAD: {row['IAD']:.4e}, AD: {row['AD']:.4e}\n"
        if not pd.isna(row.get("Kendall_tau_error", None)):
            msg += f"- Kendall's tau error: {row['Kendall_tau_error']:.4f}\n"

        msg += "\nUse this as a reference for further comparison, but standalone fit quality should be assessed against domain knowledge."
        return msg

    best_aic = df.sort_values(by='AIC').iloc[0]
    best_bic = df.sort_values(by='BIC').iloc[0]
    best_iad = df.sort_values(by='IAD').iloc[0]
    best_ad = df.sort_values(by='AD').iloc[0]
    best_tau = df.sort_values(by='Kendall_tau_error').iloc[0] if 'Kendall_tau_error' in df.columns else None

    msg = f"Among the {len(df)} evaluated copulas:\n"
    msg += f"- Best AIC: {best_aic['Copula']} (AIC = {best_aic['AIC']:.2f})\n"
    msg += f"- Best BIC: {best_bic['Copula']} (BIC = {best_bic['BIC']:.2f})\n"
    msg += f"- Best IAD (fit to empirical copula): {best_iad['Copula']} (IAD = {best_iad['IAD']:.2e})\n"
    msg += f"- Best AD (tail-weighted deviation): {best_ad['Copula']} (AD = {best_ad['AD']:.2e})\n"
    if best_tau is not None:
        msg += f"- Best Kendall's tau match: {best_tau['Copula']} (error = {best_tau['Kendall_tau_error']:.4f})\n"

    msg += "\nInterpretation Tips:\n"
    msg += "- AIC/BIC balance fit quality and complexity → Lower is better.\n"
    msg += "- IAD focuses on global empirical fit, AD is more sensitive to tails.\n"
    msg += "- Kendall's tau error shows deviation from empirical concordance.\n"
    msg += "- Prefer a copula performing well across multiple criteria rather than optimizing only one.\n"

    return msg


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
        param = copula.parameters
        ltdc = copula.LTDC(param)
        utdc = copula.UTDC(param)
        error = np.sqrt((ltdc - lambda_L_emp) ** 2 + (utdc - lambda_U_emp) ** 2)
        rows.append({
            "Copula": copula.name,
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






