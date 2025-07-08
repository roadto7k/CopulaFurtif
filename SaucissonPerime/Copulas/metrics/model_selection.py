import numpy as np
import pandas as pd

from SaucissonPerime.Copulas.fitting.estimation import pseudo_obs


def copula_diagnostics(data, copulas, verbose=True, quick=True):
    """
    Perform a comprehensive diagnostic evaluation of one or several fitted copulas.
    Computes model selection criteria and goodness-of-fit metrics.

    Parameters
    ----------
    data : list of arrays
        [X, Y] observations (raw data; pseudo-observations will be computed internally).
    copulas : list
        List of fitted copula objects (must have .parameters and .n_obs set).
        Can also be a single copula (not in a list).
    verbose : bool
        If True, print informative logs to guide interpretation.
    quick : bool, optional (default=False)
        If True, skip computing IAD and AD scores (set them to np.nan) for a faster evaluation.

    Returns
    -------
    pd.DataFrame
        Summary table with AIC, BIC, IAD, AD, Kendall's tau error, etc.
    """
    # Ensure list
    if not isinstance(copulas, list):
        copulas = [copulas]

    results = []

    u, v = pseudo_obs(data)
    u = np.asarray(u).flatten()
    v = np.asarray(v).flatten()

    assert u.ndim == 1 and v.ndim == 1, "u and v must be 1D after flattening"

    cdf = [u, v]

    for cop in copulas:
        if verbose:
            print(f"\nEvaluating: {cop.get_name()}")

        # Extract required data
        try:
            loglik = cop.log_likelihood_
            n_param = len(cop.bounds_param)
            n_obs = cop.n_obs
        except Exception as e:
            print(f"[ERROR] Missing attributes in copula '{cop.get_name()}': {e}")
            continue

        # Model selection criteria
        aic = cop.AIC()
        bic = cop.BIC()

        # IAD & AD: if quick, skip computation
        if quick:
            iad = np.nan
            ad = np.nan
        else:
            try:
                iad = cop.IAD(cdf)
            except Exception as e:
                print(f"[ERROR] IAD computation failed for {cop.get_name()}: {e}")
                iad = np.nan

            try:
                ad = cop.AD(cdf)
            except Exception as e:
                print(f"[ERROR] AD computation failed for {cop.get_name()}: {e}")
                ad = np.nan

        # Kendall's tau
        try:
            tau_error = cop.kendall_tau_error(data)
        except Exception as e:
            print(f"[ERROR] Kendall's tau error computation failed for {cop.get_name()}: {e}")
            tau_error = np.nan

        results.append({
            "Copula": cop.get_name(),
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








