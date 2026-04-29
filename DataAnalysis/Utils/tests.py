import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen


# ============================================================
# Helpers
# ============================================================

def _clean_1d_series(series, min_obs: int):
    """
    Convertit une série en array 1D float propre.
    Retourne None si la série est trop courte, constante ou invalide.
    """
    x = (
        pd.Series(series)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .astype(float)
        .values
    )

    if len(x) < min_obs:
        return None

    if not np.all(np.isfinite(x)):
        return None

    if np.std(x) < 1e-12:
        return None

    return x


# ============================================================
# ADF / Engle-Granger stationarity test — paper-faithful
# ============================================================

def run_adf_test(
    series,
    maxlag: int = 1,
    autolag=None,
    min_obs: int = 30,
):
    """
    Augmented Dickey-Fuller test pour le spread process — version Tadi & Witzany 2025.

    Spec littérale du papier (Eq. 7) :
        ΔS_t = β·S_{t-1} + Σ γ_i·ΔS_{t-i} + ε_t      (PAS de constante)

    Procédure :
        1. Démean explicite du spread.
        2. ADF avec regression='n' (no constant) sur la série démeanée.

    Cohérent avec Eq. (6) où β du spread est estimé sans intercept (compute_beta).
    Mélanger β-no-intercept avec ADF regression='c' donne des p-values fausses
    (testé : 0.252 au lieu de 0.087 sur LTC-BTC Week 1, 5-min).

    Default :
        maxlag = 1, autolag = None  → déterministe.
                                      Mettre autolag='AIC' pour la sélection auto.
    """
    x = _clean_1d_series(series, min_obs=min_obs)

    if x is None:
        return np.nan, 1.0, {}

    # Demean AVANT l'ADF regression='n', sinon la mean est traitée comme du signal
    # de non-stationnarité.
    x = x - np.mean(x)

    try:
        result = adfuller(
            x,
            maxlag=maxlag,
            autolag=autolag,
            regression="n",
        )
        return float(result[0]), float(result[1]), result[4]
    except Exception:
        return np.nan, 1.0, {}


# ============================================================
# KSS nonlinear unit-root test — paper Eq. (10)
# ============================================================

def kss_test(
    series,
    crit: float = -1.92,
    demean: bool = True,
    lags: int = 0,
    min_obs: int = 40,
):
    """
    Kapetanios-Shin-Snell nonlinear unit-root test.

    Régression (Eq. 10 du papier, Taylor expansion avec p=1, c=0) :
        ΔS_t = δ · S_{t-1}^3 + ε_t

    Décision :
        H0 : unit root / non-stationnaire
        H1 : nonlinear stationarity
        Reject H0 ssi  t_stat < crit

    Defaults paper :
        crit = -1.92    (Kapetanios et al. 2003, 10% asymptotic critical value)
        demean = True   (le papier teste sur spread démeané, comme ADF)
        lags = 0        (Eq. 10 = forme simple sans lags augmentés)

    Note : pas d'intercept dans la régression test, donc le démean explicite est requis.
    """
    x = _clean_1d_series(series, min_obs=min_obs)

    if x is None:
        return np.nan, crit

    if demean:
        x = x - np.mean(x)

    dx = np.diff(x)
    x_lag = x[:-1]

    if len(dx) <= lags + 5:
        return np.nan, crit

    y = dx[lags:]
    z = x_lag[lags:] ** 3

    if lags > 0:
        lagged_diffs = [dx[lags - j: -j] for j in range(1, lags + 1)]
        X = np.column_stack([z] + lagged_diffs)
    else:
        X = z[:, None]

    if X.shape[0] <= X.shape[1]:
        return np.nan, crit

    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        return np.nan, crit

    try:
        beta_hat, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ beta_hat

        nobs, k = X.shape
        dof = max(nobs - k, 1)
        sigma2 = float(np.sum(residuals ** 2) / dof)

        xtx_inv = np.linalg.pinv(X.T @ X)
        se_delta = float(np.sqrt(sigma2 * xtx_inv[0, 0]))
        delta = float(beta_hat[0])

        if se_delta <= 0 or not np.isfinite(se_delta):
            return np.nan, crit

        t_stat = delta / se_delta
        return float(t_stat), crit

    except Exception:
        return np.nan, crit


# ============================================================
# Johansen test (inchangé)
# ============================================================

def johansen_stat(x, y):
    """
    Johansen trace statistic — pas utilisé pour la réplication Table 8,
    conservé pour le framework plus large.
    """
    arr = pd.concat([x, y], axis=1).dropna().values

    if arr.shape[0] < 40:
        return np.nan, np.nan

    result = coint_johansen(arr, det_order=0, k_ar_diff=1)

    trace_stat = float(result.lr1[0])
    crit_val = float(result.cvt[0, 1])  # 5%

    return trace_stat, crit_val