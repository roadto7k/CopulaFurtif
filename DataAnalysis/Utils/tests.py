import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def run_adf_test(series):
    x = series.dropna().values
    if len(x) < 30 or np.std(x) < 1e-12:
        return np.nan, 1.0, {}
    result = adfuller(x, autolag='AIC')
    stat, pvalue, _, _, crit, _ = result
    return stat, pvalue, crit

#TT ca important aussi HERE
def kss_test(series):
    x = np.array(series.dropna(), dtype=float)
    if len(x) < 40:
        return np.nan, -1.92
    dx = np.diff(x)
    x_lag = x[:-1]
    y = dx
    z = x_lag**3
    try:
        beta = np.linalg.lstsq(z[:, None], y, rcond=None)[0][0]
        res = y - z * beta
        s2 = np.sum(res**2) / max(len(y) - 1, 1)
        se = np.sqrt(s2 / np.sum(z**2))
        t_stat = beta / (se if se > 0 else np.nan)
    except Exception:
        t_stat = np.nan
    return t_stat, -1.92  # environ 10%

def johansen_stat(x, y):
    arr = pd.concat([x, y], axis=1).dropna().values
    if arr.shape[0] < 40:
        return np.nan, np.nan
    result = coint_johansen(arr, det_order=0, k_ar_diff=1)
    trace_stat = float(result.lr1[0])
    crit_val = float(result.cvt[0,1])  # 5%
    return trace_stat, crit_val