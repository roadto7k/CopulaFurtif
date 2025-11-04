import numpy as np
import pandas as pd

def compute_beta(x, y):
    x, y = x.align(y, join='inner')
    mask = (~x.isna()) & (~y.isna()) & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 30 or np.std(y[mask]) < 1e-12:
        return np.nan
    return np.polyfit(y[mask], x[mask], 1)[0]

def compute_spread(reference, coin):
    beta = compute_beta(reference, coin)
    spread = reference - beta * coin if np.isfinite(beta) else pd.Series(index=reference.index, dtype=float)
    return spread.dropna(), beta