import numpy as np
import pandas as pd


def compute_beta(reference, coin):
    """
    OLS through the origin — Tadi & Witzany (2025), Eq. (6).

        β = Σ(P_ref · P_coin) / Σ(P_coin²)

    Aucun intercept n'est estimé.
    Le spread S = P_ref − β·P_coin doit ensuite être démeané AVANT le test ADF/KSS,
    pour rester cohérent avec Eq. (7) (régression sans constante).
    """
    df = pd.concat([reference, coin], axis=1).dropna()
    df = df[np.isfinite(df).all(axis=1)]

    if len(df) < 30:
        return np.nan

    r = df.iloc[:, 0].values
    c = df.iloc[:, 1].values

    if np.std(c) < 1e-12:
        return np.nan

    denom = float((c * c).sum())
    if denom == 0.0:
        return np.nan

    return float((r * c).sum() / denom)


def compute_spread(reference, coin):
    """
    Spread paper-faithful : S = P_ref − β·P_coin avec β estimé sans intercept.
    """
    beta = compute_beta(reference, coin)

    if not np.isfinite(beta):
        empty = pd.Series(index=reference.index, dtype=float).dropna()
        return empty, np.nan

    spread = (reference - beta * coin).dropna()
    return spread, beta