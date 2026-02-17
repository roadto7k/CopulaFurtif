import pandas as pd
import numpy as np
import math

def _to_datetime(s):
    return pd.to_datetime(s, utc=False)

def _safe_pct(x):
    return float(x) * 100.0

def _annualization_factor(interval: str):
    # Approx. number of bars per year
    if interval.endswith("m"):
        minutes = int(interval[:-1])
        return (365.0 * 24.0 * 60.0) / minutes
    if interval.endswith("h"):
        hours = int(interval[:-1])
        return (365.0 * 24.0) / hours
    if interval.endswith("d"):
        days = int(interval[:-1])
        return 365.0 / days
    return 365.0 * 24.0  # fallback

def performance_metrics(equity, returns, interval: str = "D"):
    eq = pd.Series(equity).replace([np.inf, -np.inf], np.nan).dropna()
    if len(eq) < 3:
        return dict(
            total_return=np.nan,
            annual_return=np.nan,
            annual_vol=np.nan,
            sharpe=np.nan,
            max_drawdown=np.nan,
            romad=np.nan,
        )

    rets = eq.pct_change().dropna()
    ann = _annualization_factor(interval)
    mu = rets.mean() * ann
    vol = rets.std(ddof=1) * math.sqrt(ann) if rets.std(ddof=1) > 0 else np.nan
    sharpe = (mu / vol) if (vol is not None and np.isfinite(vol) and vol > 0) else np.nan

    # drawdown
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    mdd = float(dd.min())
    tot = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    romad = (tot / abs(mdd)) if (np.isfinite(mdd) and mdd < 0) else np.nan

    return dict(
        total_return=tot,
        annual_return=float(mu),
        annual_vol=float(vol) if vol is not None else np.nan,
        sharpe=float(sharpe) if sharpe is not None else np.nan,
        max_drawdown=float(mdd),
        romad=float(romad) if romad is not None else np.nan,
    )
