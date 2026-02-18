# dash_bot/data/cleaning.py
import pandas as pd
from typing import Dict, Tuple

def clean_prices(
    prices: pd.DataFrame,
    ref: str,
    min_coverage: float,
    min_points: int,
    ffill_limit: int = 2,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Nettoie les données:
    - index trié / unique
    - supprime les symboles avec couverture trop faible (vs timeline du ref)
    - supprime les symboles avec trop peu de points non-NA
    - comble petites lacunes (ffill limit)
    """
    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="first")]

    # keep only numeric
    for c in prices.columns:
        prices[c] = pd.to_numeric(prices[c], errors="coerce")

    reasons: Dict[str, str] = {}
    if ref in prices.columns:
        ref_mask = prices[ref].notna()
        if ref_mask.sum() < min_points:
            # can't do much; return as-is
            return prices, {"__all__": f"Ref {ref} has too few points ({int(ref_mask.sum())})"}
    else:
        ref_mask = pd.Series(True, index=prices.index)

    cov = prices.loc[ref_mask].notna().mean()
    pts = prices.loc[ref_mask].notna().sum()

    for sym in list(prices.columns):
        if sym == ref:
            continue
        if float(cov.get(sym, 0.0)) < float(min_coverage):
            reasons[sym] = f"coverage {float(cov.get(sym, 0.0)):.2%} < {min_coverage:.2%}"
        elif int(pts.get(sym, 0)) < int(min_points):
            reasons[sym] = f"non-NA points {int(pts.get(sym, 0))} < {int(min_points)}"

    drop_cols = [s for s in reasons.keys() if s in prices.columns]
    if drop_cols:
        prices = prices.drop(columns=drop_cols, errors="ignore")

    # fill small holes only (avoid bridging long gaps)
    prices = prices.ffill(limit=int(ffill_limit))

    return prices, reasons

def clean_prices_basic(prices):
    """Nettoyage sans filtre look-ahead sur l'univers (walk-forward friendly).
    - index datetime trié / unique
    - conversion numérique
    - supprime les lignes entièrement NaN
    Important: pas de ffill ici (évite de créer des prix fictifs pour l'exécution).
    """
    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index)
    prices = prices[~prices.index.duplicated(keep="last")]
    prices = prices.sort_index()
    for c in prices.columns:
        prices[c] = pd.to_numeric(prices[c], errors="coerce")
    prices = prices.dropna(how="all")
    return prices

def _map_usdt_to_yf(symbol_usdt: str) -> str:
    # 'BTCUSDT' -> 'BTC-USD'
    s = symbol_usdt.upper()
    if s.endswith("USDT"):
        base = s[:-4]
        return f"{base}-USD"
    return s