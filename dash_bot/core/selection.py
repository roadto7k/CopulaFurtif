# dash_bot/core/selection.py
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

from DataAnalysis.Utils.spread import compute_spread
from DataAnalysis.Utils.tests import run_adf_test, kss_test

def select_stationary_spreads(
    prices: pd.DataFrame,
    ref: str,
    candidates: List[str],
    adf_alpha: float,
    use_kss: bool,
    kss_crit: float,
    min_obs: int,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series], Dict[str, float]]:
    """
    Calcule spreads S_i = P_ref - beta_i * P_i sur la fenêtre de formation,
    filtre stationnarité (ADF, optionnel KSS), retourne:
      - summary dataframe
      - spreads dict
      - betas dict
    """
    rows = []
    spreads: Dict[str, pd.Series] = {}
    betas: Dict[str, float] = {}

    if ref not in prices.columns:
        return pd.DataFrame(), spreads, betas

    ref_series = prices[ref].dropna()

    for coin in candidates:
        if coin == ref or coin not in prices.columns:
            continue

        s, beta = compute_spread(ref_series, prices[coin])
        s = s.dropna()
        betas[coin] = float(beta) if beta is not None and np.isfinite(beta) else np.nan

        if len(s) < min_obs or np.std(s) < 1e-12:
            rows.append(dict(coin=coin, n=len(s), beta=betas[coin], adf_p=np.nan, kss=np.nan, accepted=False))
            continue

        adf_stat, adf_p, _ = run_adf_test(s)
        kss_stat, _ = kss_test(s) if use_kss else (np.nan, kss_crit)

        accepted = bool(np.isfinite(adf_p) and adf_p < adf_alpha)
        if use_kss:
            accepted = accepted and bool(np.isfinite(kss_stat) and kss_stat < kss_crit)

        rows.append(dict(
            coin=coin,
            n=len(s),
            beta=betas[coin],
            adf_p=float(adf_p) if np.isfinite(adf_p) else np.nan,
            kss=float(kss_stat) if np.isfinite(kss_stat) else np.nan,
            accepted=accepted,
        ))
        spreads[coin] = s

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(["accepted", "adf_p"], ascending=[False, True]).reset_index(drop=True)
    return summary, spreads, betas

def rank_coins(
    prices: pd.DataFrame,
    spreads: Dict[str, pd.Series],
    ref: str,
    coins: List[str],
    method: str,
) -> pd.DataFrame:
    """
    Ranking par Kendall τ (comme dans l'article).
    - kendall_prices: τ(P_ref, P_coin)
    - kendall_spread_ref: τ(S_ref,coin, P_ref) (style ton dashboard existant)
    """
    from scipy.stats import kendalltau

    if ref not in prices.columns:
        return pd.DataFrame(columns=["coin", "tau", "abs_tau"])

    ref_series = prices[ref].dropna()
    rows = []

    for c in coins:
        if c == ref or c not in prices.columns:
            continue
        try:
            if method == "kendall_spread_ref":
                s = spreads.get(c)
                if s is None or s.empty:
                    continue
                idx = s.index.intersection(ref_series.index)
                if len(idx) < 30:
                    continue
                tau, _ = kendalltau(s.loc[idx], ref_series.loc[idx])
            else:
                idx = prices[[ref, c]].dropna().index
                if len(idx) < 30:
                    continue
                tau, _ = kendalltau(prices.loc[idx, ref], prices.loc[idx, c])
            if tau is None or not np.isfinite(tau):
                continue
            rows.append(dict(coin=c, tau=float(tau), abs_tau=float(abs(tau))))
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("abs_tau", ascending=False).reset_index(drop=True)
    return df
