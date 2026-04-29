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
    cointegration_test: str,         # "adf" | "kss" | "both"
    kss_crit: float,
    min_obs: int,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series], Dict[str, float]]:
    """
    Calcule les spreads S_i = P_ref − β_i · P_i sur la fenêtre de formation,
    filtre stationnarité selon la stratégie cointegration choisie.

    cointegration_test :
      "adf"  → Engle-Granger : régression sans intercept (Eq. 6) + ADF sur le spread (Eq. 7)
      "kss"  → Régression sans intercept + KSS nonlinear unit-root (Eq. 10)
      "both" → ADF AND KSS (strict conjunction, hors paper)

    β estimé via OLS sans intercept (Eq. 6 du papier).
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
            rows.append(dict(
                coin=coin, n=len(s), beta=betas[coin],
                adf_p=np.nan, kss=np.nan, accepted=False,
            ))
            continue

        run_adf = cointegration_test in ("adf", "both")
        run_kss = cointegration_test in ("kss", "both")

        adf_stat, adf_p, _ = run_adf_test(s) if run_adf else (np.nan, np.nan, None)
        kss_stat, _ = kss_test(s) if run_kss else (np.nan, kss_crit)

        if cointegration_test == "adf":
            accepted = bool(np.isfinite(adf_p) and adf_p < adf_alpha)
        elif cointegration_test == "kss":
            accepted = bool(np.isfinite(kss_stat) and kss_stat < kss_crit)
        else:  # "both"
            accepted = (
                bool(np.isfinite(adf_p) and adf_p < adf_alpha)
                and bool(np.isfinite(kss_stat) and kss_stat < kss_crit)
            )

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
        summary = summary.sort_values(
            ["accepted", "adf_p"], ascending=[False, True]
        ).reset_index(drop=True)
    return summary, spreads, betas


def rank_coins(
    prices: pd.DataFrame,
    spreads: Dict[str, pd.Series],
    ref: str,
    coins: List[str],
    method: str = "kendall_returns",
) -> pd.DataFrame:
    """
    Ranking des altcoins par Kendall's τ avec BTC.

    Paper Tadi & Witzany (p.14) :
      "After calculating τ of BTCUSDT with each of the 19 altcoins, we select
       the two altcoins that have the highest τ with BTCUSDT"

    Le papier ne précise PAS si τ est sur prix ou returns. Mais :
      - τ sur prix bruts : deux random walks qui dérivent ensemble auront τ→1
        même sans relation économique. Sur Week 1, ça remonte TRX, XRP, XLM
        (sub-1$ qui ont grimpé en parallèle), ce qui ne matche pas Table 8.
      - τ sur returns : returns stationnaires, mesure la vraie co-dépendance.
        Sur Week 1 → LTC=0.598, BCH=0.557 → match exact Table 8.

    → Default = "kendall_returns" car c'est le seul qui reproduit le papier.

    Méthodes :

      kendall_returns      → τ(r_BTC, r_alt) sur log-returns [DEFAULT, paper]

      kendall_prices       → τ(P_BTC, P_alt) sur prix bruts
                             ⚠ Statistiquement discutable sur séries non-stationnaires

      kendall_spread_pair  → τ(S_i, S_j) per Eq. 33
                             Lecture alternative de la formule (contredit le texte)

      kendall_spread_ref   → τ(S_alt, P_BTC)
                             Hybride, pas de justification théorique
    """
    from scipy.stats import kendalltau
    from itertools import combinations

    if ref not in prices.columns:
        return pd.DataFrame(columns=["coin", "tau", "abs_tau"])

    # ─────────────────────────────────────────────────────────────────
    # kendall_spread_pair — τ(S_i, S_j) per Eq. 33
    # ─────────────────────────────────────────────────────────────────
    if method == "kendall_spread_pair":
        all_pairs = []

        for c1, c2 in combinations(coins, 2):
            s1 = spreads.get(c1)
            s2 = spreads.get(c2)
            if s1 is None or s2 is None or s1.empty or s2.empty:
                continue
            idx = s1.index.intersection(s2.index)
            if len(idx) < 30:
                continue
            try:
                tau, _ = kendalltau(s1.loc[idx], s2.loc[idx])
                if np.isfinite(tau):
                    all_pairs.append((c1, c2, float(tau)))
            except Exception:
                continue

        if not all_pairs:
            return pd.DataFrame(columns=["coin", "tau", "abs_tau"])

        all_pairs.sort(key=lambda r: -abs(r[2]))
        c1, c2, tau = all_pairs[0]

        return pd.DataFrame([
            dict(coin=c1, tau=tau, abs_tau=abs(tau)),
            dict(coin=c2, tau=tau, abs_tau=abs(tau)),
        ])

    # ─────────────────────────────────────────────────────────────────
    # kendall_returns — DEFAULT, paper-faithful
    # ─────────────────────────────────────────────────────────────────
    if method == "kendall_returns":
        ref_returns = np.log(prices[ref]).diff().dropna()
        rows = []
        for c in coins:
            if c == ref or c not in prices.columns:
                continue
            try:
                alt_returns = np.log(prices[c]).diff().dropna()
                idx = ref_returns.index.intersection(alt_returns.index)
                if len(idx) < 30:
                    continue
                tau, _ = kendalltau(ref_returns.loc[idx], alt_returns.loc[idx])
                if pd.isna(tau) or not np.isfinite(tau):
                    continue
                rows.append(dict(coin=c, tau=float(tau), abs_tau=float(abs(tau))))
            except Exception:
                continue

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("abs_tau", ascending=False).reset_index(drop=True)
        return df

    # ─────────────────────────────────────────────────────────────────
    # kendall_prices / kendall_spread_ref (alternatives)
    # ─────────────────────────────────────────────────────────────────
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
            else:  # kendall_prices
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