# dash_bot/core/selection.py

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

from DataAnalysis.Utils.spread import compute_spread
from DataAnalysis.Utils.tests import run_adf_test, kss_test


def select_stationary_spreads(
    prices: pd.DataFrame,
    ref: str,
    candidates: List[str],
    adf_alpha: float,
    cointegration_test: str,   # "adf" | "kss" | "both"
    kss_crit: float,
    min_obs: int,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series], Dict[str, float]]:
    """
    Step 1 — Cointegration filter.

    For each altcoin i:
        S_i(t) = P_ref(t) - beta_i * P_i(t)

    beta_i is estimated without intercept in compute_spread().

    Then:
        - "adf"  : Engle-Granger = ADF on the spread
        - "kss"  : nonlinear unit-root test on the spread
        - "both" : ADF AND KSS, stricter than the paper
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

        spread, beta = compute_spread(ref_series, prices[coin])
        spread = spread.dropna()

        beta = float(beta) if beta is not None and np.isfinite(beta) else np.nan
        betas[coin] = beta

        if len(spread) < min_obs or np.std(spread) < 1e-12:
            rows.append({
                "coin": coin,
                "n": len(spread),
                "beta": beta,
                "adf_stat": np.nan,
                "adf_p": np.nan,
                "kss": np.nan,
                "accepted": False,
            })
            continue

        run_adf = cointegration_test in ("adf", "both")
        run_kss = cointegration_test in ("kss", "both")

        if run_adf:
            adf_stat, adf_p, _ = run_adf_test(spread)
        else:
            adf_stat, adf_p = np.nan, np.nan

        if run_kss:
            kss_stat, _ = kss_test(spread, crit=kss_crit)
        else:
            kss_stat = np.nan

        if cointegration_test == "adf":
            accepted = np.isfinite(adf_p) and adf_p < adf_alpha

        elif cointegration_test == "kss":
            accepted = np.isfinite(kss_stat) and kss_stat < kss_crit

        elif cointegration_test == "both":
            accepted = (
                np.isfinite(adf_p)
                and adf_p < adf_alpha
                and np.isfinite(kss_stat)
                and kss_stat < kss_crit
            )

        else:
            raise ValueError(
                "cointegration_test must be one of: 'adf', 'kss', 'both'"
            )

        rows.append({
            "coin": coin,
            "n": len(spread),
            "beta": beta,
            "adf_stat": float(adf_stat) if np.isfinite(adf_stat) else np.nan,
            "adf_p": float(adf_p) if np.isfinite(adf_p) else np.nan,
            "kss": float(kss_stat) if np.isfinite(kss_stat) else np.nan,
            "accepted": bool(accepted),
        })

        spreads[coin] = spread

    summary = pd.DataFrame(rows)

    if not summary.empty:
        if cointegration_test == "adf":
            summary = summary.sort_values(
                ["accepted", "adf_p"],
                ascending=[False, True],
            ).reset_index(drop=True)

        elif cointegration_test == "kss":
            summary = summary.sort_values(
                ["accepted", "kss"],
                ascending=[False, True],
            ).reset_index(drop=True)

        else:
            summary = summary.sort_values(
                ["accepted", "adf_p", "kss"],
                ascending=[False, True, True],
            ).reset_index(drop=True)

    return summary, spreads, betas


def rank_coins_by_kendall_returns(
    prices: pd.DataFrame,
    ref: str,
    coins: List[str],
    min_obs: int = 30,
) -> pd.DataFrame:
    """
    Step 2 — Paper-faithful ranking.

    Among coins accepted by the cointegration filter, rank altcoins by
    Kendall's tau between BTCUSDT log-returns and altcoin log-returns.

        r_ref(t) = log(P_ref(t)) - log(P_ref(t-1))
        r_i(t)   = log(P_i(t))   - log(P_i(t-1))

    Then select the two altcoins with the highest tau.
    """
    from scipy.stats import kendalltau

    if ref not in prices.columns:
        return pd.DataFrame(columns=["coin", "tau", "kendall_pvalue"])

    ref_returns = (
        np.log(prices[ref])
        .replace([np.inf, -np.inf], np.nan)
        .diff()
        .dropna()
    )

    rows = []

    for coin in coins:
        if coin == ref or coin not in prices.columns:
            continue

        alt_returns = (
            np.log(prices[coin])
            .replace([np.inf, -np.inf], np.nan)
            .diff()
            .dropna()
        )

        idx = ref_returns.index.intersection(alt_returns.index)

        if len(idx) < min_obs:
            continue

        tau, pvalue = kendalltau(
            ref_returns.loc[idx],
            alt_returns.loc[idx],
        )

        if tau is None or not np.isfinite(tau):
            continue

        rows.append({
            "coin": coin,
            "tau": float(tau),
            "kendall_pvalue": float(pvalue) if np.isfinite(pvalue) else np.nan,
        })

    ranked = pd.DataFrame(rows)

    if not ranked.empty:
        # Paper logic: highest tau with BTC, not highest abs(tau).
        ranked = ranked.sort_values("tau", ascending=False).reset_index(drop=True)

    return ranked


def select_pair_from_formation_window(
    form_data: pd.DataFrame,
    ref: str,
    candidates: List[str],
    cointegration_test: str,
    adf_alpha: float = 0.10,
    kss_crit: float = -1.92,
    min_obs: int = 50,
    verbose: bool = False,
):
    """
    Full pair-selection pipeline:

    1. Build BTC-altcoin spreads.
    2. Keep only stationary spreads using ADF or KSS.
    3. Rank accepted coins by Kendall tau on log-returns vs BTC.
    4. Return top two coins as the selected pair.
    """
    summary, spreads, betas = select_stationary_spreads(
        prices=form_data,
        ref=ref,
        candidates=candidates,
        adf_alpha=adf_alpha,
        cointegration_test=cointegration_test,
        kss_crit=kss_crit,
        min_obs=min_obs,
    )

    if summary.empty:
        return None, summary, spreads, betas, pd.DataFrame()

    accepted = summary.loc[summary["accepted"], "coin"].tolist()

    if verbose:
        stat_col = "adf_p" if cointegration_test == "adf" else "kss"
        stat_label = "p-value" if cointegration_test == "adf" else "t-stat"

        print(
            f"    [{cointegration_test.upper()}] accepted ({len(accepted)}): "
            f"{[c.replace('USDT', '') for c in accepted]}"
        )

        for _, row in summary[["coin", stat_col, "accepted"]].iterrows():
            tick = "✓" if row["accepted"] else " "
            val = f"{row[stat_col]:.4f}" if pd.notna(row[stat_col]) else "nan"
            print(f"      {tick} {row['coin'].replace('USDT', ''):<8} {stat_label}={val}")

    if len(accepted) < 2:
        return None, summary, spreads, betas, pd.DataFrame()

    ranked = rank_coins_by_kendall_returns(
        prices=form_data,
        ref=ref,
        coins=accepted,
        min_obs=30,
    )

    if ranked.empty or len(ranked) < 2:
        return None, summary, spreads, betas, ranked

    c1 = ranked.iloc[0]["coin"]
    c2 = ranked.iloc[1]["coin"]

    stat_col = "adf_p" if cointegration_test == "adf" else "kss"

    def get_stat(coin):
        row = summary.loc[summary["coin"] == coin]
        return float(row[stat_col].iloc[0]) if not row.empty else np.nan

    result = {
        "pair": f"{c1.replace('USDT', '')}-{c2.replace('USDT', '')}",
        "coin1": c1,
        "coin2": c2,
        "stat1": get_stat(c1),
        "stat2": get_stat(c2),
        "tau1": float(ranked.iloc[0]["tau"]),
        "tau2": float(ranked.iloc[1]["tau"]),
    }

    if verbose:
        print(
            f"    → Pair: {result['pair']}  "
            f"tau1={result['tau1']:.4f}, tau2={result['tau2']:.4f}"
        )

    return result, summary, spreads, betas, ranked