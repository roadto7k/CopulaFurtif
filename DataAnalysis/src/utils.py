import pandas as pd
import numpy as np
from pathlib import Path
import os, sys
from typing import Tuple, Dict, List
from scipy.stats import kendalltau
from statsmodels.tsa.stattools import adfuller
from itertools import combinations
import json
from dataclasses import dataclass
from datetime import datetime


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, utc=False)
        except Exception:
            raise ValueError("Index non temporel et non convertible en datetime.")
    return df.sort_index()


def load_all_prices(data_path: str, symbols: List[str]) -> pd.DataFrame:
    """
    Charges toutes les colonnes 'close' des CSV présents dans DATA_PATH.
    On garde uniquement les SYMBOLS listés dans config (si présents).

    Chaque CSV doit avoir un index datetime (ou une première colonne parsable en datetime)
    et une colonne 'close'.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"DATA_PATH introuvable: {data_path.resolve()}")

    frames = []
    found = set()
    for sym in symbols:
        p = data_path / f"{sym}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, index_col=0)
        df = ensure_datetime_index(df)
        if 'close' not in df.columns:
            lower = {c.lower(): c for c in df.columns}
            if 'close' in lower:
                df.rename(columns={lower['close']: 'close'}, inplace=True)
            else:
                raise ValueError(f"Fichier {p.name} sans colonne 'close'. Colonnes: {df.columns.tolist()}")
        frames.append(df[['close']].rename(columns={'close': sym}))
        found.add(sym)

    if not frames:
        raise RuntimeError(f"Aucun CSV trouvé pour les symbols demandés dans {data_path}.")
    prices = pd.concat(frames, axis=1).sort_index()
    prices = prices[sorted(list(found))]
    prices = prices[~prices.index.duplicated(keep='last')]
    return prices


def compute_beta(x: pd.Series, y: pd.Series) -> float:
    """
    OLS slope of x ~ beta*y (no intercept). Robust to NaN/misaligned indices.
    """
    x, y = x.align(y, join='inner')
    mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 30 or np.std(y) < 1e-12:
        return np.nan
    try:
        beta = np.polyfit(y.values, x.values, 1)[0]
        return float(beta)
    except np.linalg.LinAlgError:
        return np.nan


def kss_test(series: pd.Series) -> Tuple[float, float]:
    """
    Kapetanios–Shin–Snell (2003) non-linear unit-root test (simplified).
    Returns (t_stat, crit_10pct). Reject H0 (unit root) if t_stat < crit (≈ -1.92).
    """
    x = series.dropna().astype(float).values
    if len(x) < 40:
        return (np.nan, -1.92)
    dx = np.diff(x)
    x_lag = x[:-1]
    z = x_lag ** 3
    y = dx
    try:
        beta = np.linalg.lstsq(z[:, None], y, rcond=None)[0][0]
        res = y - z * beta
        s2 = np.sum(res**2) / max(len(y) - 1, 1)
        se = np.sqrt(s2 / np.sum(z**2))
        t_stat = beta / (se if se > 0 else np.nan)
        return (float(t_stat), -1.92)
    except Exception:
        return (np.nan, -1.92)


def adf_pvalue(series: pd.Series, maxlag: int = 10) -> float:
    x = series.dropna().astype(float).values
    if len(x) < 30 or np.std(x) < 1e-12:
        return 1.0
    try:
        return float(adfuller(x, maxlag=maxlag, autolag='AIC')[1])
    except Exception:
        return 1.0


@dataclass
class CoinStats:
    symbol: str
    n_obs: int
    frac_nan: float
    std_close: float
    beta: float
    adf_pvalue: float
    kss_t: float
    kss_crit_10pct: float
    kendall_tau_abs: float
    accepted: bool


def build_spread(ref: pd.Series, coin: pd.Series, beta: float) -> pd.Series:
    spread = ref - beta * coin
    return spread.dropna()


def screen_coin(ref: pd.Series, coin: pd.Series, symbol: str, adf_level: float, maxlag: int = 10) -> Tuple[CoinStats, pd.Series]:
    ref, coin = ref.align(coin, join='inner')
    mask = ref.notna() & coin.notna() & np.isfinite(ref) & np.isfinite(coin)
    ref, coin = ref[mask], coin[mask]

    n_obs = len(ref)
    frac_nan = 0.0  # NaN
    std_close = float(np.std(coin.values)) if n_obs else 0.0

    beta = compute_beta(ref, coin)
    if not np.isfinite(beta):
        return CoinStats(symbol, n_obs, frac_nan, std_close, np.nan, 1.0, np.nan, -1.92, 0.0, False), pd.Series(dtype=float)

    spread = build_spread(ref, coin, beta)
    p_adf = adf_pvalue(spread)
    kss_t, kss_crit = kss_test(spread)

    try:
        ktau, _ = kendalltau(spread.values, ref.loc[spread.index].values)
        ktau_abs = abs(float(ktau)) if ktau is not None else 0.0
    except Exception:
        ktau_abs = 0.0

    accepted = (p_adf < adf_level) or (np.isfinite(kss_t) and kss_t < kss_crit)

    stats = CoinStats(symbol, n_obs, frac_nan, std_close, beta, p_adf, kss_t, kss_crit, ktau_abs, accepted)
    return stats, spread


def formation_pipeline(
    prices: pd.DataFrame,
    reference_symbol: str,
    adf_level: float = 0.10,
    min_obs: int = 200,
    top_k: int = 6,
    out_dir: str = "./artifacts",
) -> Dict[str, str]:
    """
    Runs the full screening + ranking + candidate pairing.
    Returns paths of generated artifacts.
    """
    out = Path(out_dir)
    (out / "spreads").mkdir(parents=True, exist_ok=True)

    if reference_symbol not in prices.columns:
        raise ValueError(f"REFERENCE_ASSET '{reference_symbol}' absent des prix chargés.")

    usable = []
    for c in prices.columns:
        if c == reference_symbol:
            continue
        series = prices[c].dropna()
        if len(series) >= min_obs and np.std(series.values) > 1e-9:
            usable.append(c)

    ref = prices[reference_symbol].dropna()

    rows = []
    spreads: Dict[str, pd.Series] = {}
    for sym in usable:
        stats, spread = screen_coin(ref, prices[sym], sym, adf_level=adf_level)
        rows.append(stats.__dict__)
        if stats.accepted and len(spread) >= min_obs:
            spreads[sym] = spread
            spread.to_frame(name=f"S_{reference_symbol}-{sym}").to_csv(out / "spreads" / f"{sym}.csv")

    df_summary = pd.DataFrame(rows).sort_values(["accepted", "kendall_tau_abs", "adf_pvalue"], ascending=[False, False, True])
    df_summary.to_csv(out / "formation_summary.csv", index=False)

    accepted = df_summary[df_summary["accepted"]].copy()
    top = accepted.sort_values("kendall_tau_abs", ascending=False).head(top_k)
    top.to_csv(out / "top_candidates.csv", index=False)

    pair_rows = []
    top_syms = top["symbol"].tolist()
    for a, b in combinations(top_syms, 2):
        pa = float(accepted.loc[accepted.symbol == a, "adf_pvalue"].values[0])
        pb = float(accepted.loc[accepted.symbol == b, "adf_pvalue"].values[0])
        ta = float(accepted.loc[accepted.symbol == a, "kendall_tau_abs"].values[0])
        tb = float(accepted.loc[accepted.symbol == b, "kendall_tau_abs"].values[0])
        score = (pa + pb) / 2.0 - 0.25 * (ta + tb)  # lower is better
        pair_rows.append({"coin1": a, "coin2": b, "score": score, "p_adf_max": max(pa, pb), "tau_mean": (ta + tb) / 2.0})

    df_pairs = pd.DataFrame(pair_rows).sort_values("score", ascending=True)
    df_pairs.to_csv(out / "candidate_pairs.csv", index=False)

    meta = {
        "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "reference": reference_symbol,
        "symbols": sorted(usable),
        "n_symbols": len(usable),
        "n_accepted": int(accepted.shape[0]),
        "top_k": int(min(top_k, accepted.shape[0])),
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "formation_summary": str(out / "formation_summary.csv"),
        "top_candidates": str(out / "top_candidates.csv"),
        "candidate_pairs": str(out / "candidate_pairs.csv"),
        "spreads_dir": str(out / "spreads"),
        "meta": str(out / "meta.json"),
    }