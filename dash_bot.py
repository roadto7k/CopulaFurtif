
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import math
import json
import itertools
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table, Input, Output, State
import plotly.graph_objs as go

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    from DataAnalysis.Utils.spread import compute_spread, compute_beta
    from DataAnalysis.Utils.tests import run_adf_test, kss_test
    from DataAnalysis.Utils.copulas import pseudo_obs, fit_copulas
    from DataAnalysis.Utils.load_prices import load_all_prices
    try:
        from DataAnalysis.config import DATA_PATH  # type: ignore
    except Exception:
        DATA_PATH = None
except Exception:
    from spread import compute_spread, compute_beta  # type: ignore
    from tests import run_adf_test, kss_test  # type: ignore
    from copulas import pseudo_obs, fit_copulas  # type: ignore
    from load_prices import load_all_prices  # type: ignore
    DATA_PATH = None

HAS_COPULAFURTIF = False
try:
    from CopulaFurtif.copulas import CopulaFactory, CopulaType
    HAS_COPULAFURTIF = True
except Exception:
    CopulaFactory = None  # type: ignore
    CopulaType = None  # type: ignore

HAS_SM_COPULA = False
try:
    from statsmodels.distributions.copula.api import (
        GaussianCopula, StudentTCopula, ClaytonCopula, FrankCopula, GumbelCopula
    )
    HAS_SM_COPULA = True
    SM_FAMILIES = {'gaussian','normal','student','student-t','t','clayton','frank','gumbel'}
except Exception:
    SM_FAMILIES = set()
    GaussianCopula = StudentTCopula = ClaytonCopula = FrankCopula = GumbelCopula = None  # type: ignore

try:
    import yfinance as yf  # type: ignore
    HAS_YFINANCE = True
except Exception:
    HAS_YFINANCE = False

try:
    import ccxt  # type: ignore
    HAS_CCXT = True
except Exception:
    HAS_CCXT = False

DROPDOWN_DARK_CSS = """
/* Fix dcc.Dropdown (react-select) en thème sombre */
.Select-control {
  background-color: #222 !important;
  color: #fff !important;
  border-color: #444 !important;
}

.Select-menu-outer {
  background-color: #222 !important;
  color: #fff !important;
  border-color: #444 !important;
}

.Select-option {
  background-color: #222 !important;
  color: #fff !important;
}

.Select-option.is-focused {
  background-color: #333 !important;
  color: #fff !important;
}

.Select-value-label, .Select-placeholder {
  color: #fff !important;
}
"""


DEFAULT_USDT_SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BCHUSDT', 'XRPUSDT', 'EOSUSDT',
    'LTCUSDT', 'TRXUSDT', 'ETCUSDT', 'LINKUSDT', 'XLMUSDT',
    'ADAUSDT', 'XMRUSDT', 'DASHUSDT', 'ZECUSDT', 'XTZUSDT',
    'ATOMUSDT', 'BNBUSDT', 'ONTUSDT', 'IOTAUSDT', 'BATUSDT'
]

DEFAULT_USDT_SYMBOLS_EXT = DEFAULT_USDT_SYMBOLS + [
    'SOLUSDT', 'DOGEUSDT', 'MATICUSDT', 'AVAXUSDT', 'DOTUSDT', 'UNIUSDT',
    'FILUSDT', 'NEARUSDT', 'ALGOUSDT', 'ICPUSDT', 'AAVEUSDT', 'SUSHIUSDT',
    'RUNEUSDT', 'EGLDUSDT', 'KSMUSDT', 'LDOUSDT', 'OPUSDT', 'ARBUSDT',
]

DEFAULT_REFERENCE = "BTCUSDT"

INTERVALS = [
    ("5m", "5m"),
    ("15m", "15m"),
    ("1h", "1h"),
    ("4h", "4h"),
    ("1d", "1d"),
]

STRATEGIES = [
    ("reference_copula", "Proposed: Reference-asset-based copula (spreads vs ref)"),
    ("cointegration_z", "Baseline: Cointegration z-score (spread between two coins)"),
    ("return_copula", "Baseline: Return-based copula (log-returns)"),
    ("level_copula", "Baseline: Level-based copula (CMI on conditional probs)"),
]

RANK_METHODS = [
    ("kendall_prices", "Kendall τ(Price(ref), Price(coin))"),
    ("kendall_spread_ref", "Kendall τ(Spread(ref,coin), Price(ref))"),
]

COPULA_PICK = [
    ("best_aic", "Auto (best by AIC)"),
    ("manual", "Manual (force a copula family)"),
]


def _to_datetime(s: str) -> pd.Timestamp:
    return pd.to_datetime(s, utc=False)

def _safe_pct(x: float) -> float:
    return float(x) * 100.0

def _annualization_factor(interval: str) -> float:
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

def performance_metrics(equity: pd.Series, interval: str) -> Dict[str, float]:
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


def _map_usdt_to_yf(symbol_usdt: str) -> str:
    # 'BTCUSDT' -> 'BTC-USD'
    s = symbol_usdt.upper()
    if s.endswith("USDT"):
        base = s[:-4]
        return f"{base}-USD"
    return s

def fetch_prices_yfinance(symbols: List[str], start: str, end: str, interval: str) -> pd.DataFrame:
    if not HAS_YFINANCE:
        raise RuntimeError("yfinance non installé: pip install yfinance")
    tickers = [_map_usdt_to_yf(s) for s in symbols]
    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    out = {}
    if isinstance(df.columns, pd.MultiIndex):
        for t, s in zip(tickers, symbols):
            if (t, "Close") in df.columns:
                out[s] = df[(t, "Close")]
    else:
        # single ticker
        if "Close" in df.columns:
            out[symbols[0]] = df["Close"]

    res = pd.DataFrame(out).sort_index()
    res.index = pd.to_datetime(res.index)
    return res

def fetch_prices_binance_ccxt(symbols: List[str], start: str, end: str, interval: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    if not HAS_CCXT:
        raise RuntimeError("ccxt non installé: pip install ccxt")
    # Binance futures: binanceusdm (USDT-margined perpetual futures)
    ex = ccxt.binanceusdm({"enableRateLimit": True})
    ex.load_markets()

    tf_map = {"5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}
    timeframe = tf_map.get(interval, interval)

    since_ms = int(pd.to_datetime(start).timestamp() * 1000)
    end_ms = int(pd.to_datetime(end).timestamp() * 1000)

    out: Dict[str, pd.Series] = {}
    errors: Dict[str, str] = {}

    for sym in symbols:
        market_sym = sym.replace("USDT", "/USDT")
        if market_sym not in ex.markets:
            errors[sym] = f"Market not found on binanceusdm: {market_sym}"
            continue

        all_rows = []
        since = since_ms
        try:
            # paginate
            while True:
                ohlcv = ex.fetch_ohlcv(market_sym, timeframe=timeframe, since=since, limit=1500)
                if not ohlcv:
                    break
                all_rows.extend(ohlcv)
                last = ohlcv[-1][0]
                since = last + 1
                if last >= end_ms or len(ohlcv) < 1500:
                    break
        except Exception as e:
            errors[sym] = f"fetch_ohlcv error: {type(e).__name__}: {e}"
            continue

        if not all_rows:
            errors[sym] = "No OHLCV returned (empty)."
            continue

        arr = np.array(all_rows, dtype=float)
        idx = pd.to_datetime(arr[:, 0], unit="ms")
        close = pd.Series(arr[:, 4], index=idx)
        close = close[(close.index >= pd.to_datetime(start)) & (close.index < pd.to_datetime(end))]
        if close.dropna().empty:
            errors[sym] = "Only NaNs after date filter."
            continue
        out[sym] = close

    df = pd.DataFrame(out).sort_index()
    return df, errors

def load_prices_csv(data_path: str) -> pd.DataFrame:
    df = load_all_prices(data_path)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()

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

def clean_prices_basic(prices: pd.DataFrame) -> pd.DataFrame:
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



def _nudge_params(p: np.ndarray) -> np.ndarray:
    """
    Évite les paramètres exactement sur des bornes ouvertes (ex: theta=0.01 alors que (0.01,100)).
    On pousse très légèrement vers l'intérieur.
    """
    p = np.array(p, dtype=float).copy()
    # petits epsilons relatifs
    def nudge_val(x: float) -> float:
        # bornes fréquentes vues dans tes logs
        if np.isclose(x, 0.01):
            return 0.0101
        if np.isclose(x, 0.0001):
            return 0.0001001
        if np.isclose(x, 1.01):
            return 1.0101
        # corr (gaussian / student)
        if abs(x) >= 1.0:
            return float(np.clip(x, -0.999, 0.999))
        return x

    return np.array([nudge_val(float(x)) for x in p], dtype=float)


# -----------------------------------------------------------------------------
# Copula helpers
# -----------------------------------------------------------------------------
def _get_all_copula_values() -> List[str]:
    if HAS_COPULAFURTIF and CopulaType is not None:
        return [ct.value for ct in list(CopulaType)]
    # fallback limited
    return ["gaussian", "student", "clayton", "frank", "gumbel"]

def _copula_type_from_value(val: str):
    if not (HAS_COPULAFURTIF and CopulaType is not None):
        return None
    for ct in list(CopulaType):
        if ct.value.lower() == str(val).lower() or ct.name.lower() == str(val).lower():
            return ct
    return None

def build_copula(name: str, params: Any):
    """
    Reconstruit un objet copula à partir du nom (value CopulaType) + params.
    Priorité: CopulaFurtif, fallback statsmodels (quelques familles).
    """
    nm = str(name)
    if HAS_COPULAFURTIF and CopulaFactory is not None and not (HAS_SM_COPULA and str(name).lower() in SM_FAMILIES):
        ct = _copula_type_from_value(nm)
        if ct is None:
            raise ValueError(f"CopulaType introuvable pour '{nm}' (vérifie l'Enum CopulaType).")
        cop = CopulaFactory.create(ct)
        # params peut être scalaire / tuple / array
        p = _nudge_params(np.atleast_1d(params).astype(float))
        if hasattr(cop, "set_parameters"):
            cop.set_parameters(p)
        elif hasattr(cop, "set_params"):
            cop.set_params(p)
        else:
            # dernier recours
            try:
                cop.set_parameters(p)
            except Exception:
                pass
        return cop

    # statsmodels fallback
    nm = nm.lower()
    if not HAS_SM_COPULA:
        raise RuntimeError("Aucun backend copula dispo (CopulaFurtif ou statsmodels).")

    p = _nudge_params(np.atleast_1d(params).astype(float))
    if nm in ("gaussian", "normal"):
        # Student/gaussian in statsmodels use corr param
        cop = GaussianCopula()
        try:
            # corr must be within [-1,1]
            cop.corr = float(np.clip(p[0], -0.999, 0.999))  # type: ignore
        except Exception:
            pass
        return cop
    if nm in ("student", "student-t", "t"):
        df = float(p[1]) if len(p) > 1 else 5.0
        df = max(df, 2.1001)
        cop = StudentTCopula(df=df)
        # corr + df
        try:
            cop.corr = float(np.clip(p[0], -0.999, 0.999))  # type: ignore
        except Exception:
            pass
        return cop
    if nm == "clayton":
        return ClaytonCopula(theta=float(p[0]))
    if nm == "frank":
        return FrankCopula(theta=float(p[0]))
    if nm == "gumbel":
        return GumbelCopula(theta=float(p[0]))
    raise ValueError(f"Copula '{name}' non supportée en fallback statsmodels.")

def _call_cdf(cop, u: float, v: float) -> float:
    """
    Essaie d'évaluer C(u,v) pour différents backends (statsmodels / CopulaFurtif / autres).
    """
    # Clamp in (0,1)
    u = float(np.clip(u, 1e-10, 1 - 1e-10))
    v = float(np.clip(v, 1e-10, 1 - 1e-10))

    xy = np.array([[u, v]], dtype=float)
    xyp = np.array([u, v], dtype=float)
    uv_list = [u, v]

    # 1) statsmodels: .cdf(xy)
    if hasattr(cop, "cdf"):
        try:
            out = cop.cdf(xy)
            return float(np.asarray(out).ravel()[0])
        except Exception:
            pass

    # 2) méthodes possibles (CopulaFurtif / autres libs)
    cand_attrs = [
        "cdf", "CDF", "C", "Cumulative", "cumulative", "cumdist", "cum_dist",
        "cumul", "cumul_dist", "cumulative_distribution", "copula_cdf",
        "CDF2", "cdf2", "cdf2d", "cdf_2d", "Cdf", "CopulaCDF",
        "eval_cdf", "evaluate_cdf", "compute_cdf", "H", "F"
    ]

    def _try_call(fn):
        # essais de signatures courantes
        for args in (
            (u, v),
            (xy,),
            (xyp,),
            (uv_list,),
            (float(u), float(v)),
        ):
            try:
                if isinstance(args, tuple) and len(args) > 1:
                    out = fn(*args)  # type: ignore
                else:
                    out = fn(args[0])  # type: ignore
                return float(np.asarray(out).ravel()[0])
            except Exception:
                continue
        return None

    for attr in cand_attrs:
        if hasattr(cop, attr):
            fn = getattr(cop, attr)
            res = _try_call(fn)
            if res is not None:
                return float(res)

    # 3) dernier recours: objet callable
    if callable(cop):
        res = _try_call(cop)
        if res is not None:
            return float(res)

    raise RuntimeError("Impossible d'évaluer la CDF du copula (méthodes cdf/CDF/C/Cumulative).")

def copula_h_funcs(cop, u: float, v: float, eps: float = 1e-4) -> Tuple[float, float]:
    """
    h1|2 = P(U<=u | V=v), h2|1 = P(V<=v | U=u)
    - Si la copule fournit des conditionnelles built-in -> on les utilise
    - Sinon -> fallback dérivées numériques via CDF
    """
    u = float(np.clip(u, eps, 1 - eps))
    v = float(np.clip(v, eps, 1 - eps))

    # 1) Built-in conditionnelles (ton StudentCopula CopulaFurtif les a)
    if hasattr(cop, "conditional_cdf_u_given_v") and hasattr(cop, "conditional_cdf_v_given_u"):
        try:
            h12 = float(cop.conditional_cdf_u_given_v(u, v))
            h21 = float(cop.conditional_cdf_v_given_u(u, v))
            return float(np.clip(h12, 0.0, 1.0)), float(np.clip(h21, 0.0, 1.0))
        except Exception:
            # si une copule expose la méthode mais plante, on retombe sur le fallback
            pass

    # 2) Fallback: dérivées numériques via la CDF
    dv = min(eps, v - 1e-8, 1 - 1e-8 - v)
    du = min(eps, u - 1e-8, 1 - 1e-8 - u)
    dv = max(dv, 1e-6)
    du = max(du, 1e-6)

    c_up = _call_cdf(cop, u, min(v + dv, 1 - 1e-8))
    c_dn = _call_cdf(cop, u, max(v - dv, 1e-8))
    h12 = (c_up - c_dn) / (2.0 * dv)

    c_up2 = _call_cdf(cop, min(u + du, 1 - 1e-8), v)
    c_dn2 = _call_cdf(cop, max(u - du, 1e-8), v)
    h21 = (c_up2 - c_dn2) / (2.0 * du)

    return float(np.clip(h12, 0.0, 1.0)), float(np.clip(h21, 0.0, 1.0))

def ecdf_value(sorted_x: np.ndarray, x: float) -> float:
    """
    Empirical CDF (CML style): F_hat(x) = rank/(n+1).
    """
    n = len(sorted_x)
    if n == 0:
        return np.nan
    # number <= x
    k = int(np.searchsorted(sorted_x, x, side="right"))
    return float(np.clip(k / (n + 1.0), 1e-6, 1 - 1e-6))


# -----------------------------------------------------------------------------
# Backtest core
# -----------------------------------------------------------------------------
@dataclass
class BacktestParams:
    strategy: str
    interval: str
    start: str
    end: str
    ref: str
    symbols: List[str]

    # cycle config
    formation_weeks: int = 3
    trading_weeks: int = 1
    step_weeks: int = 1

    # filters
    adf_alpha: float = 0.10
    use_kss: bool = False
    kss_crit: float = -1.92
    min_obs: int = 200
    min_coverage: float = 0.90  # % non-NaN vs ref timeline (drop symbols below)
    suppress_fit_logs: bool = True  # silence CMLE/loglik boundary spam

    # selection
    rank_method: str = "kendall_prices"
    top_k: int = 2   # how many coins to pick (2 => 1 pair)

    # copula
    copula_pick: str = "best_aic"
    copula_manual: str = "gaussian"

    # trading
    entry: float = 0.10
    exit: float = 0.10
    flip_on_opposite: bool = True

    # sizing & costs
    cap_per_leg: float = 20000.0   # like the paper's per-coin cap
    initial_equity: float = 40000.0
    fee_rate: float = 0.0004       # taker futures typical

    # misc
    seed: int = 42


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



def is_copula_evaluable(name: str, params: Any) -> bool:
    """Teste si on peut calculer h-functions (donc trading) avec ce copula."""
    try:
        cop = build_copula(name, params)
        h12, h21 = copula_h_funcs(cop, 0.37, 0.61)
        return bool(np.isfinite(h12) and np.isfinite(h21))
    except Exception:
        return False

def fit_pair_copula(
    s1: pd.Series,
    s2: pd.Series,
    pick_mode: str,
    manual_name: str,
    suppress_logs: bool = True,
) -> Tuple[Optional[str], Optional[Any], pd.DataFrame, List[str]]:
    """
    Fit copulas sur pseudo-observations (CML) via Utils.fit_copulas.
    Retourne (best_name, best_params, df_fit, messages)
    """
    s1, s2 = s1.align(s2, join="inner")
    if len(s1) < 50:
        return None, None, pd.DataFrame(), ["Pas assez d'obs pour fitter le copula."]
    u, v = pseudo_obs(s1), pseudo_obs(s2)
    # fit (optionnellement en silence pour éviter le spam CMLE sur les bornes)
    import io, contextlib
    if suppress_logs:
        _buf = io.StringIO()
        with contextlib.redirect_stdout(_buf), contextlib.redirect_stderr(_buf):
            df_fit, msgs = fit_copulas(u, v)
        _fit_log = _buf.getvalue().strip()
        # on ne garde le log que si aucune sortie
        if (not msgs) and _fit_log:
            msgs = [_fit_log]
    else:
        df_fit, msgs = fit_copulas(u, v)
    if df_fit is None or df_fit.empty:
        return None, None, pd.DataFrame(), (msgs or ["fit_copulas: aucun résultat."])

    # tri par AIC si possible
    if "aic" in df_fit.columns:
        df_fit = df_fit.sort_values("aic", ascending=True).reset_index(drop=True)

    # marquer les copulas évaluables (CDF/h-functions disponibles)
    if "evaluable" not in df_fit.columns:
        df_fit = df_fit.copy()
        df_fit["evaluable"] = [is_copula_evaluable(r["name"], r["params"]) for _, r in df_fit.iterrows()]

    if pick_mode == "manual":
        target = str(manual_name).lower()
        row = df_fit[df_fit["name"].astype(str).str.lower() == target]
        if row.empty:
            msgs = (msgs or []) + [f"Copula '{manual_name}' introuvable dans le fit => fallback meilleur AIC évaluable."]
        else:
            cand = row.iloc[0]
            if bool(cand.get("evaluable", True)):
                return str(cand["name"]), cand["params"], df_fit, msgs
            msgs = (msgs or []) + [f"Copula '{manual_name}' non-évaluable (CDF/h-functions indisponibles) => fallback."]

    # pick best evaluable
    df_ok = df_fit[df_fit.get("evaluable", True) == True]  # noqa: E712
    if df_ok.empty:
        return None, None, df_fit, (msgs or []) + ["Aucune copula évaluable pour le trading (CDF/h-functions indisponibles). "
                                                  "Choisis une famille supportée (Gaussian/Student/Clayton/Frank/Gumbel) "
                                                  "ou étends _call_cdf pour CopulaFurtif."]
    best = df_ok.iloc[0]
    return str(best["name"]), best["params"], df_fit, msgs


def generate_signals_reference_copula(
    cop,
    sorted_s1: np.ndarray,
    sorted_s2: np.ndarray,
    s1_val: float,
    s2_val: float,
    entry: float,
    exit: float,
) -> Tuple[int, Dict[str, float]]:
    """
    Retourne (signal, details):
      signal: +1 => long coin1 / short coin2
              -1 => short coin1 / long coin2
               0 => close/flat (no new)
    """
    u = ecdf_value(sorted_s1, s1_val)
    v = ecdf_value(sorted_s2, s2_val)
    h12, h21 = copula_h_funcs(cop, u, v)

    # Table 4 conditions
    open_short_coin1_long_coin2 = (h12 < entry) and (h21 > 1 - entry)  # => signal -1
    open_long_coin1_short_coin2 = (h12 > 1 - entry) and (h21 < entry)  # => signal +1
    close_cond = (abs(h12 - 0.5) < exit) and (abs(h21 - 0.5) < exit)

    sig = 0
    if close_cond:
        sig = 0
    elif open_long_coin1_short_coin2:
        sig = +1
    elif open_short_coin1_long_coin2:
        sig = -1

    details = dict(u=u, v=v, h12=h12, h21=h21, close=float(close_cond))
    return sig, details


def backtest_reference_copula(prices: pd.DataFrame, p: BacktestParams) -> Dict[str, Any]:
    """
    Backtest de la stratégie proposée (reference spread copula).
    """
    prices = prices.copy()
    prices = prices.loc[(prices.index >= _to_datetime(p.start)) & (prices.index < _to_datetime(p.end))]
    prices = prices.sort_index()
    prices = prices[p.symbols].dropna(how="all")

    if p.ref not in prices.columns:
        raise ValueError(f"Reference asset '{p.ref}' absent des données.")

    # cycle windows in calendar time
    start_dt = prices.index.min()
    end_dt = prices.index.max()
    if start_dt is pd.NaT or end_dt is pd.NaT:
        raise ValueError("Pas de données dans la fenêtre demandée.")

    formation_delta = pd.Timedelta(days=7 * p.formation_weeks)
    trading_delta = pd.Timedelta(days=7 * p.trading_weeks)
    step_delta = pd.Timedelta(days=7 * p.step_weeks)

    # outputs
    equity = pd.Series(index=prices.index, dtype=float)
    equity_gross = pd.Series(index=prices.index, dtype=float)
    equity.iloc[:] = np.nan
    equity_gross.iloc[:] = np.nan

    trades = []
    weekly = []

    current_equity = float(p.initial_equity)
    current_equity_gross = float(p.initial_equity)

    # iterate cycles
    t0 = start_dt
    cycle_id = 0

    while True:
        t_form_end = t0 + formation_delta
        t_trade_end = t_form_end + trading_delta
        if t_trade_end > end_dt:
            break

        df_form = prices.loc[(prices.index >= t0) & (prices.index < t_form_end)]
        df_trade = prices.loc[(prices.index >= t_form_end) & (prices.index < t_trade_end)]

        # stationarity filter on formation
        # 1) candidates: coins dispo dans ce df_form
        candidates = [s for s in p.symbols if (s != p.ref and s in df_form.columns)]

        # 2) calcule coverage / points SUR LA FORMATION WINDOW uniquement
        ref_ok = df_form[p.ref].notna()
        form_slice = df_form.loc[ref_ok, candidates]

        coverage = form_slice.notna().mean()  # % de points non-NaN
        points = form_slice.notna().sum()  # nb de points non-NaN

        # 3) garde seulement ceux qui passent les seuils
        eligible = [
            s for s in candidates
            if float(coverage.get(s, 0.0)) >= float(p.min_coverage)
               and int(points.get(s, 0)) >= int(p.min_obs)
        ]

        # 4) stationarity test uniquement sur eligible
        summary, spreads, betas = select_stationary_spreads(
            prices=df_form,
            ref=p.ref,
            candidates=eligible,
            adf_alpha=p.adf_alpha,
            use_kss=p.use_kss,
            kss_crit=p.kss_crit,
            min_obs=p.min_obs,
        )
        accepted = summary[summary["accepted"] == True]["coin"].tolist() if not summary.empty else []
        if len(accepted) < 2 or df_trade.empty:
            weekly.append(dict(
                cycle=cycle_id,
                formation_start=str(t0), formation_end=str(t_form_end),
                trade_start=str(t_form_end), trade_end=str(t_trade_end),
                status="SKIP (not enough stationary spreads)",
                selected_pair=None,
                copula=None,
            ))
            # write equity (flat) over trading window
            for ts in df_trade.index:
                equity.loc[ts] = current_equity
                equity_gross.loc[ts] = current_equity_gross
            t0 = t0 + step_delta
            cycle_id += 1
            continue

        # rank
        rank_df = rank_coins(df_form, spreads, p.ref, accepted, p.rank_method)
        top = rank_df["coin"].tolist()[:max(2, p.top_k)] if not rank_df.empty else accepted[:max(2, p.top_k)]
        if len(top) < 2:
            weekly.append(dict(
                cycle=cycle_id, formation_start=str(t0), formation_end=str(t_form_end),
                trade_start=str(t_form_end), trade_end=str(t_trade_end),
                status="SKIP (ranking failed)",
                selected_pair=None, copula=None,
            ))
            for ts in df_trade.index:
                equity.loc[ts] = current_equity
                equity_gross.loc[ts] = current_equity_gross
            t0 = t0 + step_delta
            cycle_id += 1
            continue

        # only 1 pair (top2) for now
        coin1, coin2 = top[0], top[1]
        s1_form = spreads[coin1]
        s2_form = spreads[coin2]
        beta1 = betas.get(coin1, np.nan)
        beta2 = betas.get(coin2, np.nan)

        best_name, best_params, df_fit, msgs = fit_pair_copula(
            s1_form, s2_form, p.copula_pick, p.copula_manual, suppress_logs=p.suppress_fit_logs
        )
        if best_name is None:
            weekly.append(dict(
                cycle=cycle_id, formation_start=str(t0), formation_end=str(t_form_end),
                trade_start=str(t_form_end), trade_end=str(t_trade_end),
                status="SKIP (copula fit failed)",
                selected_pair=f"{coin1}-{coin2}", copula=None,
                messages="; ".join(msgs),
            ))
            for ts in df_trade.index:
                equity.loc[ts] = current_equity
                equity_gross.loc[ts] = current_equity_gross
            t0 = t0 + step_delta
            cycle_id += 1
            continue

        cop = build_copula(best_name, best_params)

        # prepare ECDF on formation spreads
        s1_sorted = np.sort(s1_form.dropna().values.astype(float))
        s2_sorted = np.sort(s2_form.dropna().values.astype(float))

        # quantities fixed for the week using opening prices (paper: <= 20k USDT each coin)
        first_bar = df_trade[[coin1, coin2]].dropna().head(1)
        if first_bar.empty:
            weekly.append(dict(
                cycle=cycle_id, formation_start=str(t0), formation_end=str(t_form_end),
                trade_start=str(t_form_end), trade_end=str(t_trade_end),
                status="SKIP (no trade prices)",
                selected_pair=f"{coin1}-{coin2}", copula=best_name
            ))
            for ts in df_trade.index:
                equity.loc[ts] = current_equity
                equity_gross.loc[ts] = current_equity_gross
            t0 = t0 + step_delta
            cycle_id += 1
            continue

        p1_open = float(first_bar.iloc[0][coin1])
        p2_open = float(first_bar.iloc[0][coin2])
        q1 = (p.cap_per_leg / p1_open) if (np.isfinite(p1_open) and p1_open > 0) else 0.0
        q2 = (p.cap_per_leg / p2_open) if (np.isfinite(p2_open) and p2_open > 0) else 0.0

        pos = 0  # +1 long coin1 short coin2 ; -1 short coin1 long coin2 ; 0 flat
        entry_ts = None
        entry_p1 = entry_p2 = None
        entry_fee = 0.0
        gross_pnl = 0.0
        net_pnl = 0.0
        fees_paid = 0.0

        prev_ts = None
        prev_p1 = prev_p2 = None

        # run through trading window bars (signals at t, execution at t+1 -> apply pending orders)
        pending_sig: int = 0
        pending_close: bool = False

        def _fee(notional: float) -> float:
            return float(p.fee_rate) * float(abs(notional))

        # run through trading window bars (only where ref/coin1/coin2 prices exist)
        trade_bars = df_trade[[p.ref, coin1, coin2]].dropna()
        for ts, row in trade_bars.iterrows():
            pref = float(row[p.ref])
            p1 = float(row[coin1])
            p2 = float(row[coin2])

            # 1) mark-to-market PnL since prev bar (position held during prev->ts)
            if prev_ts is not None and pos != 0:
                dp1 = p1 - float(prev_p1)  # type: ignore
                dp2 = p2 - float(prev_p2)  # type: ignore
                if pos == +1:
                    step_pnl = q1 * dp1 - q2 * dp2
                else:
                    step_pnl = -q1 * dp1 + q2 * dp2
                gross_pnl += step_pnl
                net_pnl += step_pnl

            # 2) execute pending orders from previous bar at *current* prices
            notional1 = q1 * p1
            notional2 = q2 * p2
            trade_notional = abs(notional1) + abs(notional2)

            # pending close first
            if pos != 0 and pending_close:
                fee = _fee(trade_notional)
                fees_paid += fee
                net_pnl -= fee
                trades.append(dict(
                    cycle=cycle_id,
                    pair=f"{coin1}-{coin2}",
                    copula=best_name,
                    entry_time=str(entry_ts),
                    exit_time=str(ts),
                    direction=("LONG1_SHORT2" if pos == +1 else "SHORT1_LONG2"),
                    q1=q1, q2=q2,
                    entry_p1=entry_p1, entry_p2=entry_p2,
                    exit_p1=p1, exit_p2=p2,
                    gross_pnl=gross_pnl,
                    net_pnl=net_pnl,
                    fees=fees_paid,
                    bars=int(
                        (pd.to_datetime(ts) - pd.to_datetime(entry_ts)).total_seconds() // 60) if entry_ts else np.nan,
                    forced_week_end=False,
                ))
                # realize PnL at execution time
                current_equity = float(current_equity + net_pnl)
                current_equity_gross = float(current_equity_gross + gross_pnl)
                pos = 0
                entry_ts = None
                entry_p1 = entry_p2 = None
                gross_pnl = 0.0
                net_pnl = 0.0
                fees_paid = 0.0

            pending_close = False  # consumed

            # pending open/flip
            if pending_sig != 0:
                if pos == 0:
                    # open at current prices
                    fee = _fee(trade_notional)
                    fees_paid += fee
                    net_pnl -= fee
                    pos = int(pending_sig)
                    entry_ts = ts
                    entry_p1, entry_p2 = p1, p2
                else:
                    # flip at current prices (close then open) if allowed
                    if p.flip_on_opposite and int(pending_sig) != int(pos):
                        fee_close = _fee(trade_notional)
                        fees_paid += fee_close
                        net_pnl -= fee_close
                        trades.append(dict(
                            cycle=cycle_id,
                            pair=f"{coin1}-{coin2}",
                            copula=best_name,
                            entry_time=str(entry_ts),
                            exit_time=str(ts),
                            direction=("LONG1_SHORT2" if pos == +1 else "SHORT1_LONG2"),
                            q1=q1, q2=q2,
                            entry_p1=entry_p1, entry_p2=entry_p2,
                            exit_p1=p1, exit_p2=p2,
                            gross_pnl=gross_pnl,
                            net_pnl=net_pnl,
                            fees=fees_paid,
                            bars=int((pd.to_datetime(ts) - pd.to_datetime(
                                entry_ts)).total_seconds() // 60) if entry_ts else np.nan,
                            forced_week_end=False,
                        ))
                        # realize PnL at flip-close
                        current_equity = float(current_equity + net_pnl)
                        current_equity_gross = float(current_equity_gross + gross_pnl)

                        # reset and open new
                        pos = 0
                        entry_ts = None
                        entry_p1 = entry_p2 = None
                        gross_pnl = 0.0
                        net_pnl = 0.0
                        fees_paid = 0.0

                        fee_open = _fee(trade_notional)
                        fees_paid += fee_open
                        net_pnl -= fee_open
                        pos = int(pending_sig)
                        entry_ts = ts
                        entry_p1, entry_p2 = p1, p2

            pending_sig = 0  # consumed

            # 3) compute current spreads (using betas from formation) + signal for next bar
            s1_val = float(pref - beta1 * p1) if np.isfinite(beta1) else np.nan
            s2_val = float(pref - beta2 * p2) if np.isfinite(beta2) else np.nan
            if not np.isfinite(s1_val) or not np.isfinite(s2_val):
                sig = 0
                close_now = False
            else:
                sig, det = generate_signals_reference_copula(
                    cop=cop, sorted_s1=s1_sorted, sorted_s2=s2_sorted,
                    s1_val=s1_val, s2_val=s2_val,
                    entry=p.entry, exit=p.exit
                )
                close_now = bool(det.get("close", 0.0) > 0.5)

            # schedule execution at next bar
            if pos != 0 and close_now:
                pending_close = True
            elif sig != 0:
                pending_sig = int(sig)

            # 4) update equity marks (equity series is global portfolio)
            equity.loc[ts] = current_equity + net_pnl  # net_pnl inclut fees/unrealized
            equity_gross.loc[ts] = current_equity_gross + gross_pnl

            prev_ts = ts
            prev_p1, prev_p2 = p1, p2

        last_bar = df_trade[[coin1, coin2]].dropna().tail(1)
        if not last_bar.empty:
            ts_end = last_bar.index[0]
            p1_end = float(last_bar.iloc[0][coin1])
            p2_end = float(last_bar.iloc[0][coin2])

            if pos != 0:
                notional_end = abs(q1 * p1_end) + abs(q2 * p2_end)
                fee = float(p.fee_rate) * notional_end
                fees_paid += fee
                net_pnl -= fee
                trades.append(dict(
                    cycle=cycle_id,
                    pair=f"{coin1}-{coin2}",
                    copula=best_name,
                    entry_time=str(entry_ts),
                    exit_time=str(ts_end),
                    direction=("LONG1_SHORT2" if pos == +1 else "SHORT1_LONG2"),
                    q1=q1, q2=q2,
                    entry_p1=entry_p1, entry_p2=entry_p2,
                    exit_p1=p1_end, exit_p2=p2_end,
                    gross_pnl=gross_pnl,
                    net_pnl=net_pnl,
                    fees=fees_paid,
                    bars=int((pd.to_datetime(ts_end) - pd.to_datetime(entry_ts)).total_seconds() // 60) if entry_ts else np.nan,
                    forced_week_end=True
                ))
                pos = 0

            # update equity at week end and roll portfolio equity forward
            current_equity = float(current_equity + net_pnl)
            current_equity_gross = float(current_equity_gross + gross_pnl)

            # fill equity series for bars not set (if any)
            for ts in df_trade.index:
                if pd.isna(equity.loc[ts]):
                    equity.loc[ts] = current_equity
                if pd.isna(equity_gross.loc[ts]):
                    equity_gross.loc[ts] = current_equity_gross

        weekly.append(dict(
            cycle=cycle_id,
            formation_start=str(t0), formation_end=str(t_form_end),
            trade_start=str(t_form_end), trade_end=str(t_trade_end),
            status="OK",
            selected_pair=f"{coin1}-{coin2}",
            copula=best_name,
            beta1=float(beta1) if np.isfinite(beta1) else np.nan,
            beta2=float(beta2) if np.isfinite(beta2) else np.nan,
            q1=q1, q2=q2,
            top_list=",".join(top),
            fit_messages="; ".join(msgs) if msgs else "",
        ))

        t0 = t0 + step_delta
        cycle_id += 1

    equity = equity.ffill().dropna()
    equity_gross = equity_gross.ffill().dropna()

    trades_df = pd.DataFrame(trades)
    weekly_df = pd.DataFrame(weekly)

    # perf metrics
    metrics = performance_metrics(equity, p.interval)
    metrics_g = performance_metrics(equity_gross, p.interval)

    # monthly returns from net equity
    monthly = None
    if len(equity) > 10:
        eq_m = equity.resample("M").last()
        monthly = eq_m.pct_change().dropna()
    else:
        monthly = pd.Series(dtype=float)

    # copula frequency
    cop_freq = None
    if not weekly_df.empty and "copula" in weekly_df.columns:
        cop_freq = weekly_df["copula"].value_counts().reset_index()
        cop_freq.columns = ["copula", "count"]
    else:
        cop_freq = pd.DataFrame(columns=["copula", "count"])

    return dict(
        equity=equity,
        equity_gross=equity_gross,
        trades=trades_df,
        weekly=weekly_df,
        metrics=metrics,
        metrics_gross=metrics_g,
        monthly_returns=monthly,
        copula_freq=cop_freq,
        params=p,
    )


# -----------------------------------------------------------------------------
# Plotly builders
# -----------------------------------------------------------------------------
def fig_empty(title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template="plotly_dark", autosize=False, height=320, title=title,
                      plot_bgcolor="#181818", paper_bgcolor="#181818")
    return fig

def fig_equity(equity: pd.Series, equity_gross: Optional[pd.Series] = None) -> go.Figure:
    equity = pd.Series(equity).replace([np.inf, -np.inf], np.nan).dropna()
    if equity_gross is not None:
        equity_gross = pd.Series(equity_gross).replace([np.inf, -np.inf], np.nan).dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, name="Equity (net)"))
    if equity_gross is not None and len(equity_gross) > 0:
        fig.add_trace(go.Scatter(x=equity_gross.index, y=equity_gross.values, name="Equity (gross)", line=dict(dash="dot")))
    fig.update_layout(template="plotly_dark", autosize=False, height=360, title="Equity curve",
                      plot_bgcolor="#181818", paper_bgcolor="#181818",
                      xaxis_title="Date", yaxis_title="USDT", margin=dict(l=50,r=15,t=55,b=40))
    return fig

def fig_drawdown(equity: pd.Series) -> go.Figure:
    eq = pd.Series(equity).replace([np.inf, -np.inf], np.nan).dropna()
    if len(eq) < 3:
        return fig_empty("Drawdown")
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name="Drawdown"))
    fig.update_layout(template="plotly_dark", autosize=False, height=260, title="Drawdown",
                      plot_bgcolor="#181818", paper_bgcolor="#181818",
                      xaxis_title="Date", yaxis_title="Drawdown", margin=dict(l=50,r=15,t=55,b=40))
    return fig

def fig_monthly_heatmap(monthly_returns: pd.Series) -> go.Figure:
    if monthly_returns is None:
        return fig_empty("Monthly returns heatmap")
    monthly_returns = pd.Series(monthly_returns).replace([np.inf, -np.inf], np.nan).dropna()
    if len(monthly_returns) == 0:
        return fig_empty("Monthly returns heatmap")

    mr = monthly_returns.copy()
    df = pd.DataFrame({"ret": mr})
    df["year"] = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot_table(index="year", columns="month", values="ret", aggfunc="sum")

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[str(m) for m in pivot.columns],
        y=[str(y) for y in pivot.index],
        colorbar=dict(title="Return"),
    ))
    fig.update_layout(template="plotly_dark", autosize=False, height=340, title="Monthly returns (net)",
                      plot_bgcolor="#181818", paper_bgcolor="#181818",
                      xaxis_title="Month", yaxis_title="Year", margin=dict(l=50,r=15,t=55,b=40))
    return fig

def fig_copula_freq(cop_freq: pd.DataFrame) -> go.Figure:
    if cop_freq is None or cop_freq.empty:
        return fig_empty("Copula frequency")
    fig = go.Figure(data=go.Bar(
        x=cop_freq["copula"].astype(str),
        y=cop_freq["count"].astype(int),
        name="count"
    ))
    fig.update_layout(template="plotly_dark", autosize=False, height=320, title="Copula chosen per week",
                      plot_bgcolor="#181818", paper_bgcolor="#181818",
                      xaxis_title="Copula", yaxis_title="# weeks", margin=dict(l=50,r=15,t=55,b=40))
    return fig


# -----------------------------------------------------------------------------
# Dash App
# -----------------------------------------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.index_string = """
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
""" + DROPDOWN_DARK_CSS + """
    </style>
  </head>
  <body>
    {%app_entry%}
    <footer>
      {%config%}
      {%scripts%}
      {%renderer%}
    </footer>
  </body>
</html>
"""
server = app.server

ALL_COPULAS = _get_all_copula_values()

def _opts(vals: List[str]) -> List[Dict[str, str]]:
    return [{"label": v, "value": v} for v in vals]

app.layout = dbc.Container(fluid=True, children=[
    html.H2("Copula Trading Bot — Backtest Dashboard", style={"textAlign":"center", "marginTop":"12px"}),
    html.Div(
        "Backtest sur données réelles (CSV / yfinance / Binance) avec coûts de transaction, "
        "sélection dynamique des paires (formation 3 semaines / trading 1 semaine par défaut).",
        style={"textAlign":"center", "color":"#AAAAAA", "marginBottom":"12px"}
    ),

    dbc.Row([
        dbc.Col(width=3, children=[
            dbc.Card([
                dbc.CardHeader("Paramètres"),
                dbc.CardBody([
                    html.Label("Source de données"),
                    dcc.Dropdown(
                        id="data-source",
                        options=[
                            {"label":"CSV (DATA_PATH)", "value":"csv"},
                            {"label":"yfinance (spot, tickers -USD)", "value":"yfinance"},
                            {"label":"Binance via ccxt (USDT futures)", "value":"binance"},
                        ],
                        value="csv",
                        clearable=False,
                    ),
                    html.Div(id="data-source-warning", style={"marginTop":"6px", "color":"#FFCC00"}),

                    html.Hr(),

                    html.Label("Strategy"),
                    dcc.Dropdown(
                        id="strategy",
                        options=[{"label": lbl, "value": v} for v, lbl in STRATEGIES],
                        value="reference_copula",
                        clearable=False,
                    ),

                    html.Label("Interval"),
                    dcc.Dropdown(
                        id="interval",
                        options=[{"label": a, "value": b} for a, b in INTERVALS],
                        value="1h",
                        clearable=False,
                    ),

                    html.Label("Start / End"),
                    dcc.DatePickerRange(
                        id="date-range",
                        start_date=(pd.Timestamp.today() - pd.Timedelta(days=365*2)).date(),
                        end_date=pd.Timestamp.today().date(),
                        display_format="YYYY-MM-DD",
                    ),

                    html.Hr(),

                    html.Label("Universe (symbols)"),
                    dcc.Dropdown(
                        id="symbols",
                        options=_opts(sorted(set(DEFAULT_USDT_SYMBOLS_EXT))),
                        value=DEFAULT_USDT_SYMBOLS,
                        multi=True,
                    ),
                    html.Div("Tip: en mode Binance/ccxt, tu peux mettre beaucoup plus de symbols.", style={"fontSize":"12px", "color":"#999"}),

                    html.Label("Reference asset"),
                    dcc.Dropdown(id="ref-asset", options=_opts(sorted(set(DEFAULT_USDT_SYMBOLS_EXT))), value=DEFAULT_REFERENCE, clearable=False),

                    html.Hr(),

                    html.Label("Formation / Trading / Step (weeks)"),
                    dbc.Row([
                        dbc.Col(dbc.Input(id="formation-weeks", type="number", value=3, min=1, step=1), width=4),
                        dbc.Col(dbc.Input(id="trading-weeks", type="number", value=1, min=1, step=1), width=4),
                        dbc.Col(dbc.Input(id="step-weeks", type="number", value=1, min=1, step=1), width=4),
                    ], style={"marginBottom":"8px"}),

                    html.Label("ADF significance level (alpha)"),
                    dbc.Input(id="adf-alpha", type="number", value=0.10, min=0.01, max=0.25, step=0.01),

                    dbc.Checklist(
                        options=[{"label":"Apply KSS filter (t-stat < -1.92)", "value":"kss"}],
                        value=[],
                        id="use-kss",
                        style={"marginTop":"6px"}
                    ),

                    html.Label("Min obs in formation (N_obs)"),
                    dbc.Input(id="min-obs", type="number", value=200, min=50, step=10),

                    html.Label("Min coverage per symbol (ratio non-NaN vs ref timeline)"),
                    dbc.Input(id="min-coverage", type="number", value=0.90, min=0.50, max=1.00, step=0.01),

                    html.Hr(),

                    html.Label("Ranking method (pair selection)"),
                    dcc.Dropdown(
                        id="rank-method",
                        options=[{"label": lbl, "value": v} for v, lbl in RANK_METHODS],
                        value="kendall_prices",
                        clearable=False,
                    ),

                    html.Label("Top-K coins (2 => 1 pair/week)"),
                    dbc.Input(id="top-k", type="number", value=2, min=2, max=50, step=1),

                    html.Hr(),

                    html.Label("Copula selection"),
                    dcc.Dropdown(
                        id="copula-pick",
                        options=[{"label": lbl, "value": v} for v, lbl in COPULA_PICK],
                        value="best_aic",
                        clearable=False,
                    ),
                    html.Label("Manual copula (if manual)"),
                    dcc.Dropdown(id="copula-manual", options=_opts(ALL_COPULAS), value="gaussian", clearable=False),

                    dbc.Checklist(
                        options=[{"label":"Suppress CMLE/fit logs (clean console)", "value":"suppress"}],
                        value=["suppress"],
                        id="suppress-logs",
                        style={"marginTop":"8px"}
                    ),

                    html.Hr(),

                    html.Label("Entry / Exit thresholds"),
                    dbc.Row([
                        dbc.Col(dbc.Input(id="entry", type="number", value=0.10, min=0.01, max=0.49, step=0.01), width=6),
                        dbc.Col(dbc.Input(id="exit", type="number", value=0.10, min=0.01, max=0.49, step=0.01), width=6),
                    ], style={"marginBottom":"8px"}),

                    dbc.Checklist(
                        options=[{"label":"Flip on opposite signal", "value":"flip"}],
                        value=["flip"],
                        id="flip",
                    ),

                    html.Hr(),

                    html.Label("Capital & fees"),
                    html.Div("Cap per leg (USDT)"),
                    dbc.Input(id="cap-per-leg", type="number", value=20000, min=100, step=100),
                    html.Div("Initial equity (USDT)"),
                    dbc.Input(id="initial-equity", type="number", value=40000, min=100, step=100),
                    html.Div("Fee rate (taker ~ 0.0004)"),
                    dbc.Input(id="fee-rate", type="number", value=0.0004, min=0.0, max=0.01, step=0.0001),

                    html.Hr(),

                    dbc.Button("Run backtest", id="run-btn", color="success", className="w-100"),
                    html.Div(id="run-status", style={"marginTop":"10px", "color":"#00E676"}),
                ])
            ], color="dark", inverse=True),
        ]),

        dbc.Col(width=9, children=[
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.Div("Total net return", style={"color":"#AAAAAA"}),
                    html.H3(id="m-total", children="—")
                ]), color="dark", inverse=True), width=2),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.Div("Annual net return", style={"color":"#AAAAAA"}),
                    html.H3(id="m-ann", children="—")
                ]), color="dark", inverse=True), width=2),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.Div("Sharpe", style={"color":"#AAAAAA"}),
                    html.H3(id="m-sharpe", children="—")
                ]), color="dark", inverse=True), width=2),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.Div("Max drawdown", style={"color":"#AAAAAA"}),
                    html.H3(id="m-mdd", children="—")
                ]), color="dark", inverse=True), width=2),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.Div("# Trades", style={"color":"#AAAAAA"}),
                    html.H3(id="m-trades", children="—")
                ]), color="dark", inverse=True), width=2),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.Div("Fees (total)", style={"color":"#AAAAAA"}),
                    html.H3(id="m-fees", children="—")
                ]), color="dark", inverse=True), width=2),
            ], style={"marginBottom":"10px"}),

            dcc.Tabs(id="tabs", value="tab-equity", children=[
                dcc.Tab(label="Equity & Risk", value="tab-equity"),
                dcc.Tab(label="Weekly selection", value="tab-weekly"),
                dcc.Tab(label="Trades", value="tab-trades"),
                dcc.Tab(label="Copula stats", value="tab-copulas"),
            ]),

            html.Div(id="tab-content", style={"marginTop":"10px"}),

            # stores
            dcc.Store(id="store-results"),
        ])
    ]),
], style={"maxWidth":"1600px"})


def _serialize_results(res: Dict[str, Any]) -> Dict[str, Any]:
    def ser_series(s: pd.Series) -> Dict[str, Any]:
        return {"index": s.index.astype(str).tolist(), "values": s.values.tolist()}
    out = {
        "equity": ser_series(res["equity"]),
        "equity_gross": ser_series(res["equity_gross"]),
        "metrics": res["metrics"],
        "metrics_gross": res["metrics_gross"],
        "trades": res["trades"].to_dict("records") if isinstance(res["trades"], pd.DataFrame) else [],
        "weekly": res["weekly"].to_dict("records") if isinstance(res["weekly"], pd.DataFrame) else [],
        "copula_freq": res["copula_freq"].to_dict("records") if isinstance(res["copula_freq"], pd.DataFrame) else [],
        "monthly_returns": ser_series(res["monthly_returns"]) if isinstance(res["monthly_returns"], pd.Series) else {"index": [], "values": []},
        "params": res["params"].__dict__,
    }
    return out

def _deserialize_series(obj: Dict[str, Any]) -> pd.Series:
    idx = pd.to_datetime(obj.get("index", []))
    vals = obj.get("values", [])
    s = pd.Series(vals, index=idx)
    s = s[~s.index.isna()]
    return s.sort_index()


@app.callback(
    Output("data-source-warning", "children"),
    Input("data-source", "value"),
)
def data_source_warning(src: str):
    msgs = []
    if src == "csv":
        if not DATA_PATH:
            msgs.append("⚠️ DATA_PATH non trouvé. Tu peux quand même utiliser yfinance/ccxt, ou renseigner DATA_PATH dans DataAnalysis/config.py.")
    if src == "yfinance" and not HAS_YFINANCE:
        msgs.append("⚠️ yfinance non installé: pip install yfinance")
    if src == "binance" and not HAS_CCXT:
        msgs.append("⚠️ ccxt non installé: pip install ccxt")
    if not HAS_COPULAFURTIF and not HAS_SM_COPULA:
        msgs.append("⚠️ Aucun backend copula: installe CopulaFurtif ou statsmodels>=0.13")
    return html.Div(msgs) if msgs else ""


@app.callback(
    Output("store-results", "data"),
    Output("run-status", "children"),
    Input("run-btn", "n_clicks"),
    State("data-source", "value"),
    State("strategy", "value"),
    State("interval", "value"),
    State("date-range", "start_date"),
    State("date-range", "end_date"),
    State("symbols", "value"),
    State("ref-asset", "value"),
    State("formation-weeks", "value"),
    State("trading-weeks", "value"),
    State("step-weeks", "value"),
    State("adf-alpha", "value"),
    State("use-kss", "value"),
    State("min-obs", "value"),
    State("min-coverage", "value"),
    State("rank-method", "value"),
    State("top-k", "value"),
    State("copula-pick", "value"),
    State("copula-manual", "value"),
    State("suppress-logs", "value"),
    State("entry", "value"),
    State("exit", "value"),
    State("flip", "value"),
    State("cap-per-leg", "value"),
    State("initial-equity", "value"),
    State("fee-rate", "value"),
    prevent_initial_call=True
)
def run_backtest(
    n_clicks,
    data_source,
    strategy,
    interval,
    start_date,
    end_date,
    symbols,
    ref_asset,
    formation_weeks,
    trading_weeks,
    step_weeks,
    adf_alpha,
    use_kss,
    min_obs,
    min_coverage,
    rank_method,
    top_k,
    copula_pick,
    copula_manual,
    suppress_logs,
    entry,
    exit,
    flip,
    cap_per_leg,
    initial_equity,
    fee_rate,
):
    if not symbols or ref_asset not in symbols:
        # auto add ref into symbols
        symbols = (symbols or []) + [ref_asset]

    # fetch data
    fetch_errors = {}
    try:
        if data_source == "yfinance":
            prices = fetch_prices_yfinance(symbols, str(start_date), str(end_date), interval)
            fetch_errors = {}
        elif data_source == "binance":
            prices, fetch_errors = fetch_prices_binance_ccxt(symbols, str(start_date), str(end_date), interval)
        else:
            if not DATA_PATH:
                raise RuntimeError("DATA_PATH manquant pour mode CSV.")
            prices = load_prices_csv(DATA_PATH)
            fetch_errors = {}
            prices = prices.loc[(prices.index >= pd.to_datetime(start_date)) & (prices.index < pd.to_datetime(end_date))]
            # resample if needed (best effort)
            if interval != "1h":
                rule_map = {"5m":"5T","15m":"15T","1h":"1H","4h":"4H","1d":"1D"}
                rule = rule_map.get(interval)
                if rule:
                    prices = prices.resample(rule).last()
    except Exception as e:
        return dash.no_update, f"❌ Data load error: {e}"

    # keep only requested
    prices = prices[[c for c in symbols if c in prices.columns]].dropna(how="all")
    if prices.empty:
        return dash.no_update, "❌ No price data returned for selected symbols/time range."

    # data clean / drop problematic symbols (late listing, holes, etc.)
    prices = clean_prices_basic(prices)  # fonction à ajouter (voir ci-dessous)
    drop_reasons = {}

    # Keep only requested symbols that survived cleaning
    available_syms = [s for s in (symbols or []) if s in prices.columns]
    if str(ref_asset) not in available_syms and str(ref_asset) in prices.columns:
        available_syms = available_syms + [str(ref_asset)]
    symbols = available_syms

    # params
    p = BacktestParams(
        strategy=strategy,
        interval=interval,
        start=str(start_date),
        end=str(end_date),
        ref=str(ref_asset),
        symbols=list(dict.fromkeys(symbols)),  # dedup keep order
        formation_weeks=int(formation_weeks or 3),
        trading_weeks=int(trading_weeks or 1),
        step_weeks=int(step_weeks or 1),
        adf_alpha=float(adf_alpha or 0.10),
        use_kss=("kss" in (use_kss or [])),
        min_obs=int(min_obs or 200),
        min_coverage=float(min_coverage or 0.90),
        suppress_fit_logs=("suppress" in (suppress_logs or ["suppress"])),
        rank_method=str(rank_method),
        top_k=int(top_k or 2),
        copula_pick=str(copula_pick),
        copula_manual=str(copula_manual),
        entry=float(entry or 0.10),
        exit=float(exit or 0.10),
        flip_on_opposite=("flip" in (flip or [])),
        cap_per_leg=float(cap_per_leg or 20000.0),
        initial_equity=float(initial_equity or 40000.0),
        fee_rate=float(fee_rate or 0.0004),
    )

    # NOTE: pour l'instant, seul le backtest de la stratégie proposée est implémenté.
    if p.strategy != "reference_copula":
        return dash.no_update, "⚠️ Pour l'instant, ce dashboard backteste uniquement la stratégie 'reference_copula'."

    try:
        res = backtest_reference_copula(prices, p)
    except Exception as e:
        return dash.no_update, f"❌ Backtest error: {e}"

    payload = _serialize_results(res)
    warns = []
    if fetch_errors:
        # keep it short
        bad = ", ".join(list(fetch_errors.keys())[:8])
        extra = "" if len(fetch_errors) <= 8 else f" (+{len(fetch_errors)-8})"
        warns.append(f"⚠️ fetch issues: {bad}{extra}")
    if drop_reasons:
        bad = [k for k in drop_reasons.keys() if not k.startswith("__")]
        if bad:
            shown = ", ".join(bad[:8])
            extra = "" if len(bad) <= 8 else f" (+{len(bad)-8})"
            warns.append(f"⚠️ dropped (coverage/holes): {shown}{extra}")
    status = "✅ Backtest terminé."
    if warns:
        status = status + "  " + "  |  ".join(warns)
    return payload, status


@app.callback(
    Output("m-total", "children"),
    Output("m-ann", "children"),
    Output("m-sharpe", "children"),
    Output("m-mdd", "children"),
    Output("m-trades", "children"),
    Output("m-fees", "children"),
    Input("store-results", "data"),
)
def update_metrics(store):
    if not store:
        return ("—", "—", "—", "—", "—", "—")
    m = store.get("metrics", {})
    tot = m.get("total_return", np.nan)
    ann = m.get("annual_return", np.nan)
    sh = m.get("sharpe", np.nan)
    mdd = m.get("max_drawdown", np.nan)

    trades = store.get("trades", [])
    fees = 0.0
    if trades:
        try:
            fees = float(np.nansum([t.get("fees", 0.0) for t in trades]))
        except Exception:
            fees = np.nan

    def f(x, pct=False):
        if x is None or not np.isfinite(x):
            return "—"
        return f"{_safe_pct(x):.1f}%" if pct else f"{x:.2f}"

    return (
        f(tot, pct=True),
        f(ann, pct=True),
        f(sh, pct=False),
        f(mdd, pct=True),
        str(len(trades) if trades else 0),
        (f"{fees:,.0f}" if np.isfinite(fees) else "—"),
    )


@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("store-results", "data"),
)
def render_tab(tab, store):
    if not store:
        return dbc.Alert("Clique sur 'Run backtest' pour lancer la simulation.", color="info")

    equity = _deserialize_series(store["equity"])
    equity_g = _deserialize_series(store["equity_gross"])
    trades = pd.DataFrame(store.get("trades", []))
    weekly = pd.DataFrame(store.get("weekly", []))
    copfreq = pd.DataFrame(store.get("copula_freq", []))
    monthly = _deserialize_series(store.get("monthly_returns", {"index": [], "values": []}))

    if tab == "tab-equity":
        return dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_equity(equity, equity_g), style={"height":"420px"}, config={"responsive": False}), width=8),
            dbc.Col([
                dcc.Graph(figure=fig_drawdown(equity), style={"height":"300px"}, config={"responsive": False}),
                dcc.Graph(figure=fig_monthly_heatmap(monthly), style={"height":"360px"}, config={"responsive": False}),
            ], width=4),
        ])

    if tab == "tab-weekly":
        cols = [{"name": c, "id": c} for c in weekly.columns] if not weekly.empty else []
        return dbc.Card(dbc.CardBody([
            html.H4("Weekly selection (formation/trading cycles)"),
            dash_table.DataTable(
                data=weekly.to_dict("records") if not weekly.empty else [],
                columns=cols,
                page_size=12,
                style_table={"overflowX":"auto"},
                style_cell={"backgroundColor":"#181818","color":"white","border":"1px solid #333","fontSize":"12px"},
                style_header={"backgroundColor":"#101010","fontWeight":"bold"},
            )
        ]), color="dark", inverse=True)

    if tab == "tab-trades":
        cols = [{"name": c, "id": c} for c in trades.columns] if not trades.empty else []
        return dbc.Card(dbc.CardBody([
            html.H4("Trades"),
            dash_table.DataTable(
                data=trades.to_dict("records") if not trades.empty else [],
                columns=cols,
                page_size=12,
                sort_action="native",
                filter_action="native",
                style_table={"overflowX":"auto"},
                style_cell={"backgroundColor":"#181818","color":"white","border":"1px solid #333","fontSize":"12px"},
                style_header={"backgroundColor":"#101010","fontWeight":"bold"},
            )
        ]), color="dark", inverse=True)

    if tab == "tab-copulas":
        return dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_copula_freq(copfreq), style={"height":"380px"}, config={"responsive": False}), width=6),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H4("Sanity checks"),
                html.Ul([
                    html.Li(f"Copula backend: {'CopulaFurtif' if HAS_COPULAFURTIF else ('statsmodels' if HAS_SM_COPULA else 'None')}"),
                    html.Li(f"# weeks: {len(weekly)}"),
                    html.Li(f"# trades: {len(trades)}"),
                    html.Li("Rappel: positions forcées fermées à la fin de chaque trading week."),
                ])
            ]), color="dark", inverse=True), width=6),
        ])

    return dbc.Alert("Unknown tab", color="warning")


if __name__ == "__main__":
    app.run(debug=True, port=8050)
