"""
price_cache.py — Cache CSV pour les données OHLCV Binance SPOT.

Principe :
  1. Lit le CSV de cache existant pour ce symbole + intervalle
  2. Détecte les ranges manquants (avant/après le cache)
  3. Télécharge uniquement ce qui manque via l'API REST Binance (requests)
  4. Fusionne + sauvegarde
  5. Retourne la tranche [start, end] demandée

Cache stocké dans : DataAnalysis/data/cache/{interval}/{SYMBOL}.csv
Aucune dépendance hors requests + pandas.
"""

import os
import time
from typing import Optional, List, Tuple, Dict

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Chemins cache
# ---------------------------------------------------------------------------

def _cache_path(base_path: str, interval: str, symbol: str) -> str:
    d = os.path.abspath(os.path.join(base_path, "..", "cache", interval))
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{symbol}.csv")


# ---------------------------------------------------------------------------
# Lecture / écriture
# ---------------------------------------------------------------------------

def _read_cache(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if df.empty:
            return None
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df
    except Exception:
        return None


def _write_cache(path: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.to_csv(path)


def _merge(existing: Optional[pd.DataFrame], new: pd.DataFrame) -> pd.DataFrame:
    if existing is None or existing.empty:
        return new.sort_index()
    merged = pd.concat([existing, new])
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged.sort_index()


# ---------------------------------------------------------------------------
# Détection des ranges manquants
# ---------------------------------------------------------------------------

def _missing_ranges(
    cached: Optional[pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Retourne les périodes à télécharger pour couvrir [start, end]."""
    now = pd.Timestamp.now("UTC").tz_convert(None)
    end = min(end, now)

    if cached is None or cached.empty:
        return [(start, end)]

    cache_start = cached.index.min()
    cache_end   = cached.index.max()
    gap         = pd.Timedelta(minutes=30)
    ranges      = []

    if start < cache_start - gap:
        ranges.append((start, cache_start))
    if end > cache_end + gap:
        ranges.append((cache_end, end))

    return ranges


# ---------------------------------------------------------------------------
# Fetch Binance REST (sans aucune lib tierce hors requests)
# ---------------------------------------------------------------------------

def _fetch_binance(
    symbol: str,
    interval: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Télécharge les klines Binance SPOT via l'API REST directe.
    Pagine automatiquement pour couvrir tout le range.
    """
    from DataAnalysis.src.data_fetching import fetch_klines
    start_ms = int(start.timestamp() * 1000)
    end_ms   = int(end.timestamp() * 1000)
    return fetch_klines(symbol, interval, start_ms, end_ms)


# ---------------------------------------------------------------------------
# Fonction principale
# ---------------------------------------------------------------------------

def get_prices_cached(
    symbols: List[str],
    interval: str,
    start: str,
    end: str,
    base_path: str = "DataAnalysis/data/raw/",
    force_refresh: bool = False,
    **kwargs,  # absorbe les arguments legacy (source=...) sans planter
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Récupère les prix de clôture Binance SPOT avec cache CSV local.

    Args:
        symbols:       Liste ex. ['BTCUSDT', 'ETHUSDT']
        interval:      Timeframe ex. '15m', '1h', '1d'
        start:         Date début ISO ex. '2024-01-01'
        end:           Date fin   ISO ex. '2026-02-23'
        base_path:     Chemin vers DataAnalysis/data/raw/
        force_refresh: Ignore le cache et retélécharge tout

    Returns:
        (DataFrame de close prices, dict d'erreurs par symbole)
    """
    start_ts = pd.to_datetime(start)
    end_ts   = pd.to_datetime(end)
    errors: Dict[str, str] = {}
    out:    Dict[str, pd.Series] = {}

    for sym in symbols:
        path   = _cache_path(base_path, interval, sym)
        cached = None if force_refresh else _read_cache(path)
        missing = _missing_ranges(cached, start_ts, end_ts)

        new_frames = []
        for (rng_start, rng_end) in missing:
            print(f"⬇️  {sym} [{rng_start.date()} → {rng_end.date()}] …")
            try:
                df_new = _fetch_binance(sym, interval, rng_start, rng_end)
                if df_new is not None and not df_new.empty:
                    if df_new.index.tz is not None:
                        df_new.index = df_new.index.tz_localize(None)
                    new_frames.append(df_new)
                    print(f"  ✅ {sym} : {len(df_new)} bougies")
                else:
                    print(f"  ⚠️  {sym} : réponse vide")
            except Exception as e:
                err = str(e)[:120]
                print(f"  ❌ {sym} : {err}")
                errors[sym] = err

        # Fusionner et sauvegarder
        if new_frames:
            merged = _merge(cached, pd.concat(new_frames))
            _write_cache(path, merged)
            full = merged
        else:
            full = cached

        if full is None or full.empty:
            if sym not in errors:
                errors[sym] = "Aucune donnée (cache vide + fetch échoué)"
            continue

        if "close" not in full.columns:
            errors[sym] = "Colonne 'close' absente du cache"
            continue

        sliced = full.loc[
            (full.index >= start_ts) & (full.index <= end_ts), "close"
        ]
        if sliced.dropna().empty:
            errors[sym] = f"Aucune donnée dans la fenêtre {start} → {end}"
            continue

        out[sym] = sliced

    result = pd.DataFrame(out).sort_index()
    return result, errors