# dash_bot/data/sources.py
"""
Sources de données — uniquement Binance SPOT (API REST directe) et CSV locaux.

Yahoo Finance est retiré : bloqué pour les crypto intraday côté Yahoo.
ccxt est retiré : on utilise directement l'API REST Binance via requests
                  (déjà implémenté dans DataAnalysis/src/data_fetching.py).
Aucune dépendance tierce nécessaire hors requests + pandas.
"""

import warnings
from typing import List, Dict, Tuple

import pandas as pd

from DataAnalysis.config import DATA_PATH

# Flags conservés pour compatibilité avec les imports existants
HAS_YFINANCE = False   # désactivé volontairement
HAS_CCXT     = False   # désactivé volontairement — on n'en a pas besoin


# ---------------------------------------------------------------------------
# Source principale : Binance SPOT via cache CSV
# ---------------------------------------------------------------------------

def fetch_prices_cached(
    symbols: List[str],
    start: str,
    end: str,
    interval: str,
    source: str = "binance",
    force_refresh: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Récupère les prix via le cache CSV local (Binance SPOT).
    Ne télécharge que les données manquantes.

    Args:
        symbols:       Liste ex. ['BTCUSDT', 'ETHUSDT']
        start:         Date début ISO ex. '2024-01-01'
        end:           Date fin   ISO ex. '2026-02-23'
        interval:      Timeframe   ex. '15m', '1h', '1d'
        source:        'binance' uniquement (yfinance supprimé)
        force_refresh: Retélécharge tout même si le cache existe

    Returns:
        (DataFrame de close prices, dict d'erreurs par symbole)
    """
    from DataAnalysis.src.price_cache import get_prices_cached
    return get_prices_cached(
        symbols=symbols,
        interval=interval,
        start=start,
        end=end,
        source="binance",   # toujours binance, yfinance supprimé
        base_path=DATA_PATH,
        force_refresh=force_refresh,
    )


# ---------------------------------------------------------------------------
# Wrapper pour compatibilité callbacks
# ---------------------------------------------------------------------------

def fetch_prices_binance_ccxt(
    symbols: List[str],
    start: str,
    end: str,
    interval: str,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Alias vers fetch_prices_cached (Binance SPOT, sans ccxt)."""
    return fetch_prices_cached(symbols, start, end, interval, source="binance")


# ---------------------------------------------------------------------------
# CSV locaux (legacy / mode offline)
# ---------------------------------------------------------------------------

def load_prices_csv(data_path: str, *, tz=None) -> pd.DataFrame:
    """Charge les prix depuis les CSV bruts (DataAnalysis/data/raw/)."""
    from DataAnalysis.src.data_fetching import fetch_price_data
    from DataAnalysis.Utils.load_prices import load_all_prices

    fetch_price_data()
    df = load_all_prices(data_path)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()