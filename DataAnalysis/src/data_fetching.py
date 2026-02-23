"""
data_fetching.py — Téléchargement des données OHLCV via l'API REST Binance.

Remplace python-binance par des appels requests directs :
  - Pas de dépendance asyncio/ProactorEventLoop (bug Windows)
  - Pas de SSL custom (bug Windows + Cloudflare)
  - Pagination manuelle explicite
  - Rate limiting respecté (~500 weight/min, bien sous les 1200 autorisés)
"""

import os
import time
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List

from DataAnalysis.config import (
    SYMBOLS, INTERVAL, START_DATE, DATA_PATH,
    USE_API_KEY, API_KEY, API_SECRET,
)

META_PATH = os.path.join(DATA_PATH, "..", "meta.json")

# ---------------------------------------------------------------------------
# Constantes Binance REST
# ---------------------------------------------------------------------------
BASE_URL = "https://api.binance.com"
KLINES_ENDPOINT = "/api/v3/klines"
MAX_LIMIT = 1000          # max candles par requête (poids = 2 pour limit > 500)
SLEEP_BETWEEN_CALLS = 0.25  # secondes entre appels (~240 req/min << 1200 limit)

_SESSION: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    """Session requests réutilisable avec headers Binance."""
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        _SESSION.headers.update({
            "Accept": "application/json",
            "User-Agent": "CopulaFurtif/1.0",
        })
        if USE_API_KEY and API_KEY and API_KEY not in ("fck_me", ""):
            _SESSION.headers["X-MBX-APIKEY"] = API_KEY
    return _SESSION


def _interval_to_ms(interval: str) -> int:
    """Convertit un intervalle Binance ('15m', '1h'…) en millisecondes."""
    units = {"m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}
    for suffix, factor in units.items():
        if interval.endswith(suffix):
            return int(interval[:-1]) * factor
    raise ValueError(f"Intervalle non reconnu : {interval}")


def _parse_start_date(date_str: str) -> int:
    """Convertit une date textuelle ou ISO en timestamp ms UTC."""
    for fmt in ("%d %B, %Y", "%d %B %Y", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return int(dt.timestamp() * 1000)
        except ValueError:
            continue
    return int(pd.to_datetime(date_str).timestamp() * 1000)


def fetch_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: Optional[int] = None,
) -> pd.DataFrame:
    """
    Récupère toutes les bougies Binance SPOT pour un symbole via pagination.

    Args:
        symbol:    ex. 'BTCUSDT'
        interval:  ex. '15m'
        start_ms:  timestamp début en millisecondes UTC
        end_ms:    timestamp fin   en millisecondes UTC (None = maintenant)

    Returns:
        DataFrame avec colonnes : open, high, low, close, volume
        Index : DatetimeIndex UTC
    """
    session = _get_session()
    if end_ms is None:
        end_ms = int(time.time() * 1000)

    interval_ms = _interval_to_ms(interval)
    all_rows: list = []
    since = start_ms

    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": since,
            "endTime": end_ms,
            "limit": MAX_LIMIT,
        }
        try:
            resp = session.get(
                BASE_URL + KLINES_ENDPOINT,
                params=params,
                timeout=20,
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"[{symbol}] Erreur réseau : {e}") from e

        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 60))
            print(f"[{symbol}] Rate limit (429) — pause {retry_after}s")
            time.sleep(retry_after)
            continue

        if resp.status_code == 418:
            raise RuntimeError(
                f"[{symbol}] IP bannie par Binance (418). "
                "Attends quelques minutes ou utilise une clé API."
            )

        if not resp.ok:
            raise RuntimeError(
                f"[{symbol}] Binance API erreur {resp.status_code}: {resp.text[:200]}"
            )

        batch = resp.json()
        if not batch:
            break

        all_rows.extend(batch)
        last_open_time = batch[-1][0]

        if len(batch) < MAX_LIMIT or last_open_time >= end_ms:
            break

        since = last_open_time + interval_ms
        time.sleep(SLEEP_BETWEEN_CALLS)

    if not all_rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    arr = np.array(all_rows, dtype=object)
    idx = pd.to_datetime(arr[:, 0].astype(np.int64), unit="ms", utc=True)
    df = pd.DataFrame({
        "open":   arr[:, 1].astype(float),
        "high":   arr[:, 2].astype(float),
        "low":    arr[:, 3].astype(float),
        "close":  arr[:, 4].astype(float),
        "volume": arr[:, 5].astype(float),
    }, index=idx)
    df.index.name = "timestamp"
    df = df[df.index <= pd.Timestamp(end_ms, unit="ms", tz="UTC")]
    return df


# ---------------------------------------------------------------------------
# Interface publique (identique à l'ancienne version)
# ---------------------------------------------------------------------------

def already_fetched_today() -> bool:
    if not os.path.exists(META_PATH):
        return False
    try:
        with open(META_PATH, "r") as f:
            meta = json.load(f)
        return meta.get("last_download_date") == datetime.utcnow().strftime("%Y-%m-%d")
    except (json.JSONDecodeError, OSError):
        return False


def update_meta() -> None:
    os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
    with open(META_PATH, "w") as f:
        json.dump({"last_download_date": datetime.utcnow().strftime("%Y-%m-%d")}, f)


def fetch_price_data(
    symbols: Optional[List[str]] = None,
    interval: Optional[str] = None,
    start_date: Optional[str] = None,
    data_path: Optional[str] = None,
    force: bool = False,
) -> None:
    """
    Télécharge les données OHLCV pour tous les symboles et les sauvegarde en CSV.

    Args:
        symbols:    liste de symboles (défaut: config.SYMBOLS)
        interval:   intervalle (défaut: config.INTERVAL)
        start_date: date de début (défaut: config.START_DATE)
        data_path:  dossier de sortie (défaut: config.DATA_PATH)
        force:      forcer le re-téléchargement même si déjà fait aujourd'hui
    """
    symbols    = symbols    or SYMBOLS
    interval   = interval   or INTERVAL
    start_date = start_date or START_DATE
    data_path  = data_path  or DATA_PATH

    if not force and already_fetched_today():
        print("✅ Données déjà à jour — téléchargement ignoré.")
        return

    os.makedirs(data_path, exist_ok=True)
    start_ms = _parse_start_date(start_date)
    end_ms   = int(time.time() * 1000)

    errors = {}
    for symbol in symbols:
        print(f"⬇️  Downloading {symbol} ({interval}) …")
        try:
            df = fetch_klines(symbol, interval, start_ms, end_ms)
            if df.empty:
                errors[symbol] = "Aucune donnée retournée"
                print(f"  ⚠️  {symbol} : vide")
                continue
            out_path = os.path.join(data_path, f"{symbol}.csv")
            df.to_csv(out_path)
            print(f"  ✅ {symbol} : {len(df)} bougies → {out_path}")
        except Exception as e:
            errors[symbol] = str(e)
            print(f"  ❌ {symbol} : {e}")
        time.sleep(0.5)

    if errors:
        print(f"\n⚠️  Erreurs sur {len(errors)} symbole(s) : {list(errors.keys())}")

    update_meta()
    print("\n✅ Téléchargement terminé.")