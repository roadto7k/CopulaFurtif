# dash_bot/config.py

APP_TITLE = "Copula Trading Bot — Backtest Dashboard"

# Exact 20 USDT-Margined Futures used in Tadi & Witzany (2023) — Table VII
# Binance hourly data 01/01/2021 → 19/01/2023
# BTCUSDT = reference asset (not traded, intermediary only)
DEFAULT_USDT_SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BCHUSDT', 'XRPUSDT', 'EOSUSDT',
    'LTCUSDT', 'TRXUSDT', 'ETCUSDT', 'LINKUSDT', 'XLMUSDT',
    'ADAUSDT', 'XMRUSDT', 'DASHUSDT', 'ZECUSDT', 'XTZUSDT',
    'ATOMUSDT', 'BNBUSDT', 'ONTUSDT', 'IOTAUSDT', 'BATUSDT',
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

# "kendall_spread_ref" is first = default (paper Eq.39)
RANK_METHODS = [
    ("kendall_spread_ref", "Kendall τ(Spread(ref,coin), Price(ref))  ← paper Eq.39"),
    ("kendall_prices",     "Kendall τ(Price(ref), Price(coin))  [raw prices, off-paper]"),
]

COINTEGRATION_TESTS = [
    ("adf",  "ADF only — paper Strategy 1 (Engle-Granger test)"),
    ("kss",  "KSS only — paper Strategy 2 (nonlinear stationarity)"),
    ("both", "ADF AND KSS — strict conjunction (hors paper)"),
]

COPULA_PICK = [
    ("best_score", "Auto (best by Score: AIC + tau + tails)"),
    ("best_score_pit", "Auto (best by Score + PIT)"),
    ("best_aic", "Auto (best by AIC only)"),
    ("manual", "Manual (force a copula family)"),
]