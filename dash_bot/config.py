# dash_bot/config.py

APP_TITLE = "Copula Trading Bot — Backtest Dashboard"

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
    ("best_score", "Auto (best by Score: AIC + tau + tails)"),
    ("best_score_pit", "Auto (best by Score + PIT)"),
    ("best_aic", "Auto (best by AIC only)"),
    ("manual", "Manual (force a copula family)"),
]