from statsmodels.tsa.stattools import adfuller
from itertools import combinations
import numpy as np
from datetime import datetime
import time
import json
import os
from config import ADF_SIGNIFICANCE_LEVEL
from config import DATA_PATH

META_PATH = os.path.join(DATA_PATH, "..", "meta.json")
def get_meta():
    if os.path.exists(META_PATH):
        with open(META_PATH, 'r') as f:
            return json.load(f)
    return {}

def save_meta(meta):
    with open(META_PATH, 'w') as f:
        json.dump(meta, f)

def get_cached_cointegrated_pairs(symbols):
    meta = get_meta()
    today = datetime.utcnow().strftime('%Y-%m-%d')
    key = ','.join(sorted(symbols))

    if (meta.get("cointegrated_pairs", {}).get("date") == today and
        meta["cointegrated_pairs"].get("symbols") == key):
        return meta["cointegrated_pairs"].get("pairs")
    return None

def cache_cointegrated_pairs(symbols, pairs):
    meta = get_meta()
    meta["last_download_date"] = datetime.utcnow().strftime('%Y-%m-%d')
    meta["cointegrated_pairs"] = {
        "date": datetime.utcnow().strftime('%Y-%m-%d'),
        "symbols": ','.join(sorted(symbols)),
        "pairs": pairs
    }
    save_meta(meta)

def adf_test(series):
    result = adfuller(series)
    return result[1]

def engle_granger_test(x, y):
    x, y = x.align(y, join='inner') 
    mask = (~x.isna()) & (~y.isna()) & np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < 10 or np.std(y) < 1e-6 or np.std(x) < 1e-6:
        return 1.0  
    try:
        beta = np.polyfit(y, x, 1)[0]
        spread = x - beta * y
        return adf_test(spread)
    except np.linalg.LinAlgError:
        return 1.0

def find_cointegrated_pairs(prices):
    pairs = []
    symbols = prices.columns
    for (sym1, sym2) in combinations(symbols, 2):
        pval = engle_granger_test(prices[sym1], prices[sym2])
        if pval < ADF_SIGNIFICANCE_LEVEL:
            pairs.append((sym1, sym2, pval))
    return sorted(pairs, key=lambda x: x[2])
