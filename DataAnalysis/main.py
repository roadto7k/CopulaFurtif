from src.data_fetching import fetch_price_data
from src.cointegration import find_cointegrated_pairs, get_cached_cointegrated_pairs, cache_cointegrated_pairs
from src.spread_generator import generate_spread
from src.copula_pipeline import run_copula_analysis
from config import DATA_PATH, REFERENCE_ASSET

import pandas as pd
import os
import numpy as np


def load_all_prices():
    data = {}
    for file in os.listdir(DATA_PATH):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(DATA_PATH, file), index_col=0, parse_dates=True)
            data[file[:-4]] = df['close']
    return pd.DataFrame(data)

def main():
    fetch_price_data()

    prices = load_all_prices()
    log_returns = np.log(prices).diff().dropna()

    cached = get_cached_cointegrated_pairs(prices.columns)

    if cached is not None:
        cointegrated_pairs = cached
        print("✅ Paires cointegrées chargées depuis le cache :")
    else:
        cointegrated_pairs = find_cointegrated_pairs(prices.dropna())
        cache_cointegrated_pairs(prices.columns, cointegrated_pairs)
        print("🔍 Paires cointegrées recalculées :")

    print("pairs founds : ", cointegrated_pairs)
    print("pairs founds : ", cointegrated_pairs)
    
    btc = log_returns[REFERENCE_ASSET]
    for pair in cointegrated_pairs:
        if REFERENCE_ASSET not in pair:
            coin1_name, coin2_name = pair[0], pair[1]
            break
    else:
        raise ValueError("No cointegrated pair excluding BTCUSDT found.")

    coin1 = log_returns[coin1_name]
    coin2 = log_returns[coin2_name]

    s1, _ = generate_spread(btc, coin1)
    s2, _ = generate_spread(btc, coin2)

    run_copula_analysis(s1, s2)

if __name__ == "__main__":
    main()
