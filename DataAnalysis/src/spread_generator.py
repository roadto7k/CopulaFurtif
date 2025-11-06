import numpy as np

def generate_spread(reference_series, coin_series):
    beta = np.polyfit(coin_series, reference_series, 1)[0]
    spread = reference_series - beta * coin_series
    return spread, beta
