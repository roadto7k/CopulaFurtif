"""
Simple trading signals or indicators.
"""

import pandas as pd

def moving_average_signal(data: pd.DataFrame, short_window=20, long_window=50):
    """
    Example: compute a signal based on moving average crossover.
    """
    data["ma_short"] = data["price"].rolling(short_window).mean()
    data["ma_long"] = data["price"].rolling(long_window).mean()
    data["signal"] = 0
    data.loc[data["ma_short"] > data["ma_long"], "signal"] = 1
    data.loc[data["ma_short"] < data["ma_long"], "signal"] = -1
    return data["signal"]
