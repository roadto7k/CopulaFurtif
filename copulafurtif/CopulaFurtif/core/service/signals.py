"""
Service for trading logic orchestration (calculating signals, placing orders, etc.).
"""

import pandas as pd

from CopulaFurtif.core.signal.signals import moving_average_signal


class TradingService:
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Example: generate some signals using domain logic.
        """
        return moving_average_signal(data)

    def backtest_signals(self, data: pd.DataFrame) -> float:
        """
        Example: naive PnL or Sharpe calculation.
        """
        signals = self.generate_signals(data)
        # backtesting logic...
        return 0.0  # placeholder
