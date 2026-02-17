# dash_bot/core/__init__.py

from .params import BacktestParams
from .metrics import performance_metrics
from .selection import select_stationary_spreads, rank_coins
from .copula_engine import (
    is_copula_evaluable,
    fit_pair_copula,
    build_copula,
    copula_h_funcs,
    ecdf_value,
)
from .signals import generate_signals_reference_copula
from .backtest import backtest_reference_copula

__all__ = [
    "BacktestParams",
    "performance_metrics",
    "select_stationary_spreads",
    "rank_coins",
    "is_copula_evaluable",
    "fit_pair_copula",
    "build_copula",
    "copula_h_funcs",
    "ecdf_value",
    "generate_signals_reference_copula",
    "backtest_reference_copula",
]
