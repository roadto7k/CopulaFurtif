from dataclasses import dataclass
from typing import List

@dataclass
class BacktestParams:
    strategy: str
    interval: str
    start: str
    end: str
    ref: str
    symbols: List[str]

    # cycle config
    formation_weeks: int = 3
    trading_weeks: int = 1
    step_weeks: int = 1

    # filters
    adf_alpha: float = 0.10
    use_kss: bool = False
    kss_crit: float = -1.92
    min_obs: int = 200
    min_coverage: float = 0.90  # % non-NaN vs ref timeline (drop symbols below)
    suppress_fit_logs: bool = True  # silence CMLE/loglik boundary spam

    # selection
    rank_method: str = "kendall_prices"
    top_k: int = 2   # how many coins to pick (2 => 1 pair)

    # copula
    copula_pick: str = "best_aic"
    copula_manual: str = "gaussian"

    # trading
    entry: float = 0.10
    exit: float = 0.10
    flip_on_opposite: bool = True

    # sizing & costs
    cap_per_leg: float = 20000.0   # like the paper's per-coin cap
    initial_equity: float = 40000.0
    fee_rate: float = 0.0004       # taker futures typical

    # ============================================================
    # RISK MANAGEMENT / STOP-LOSS
    # ============================================================
    # Trade-level stop-loss: ferme le trade si unrealized PnL < -X% du notionnel (2 legs)
    use_trade_stop_loss: bool = True
    trade_stop_loss_pct: float = 0.03      # 3% du notionnel total (2 * cap_per_leg)

    # Daily drawdown limit: stop trading pour la journée si equity baisse de > Y%
    use_daily_drawdown_limit: bool = False
    daily_drawdown_limit_pct: float = 0.02  # 2% de l'equity au début de la journée

    # Max portfolio drawdown: arrête totalement si drawdown > Z% depuis HWM
    use_max_drawdown_stop: bool = True
    max_drawdown_stop_pct: float = 0.15     # 15% de drawdown max => stop tout

    # Trailing stop sur PnL du trade (optionnel) : si le trade a gagné, protège les gains
    use_trailing_stop: bool = False
    trailing_stop_pct: float = 0.02         # 2% depuis le peak PnL du trade
    trailing_stop_activation: float = 0.01  # s'active quand PnL > 1% du notionnel

    # misc
    seed: int = 42