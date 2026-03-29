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

    # Cointegration test strategy:
    #   "adf"  → ADF only   (paper Strategy 1 — Engle-Granger test)
    #   "kss"  → KSS only   (paper Strategy 2 — nonlinear stationarity)
    #   "both" → ADF AND KSS (strict conjunction, more restrictive, hors paper)
    cointegration_test: str = "adf"

    kss_crit: float = -1.92
    min_obs: int = 200
    min_coverage: float = 0.90  # % non-NaN vs ref timeline (drop symbols below)
    suppress_fit_logs: bool = True  # silence CMLE/loglik boundary spam

    # selection
    # "kendall_spread_ref" = paper Eq.39 : τ(S_i, P_ref) on spread vs BTC
    # "kendall_prices"     = τ(P_ref, P_coin) on raw prices (off-paper)
    rank_method: str = "kendall_spread_pair"

    top_k: int = 2   # how many coins to pick (2 => 1 pair)

    # copula
    copula_pick: str = "best_aic"
    copula_manual: str = "gaussian"

    # trading
    entry: float = 0.20
    exit: float = 0.10
    flip_on_opposite: bool = False

    # sizing & costs
    # cap_per_leg is the base capital per leg.
    # Actual quantities are beta-weighted: q_i = β_i * cap_per_leg / P_i
    # This ensures the BTC leg cancels and the position is market-neutral
    # with respect to the reference asset (paper Section 4).
    cap_per_leg: float = 200000.0   # paper: each side max ~200k USDT
    initial_equity: float = 200000.0  # paper: 200k USDT total invested
    fee_rate: float = 0.0004       # taker futures typical

    # ============================================================
    # RISK MANAGEMENT / STOP-LOSS
    # ============================================================
    # Trade-level stop-loss: ferme le trade si unrealized PnL < -X% du notionnel (2 legs)
    use_trade_stop_loss: bool = False
    trade_stop_loss_pct: float = 0.03      # 3% du notionnel total (2 * cap_per_leg)

    # Daily drawdown limit: stop trading pour la journée si equity baisse de > Y%
    use_daily_drawdown_limit: bool = False
    daily_drawdown_limit_pct: float = 0.02  # 2% de l'equity au début de la journée

    # Max portfolio drawdown: arrête totalement si drawdown > Z% depuis HWM
    use_max_drawdown_stop: bool = False
    max_drawdown_stop_pct: float = 0.15     # 15% de drawdown max => stop tout

    # Trailing stop sur PnL du trade (optionnel) : si le trade a gagné, protège les gains
    use_trailing_stop: bool = False
    trailing_stop_pct: float = 0.02         # 2% depuis le peak PnL du trade
    trailing_stop_activation: float = 0.01  # s'active quand PnL > 1% du notionnel

    # Position closing behaviour
    # True → comportement papier : toutes positions fermees en fin de semaine
    # False → multi-slot : les positions ouvertes continuent jusqu'a signal/stop,
    #          le copula d'origine continue de generer les signaux de sortie
    force_week_end_close: bool = True

    # misc
    seed: int = 42