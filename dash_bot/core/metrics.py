import pandas as pd
import numpy as np
import math

def _to_datetime(s):
    return pd.to_datetime(s, utc=False)

def _safe_pct(x):
    return float(x) * 100.0

def _annualization_factor(interval: str):
    # Approx. number of bars per year
    if interval.endswith("m"):
        minutes = int(interval[:-1])
        return (365.0 * 24.0 * 60.0) / minutes
    if interval.endswith("h"):
        hours = int(interval[:-1])
        return (365.0 * 24.0) / hours
    if interval.endswith("d"):
        days = int(interval[:-1])
        return 365.0 / days
    return 365.0 * 24.0  # fallback

def performance_metrics(equity, returns, interval: str = "D"):
    eq = pd.Series(equity).replace([np.inf, -np.inf], np.nan).dropna()
    if len(eq) < 3:
        return dict(
            total_return=np.nan,
            annual_return=np.nan,
            annual_vol=np.nan,
            sharpe=np.nan,
            max_drawdown=np.nan,
            romad=np.nan,
        )

    rets = eq.pct_change().dropna()
    ann = _annualization_factor(interval)
    mu = rets.mean() * ann
    vol = rets.std(ddof=1) * math.sqrt(ann) if rets.std(ddof=1) > 0 else np.nan
    sharpe = (mu / vol) if (vol is not None and np.isfinite(vol) and vol > 0) else np.nan

    # drawdown
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    mdd = float(dd.min())
    tot = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    romad = (tot / abs(mdd)) if (np.isfinite(mdd) and mdd < 0) else np.nan

    return dict(
        total_return=tot,
        annual_return=float(mu),
        annual_vol=float(vol) if vol is not None else np.nan,
        sharpe=float(sharpe) if sharpe is not None else np.nan,
        max_drawdown=float(mdd),
        romad=float(romad) if romad is not None else np.nan,
    )


def risk_metrics(trades_df, equity, stop_loss_stats):
    """Métriques spécifiques au risk management."""
    if trades_df is None or trades_df.empty:
        return {}

    result = {}

    # Win rate
    if "net_pnl" in trades_df.columns:
        wins = (trades_df["net_pnl"] > 0).sum()
        total = len(trades_df)
        result["win_rate"] = float(wins / total) if total > 0 else 0.0
        result["avg_win"] = float(trades_df.loc[trades_df["net_pnl"] > 0, "net_pnl"].mean()) if wins > 0 else 0.0
        result["avg_loss"] = float(trades_df.loc[trades_df["net_pnl"] <= 0, "net_pnl"].mean()) if (
                                                                                                              total - wins) > 0 else 0.0
        result["profit_factor"] = abs(result["avg_win"] * wins / (result["avg_loss"] * (total - wins))) if result[
                                                                                                               "avg_loss"] != 0 else np.inf

    # Stop-loss specific
    if "exit_reason" in trades_df.columns:
        sl_types = ["TRADE_STOP_LOSS", "TRAILING_STOP", "DAILY_DD_LIMIT"]
        sl_trades = trades_df[trades_df["exit_reason"].isin(sl_types)]
        result["sl_trigger_count"] = len(sl_trades)
        result["sl_total_loss"] = float(sl_trades["net_pnl"].sum()) if not sl_trades.empty else 0.0
        result["sl_avg_loss"] = float(sl_trades["net_pnl"].mean()) if not sl_trades.empty else 0.0

        # Worst trade (sans SL) vs worst trade (avec SL)
        signal_trades = trades_df[~trades_df["exit_reason"].isin(sl_types)]
        result["worst_trade_sl"] = float(sl_trades["net_pnl"].min()) if not sl_trades.empty else 0.0
        result["worst_trade_signal"] = float(signal_trades["net_pnl"].min()) if not signal_trades.empty else 0.0

    return result
