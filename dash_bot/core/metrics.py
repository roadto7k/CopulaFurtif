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

def performance_metrics(
    equity,
    interval: str = "D",
):
    """
    Compute strategy performance metrics.

    The primary Sharpe ratio is computed from:

        annualized geometric return
        --------------------------------
        annualized monthly volatility

    This provides a stable strategy-level metric and a comparison basis
    close to the methodology reported in the reference paper.

    The previous bar-frequency Sharpe is retained as a diagnostic metric.
    """
    eq = (
        pd.Series(equity)
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    if len(eq) < 3:
        return dict(
            total_return=np.nan,
            annual_return=np.nan,
            annual_vol=np.nan,
            sharpe=np.nan,
            sharpe_bar=np.nan,
            annual_vol_bar=np.nan,
            max_drawdown=np.nan,
            romad=np.nan,
        )

    # ------------------------------------------------------------------
    # Total return
    # ------------------------------------------------------------------
    initial_equity = float(
        eq.iloc[0]
    )

    final_equity = float(
        eq.iloc[-1]
    )

    total_return = (
        final_equity / initial_equity
        - 1.0
    )

    # ------------------------------------------------------------------
    # Geometric annualized return
    # ------------------------------------------------------------------
    elapsed_seconds = (
        pd.to_datetime(eq.index[-1])
        - pd.to_datetime(eq.index[0])
    ).total_seconds()

    elapsed_years = (
        elapsed_seconds
        / (
            365.25
            * 24.0
            * 60.0
            * 60.0
        )
    )

    if (
        elapsed_years > 0.0
        and initial_equity > 0.0
        and final_equity > 0.0
    ):
        annual_return = (
            (
                final_equity
                / initial_equity
            )
            ** (
                1.0
                / elapsed_years
            )
            - 1.0
        )

    else:
        annual_return = np.nan

    # ------------------------------------------------------------------
    # Monthly strategy returns
    # ------------------------------------------------------------------
    monthly_returns = (
        eq.resample("ME")
        .last()
        .pct_change()
        .dropna()
    )

    if (
        len(monthly_returns) >= 2
        and monthly_returns.std(
            ddof=1
        ) > 0.0
    ):
        annual_vol = float(
            monthly_returns.std(
                ddof=1
            )
            * math.sqrt(12.0)
        )

        sharpe = (
            annual_return
            / annual_vol
            if (
                np.isfinite(annual_return)
                and annual_vol > 0.0
            )
            else np.nan
        )

    else:
        annual_vol = np.nan
        sharpe = np.nan

    # ------------------------------------------------------------------
    # Naive bar-frequency Sharpe
    #
    # Retained for diagnostics only.
    # ------------------------------------------------------------------
    bar_returns = (
        eq.pct_change()
        .dropna()
    )

    ann = _annualization_factor(
        interval
    )

    bar_std = bar_returns.std(
        ddof=1
    )

    if (
        len(bar_returns) >= 2
        and np.isfinite(bar_std)
        and bar_std > 0.0
    ):
        annual_vol_bar = float(
            bar_std
            * math.sqrt(ann)
        )

        sharpe_bar = float(
            (
                bar_returns.mean()
                / bar_std
            )
            * math.sqrt(ann)
        )

    else:
        annual_vol_bar = np.nan
        sharpe_bar = np.nan

    # ------------------------------------------------------------------
    # Drawdown
    # ------------------------------------------------------------------
    peak = eq.cummax()

    drawdown = (
        eq / peak
        - 1.0
    )

    max_drawdown = float(
        drawdown.min()
    )

    romad = (
        total_return
        / abs(max_drawdown)
        if (
            np.isfinite(max_drawdown)
            and max_drawdown < 0.0
        )
        else np.nan
    )

    return dict(
        total_return=float(
            total_return
        ),
        annual_return=float(
            annual_return
        ) if np.isfinite(
            annual_return
        ) else np.nan,
        annual_vol=float(
            annual_vol
        ) if np.isfinite(
            annual_vol
        ) else np.nan,
        sharpe=float(
            sharpe
        ) if np.isfinite(
            sharpe
        ) else np.nan,
        sharpe_bar=float(
            sharpe_bar
        ) if np.isfinite(
            sharpe_bar
        ) else np.nan,
        annual_vol_bar=float(
            annual_vol_bar
        ) if np.isfinite(
            annual_vol_bar
        ) else np.nan,
        max_drawdown=max_drawdown,
        romad=float(
            romad
        ) if np.isfinite(
            romad
        ) else np.nan,
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
