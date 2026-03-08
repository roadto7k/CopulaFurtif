# dash_bot/viz/figures.py
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
import plotly.graph_objs as go

TRON_LAYOUT = dict(
    template="plotly_dark",
    autosize=True,
    plot_bgcolor="rgba(6,11,20,0.0)",
    paper_bgcolor="rgba(10,22,40,0.95)",
    font=dict(family="Share Tech Mono, monospace", color="#c8dce8", size=12),
    margin=dict(l=35, r=25, t=45, b=35),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(0,240,255,0.08)",
        borderwidth=1,
        font=dict(size=10),
    ),
    xaxis=dict(
        gridcolor="rgba(0,240,255,0.06)",
        zerolinecolor="rgba(0,240,255,0.10)",
        linecolor="rgba(0,240,255,0.10)",
        tickfont=dict(color="#5a7a90"),
    ),
    yaxis=dict(
        gridcolor="rgba(0,240,255,0.06)",
        zerolinecolor="rgba(0,240,255,0.10)",
        linecolor="rgba(0,240,255,0.10)",
        tickfont=dict(color="#5a7a90"),
    ),
)

def fig_empty(title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**TRON_LAYOUT, height=320, title=title)
    return fig

def fig_equity(
    equity: pd.Series,
    equity_gross: Optional[pd.Series] = None,
    stop_loss_events: Optional[list] = None,
) -> go.Figure:
    equity = pd.Series(equity).replace([np.inf, -np.inf], np.nan).dropna()
    if equity_gross is not None:
        equity_gross = pd.Series(equity_gross).replace([np.inf, -np.inf], np.nan).dropna()

    fig = go.Figure()

    # Equity net
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values,
        name="Equity (net)", mode="lines",
        line=dict(color="#00f0ff", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(0,240,255,0.04)",
    ))

    # Equity gross
    if equity_gross is not None and len(equity_gross) > 0:
        fig.add_trace(go.Scatter(
            x=equity_gross.index, y=equity_gross.values,
            name="Equity (gross)", mode="lines",
            line=dict(dash="dot", color="#ff2eed", width=1.5),
        ))

    # >>> Stop-loss markers
    if stop_loss_events:
        sl_x, sl_y, sl_text = [], [], []

        for e in stop_loss_events:
            try:
                t = e.get("time")
                eq = e.get("equity", None)
                tp = e.get("type", "stop_loss")

                if t is None or eq is None:
                    continue

                ts = pd.to_datetime(t)
                if pd.isna(ts):
                    continue

                # equity value must be numeric
                y = float(eq)
                if not np.isfinite(y):
                    continue

                sl_x.append(ts)
                sl_y.append(y)
                sl_text.append(tp)
            except Exception:
                # On ignore les events mal formés au lieu de casser le graph
                continue

        if sl_x:
            fig.add_trace(go.Scatter(
                x=sl_x,
                y=sl_y,
                mode="markers",
                name="Stop-Loss Triggers",
                marker=dict(
                    symbol="x",
                    size=10,
                    color="#ff3355",
                    line=dict(width=2, color="#ff3355"),
                ),
                text=sl_text,
                hovertemplate="<b>%{text}</b><br>Date: %{x}<br>Equity: %{y:.0f}<extra></extra>",
            ))

    fig.update_layout(
        **TRON_LAYOUT,
        height=360,
        title="⬡  EQUITY CURVE",
        xaxis_title="Date",
        yaxis_title="USDT",
    )
    return fig

def fig_drawdown(equity: pd.Series) -> go.Figure:
    eq = pd.Series(equity).replace([np.inf, -np.inf], np.nan).dropna()
    if len(eq) < 3:
        return fig_empty("Drawdown")
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        name="Drawdown", mode="lines",
        line=dict(color="#ff3355", width=2),
        fill="tozeroy",
        fillcolor="rgba(255,51,85,0.06)",
    ))
    fig.update_layout(**TRON_LAYOUT, height=260, title="⬡  DRAWDOWN",
                      xaxis_title="Date", yaxis_title="Drawdown")
    return fig

def fig_monthly_heatmap(monthly_returns: pd.Series) -> go.Figure:
    if monthly_returns is None:
        return fig_empty("Monthly returns heatmap")
    monthly_returns = pd.Series(monthly_returns).replace([np.inf, -np.inf], np.nan).dropna()
    if len(monthly_returns) == 0:
        return fig_empty("Monthly returns heatmap")

    mr = monthly_returns.copy()
    df = pd.DataFrame({"ret": mr})
    df["year"] = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot_table(index="year", columns="month", values="ret", aggfunc="sum")

    # Tron-style colorscale: red → dark → cyan → bright
    tron_colorscale = [
        [0.0, "#ff3355"],
        [0.25, "#3a0e1a"],
        [0.45, "#0a1628"],
        [0.55, "#0a1628"],
        [0.75, "#003844"],
        [1.0, "#00f0ff"],
    ]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[str(m) for m in pivot.columns],
        y=[str(y) for y in pivot.index],
        colorscale=tron_colorscale,
        colorbar=dict(
            title=dict(text="Return", font=dict(family="Orbitron", size=10, color="#5a7a90")),
            tickfont=dict(family="Share Tech Mono", size=9, color="#5a7a90"),
            outlinecolor="rgba(0,240,255,0.1)",
            outlinewidth=1,
        ),
        xgap=2, ygap=2,
    ))
    fig.update_layout(**TRON_LAYOUT, height=340, title="⬡  MONTHLY RETURNS (NET)",
                      xaxis_title="Month", yaxis_title="Year")
    return fig

def fig_copula_freq(cop_freq: pd.DataFrame) -> go.Figure:
    if cop_freq is None or cop_freq.empty:
        return fig_empty("Copula frequency")

    neon_palette = ["#00f0ff", "#ff2eed", "#00ff88", "#ff6f1a", "#ffe01a", "#7b61ff", "#ff3355"]
    n = len(cop_freq)
    colors = [neon_palette[i % len(neon_palette)] for i in range(n)]

    fig = go.Figure(data=go.Bar(
        x=cop_freq["copula"].astype(str),
        y=cop_freq["count"].astype(int),
        name="count",
        marker=dict(
            color=colors,
            line=dict(color=colors, width=1.5),
            opacity=0.85,
        ),
    ))
    fig.update_layout(**TRON_LAYOUT, height=320, title="⬡  COPULA SELECTION FREQUENCY",
                      xaxis_title="Copula", yaxis_title="# Weeks")
    return fig

def compute_copula_stats(trades: pd.DataFrame) -> pd.DataFrame:
    """Compute per-copula performance statistics from trades DataFrame."""
    if trades is None or trades.empty or "copula" not in trades.columns:
        return pd.DataFrame()

    required = {"copula", "net_pnl"}
    if not required.issubset(trades.columns):
        return pd.DataFrame()

    df = trades[trades["copula"].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    def agg(g):
        n        = len(g)
        wins     = (g["net_pnl"] > 0).sum()
        win_rate = wins / n if n > 0 else 0.0
        avg_pnl  = g["net_pnl"].mean()
        avg_win  = g.loc[g["net_pnl"] > 0, "net_pnl"].mean() if wins > 0 else 0.0
        avg_loss = g.loc[g["net_pnl"] <= 0, "net_pnl"].mean() if (n - wins) > 0 else 0.0
        total    = g["net_pnl"].sum()
        best     = g["net_pnl"].max()
        worst    = g["net_pnl"].min()
        profit_factor = (
            abs(g.loc[g["net_pnl"] > 0, "net_pnl"].sum()) /
            abs(g.loc[g["net_pnl"] <= 0, "net_pnl"].sum())
            if (g["net_pnl"] <= 0).any() and (g["net_pnl"] > 0).any()
            else np.nan
        )
        avg_bars = g["bars"].mean() if "bars" in g.columns else np.nan
        return pd.Series({
            "Trades":        n,
            "Win Rate":      f"{win_rate:.1%}",
            "Avg PnL":       f"{avg_pnl:+.1f}",
            "Avg Win":       f"{avg_win:+.1f}",
            "Avg Loss":      f"{avg_loss:+.1f}",
            "Total PnL":     f"{total:+.1f}",
            "Best":          f"{best:+.1f}",
            "Worst":         f"{worst:+.1f}",
            "Profit Factor": f"{profit_factor:.2f}" if np.isfinite(profit_factor) else "—",
            "Avg Bars":      f"{avg_bars:.0f}" if np.isfinite(avg_bars) else "—",
        })

    stats = df.groupby("copula").apply(agg).reset_index()
    stats.rename(columns={"copula": "Copula"}, inplace=True)
    # Sort by total PnL descending (parse back to float for sorting)
    stats["_sort"] = df.groupby("copula")["net_pnl"].sum().reindex(
        stats["Copula"]
    ).values
    stats = stats.sort_values("_sort", ascending=False).drop(columns=["_sort"])
    return stats


def fig_prices(prices, *, title: str = "Prices"):
    if prices is None:
        return fig_empty(title)

    df = pd.DataFrame(prices).replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if df.empty:
        return fig_empty(title)

    fig = go.Figure()
    for c in df.columns:
        s = df[c].dropna()
        if len(s) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values,
            name=str(c),
            mode="lines",
            line=dict(width=1.6),
        ))

    fig.update_layout(**TRON_LAYOUT, height=360, title=f"⬡  {title.upper()}",
                      xaxis_title="Date", yaxis_title="Price")
    return fig

def fig_spread(spread, *, title: str = "Spread"):
    if spread is None:
        return fig_empty(title)

    s = pd.Series(spread).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return fig_empty(title)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s.index, y=s.values,
        name="Spread",
        mode="lines",
        line=dict(color="#00ff88", width=2.0),
        fill="tozeroy",
        fillcolor="rgba(0,255,136,0.05)",
    ))
    fig.update_layout(**TRON_LAYOUT, height=320, title=f"⬡  {title.upper()}",
                      xaxis_title="Date", yaxis_title="Spread")
    return fig