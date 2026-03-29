# dash_bot/viz/figures_diagnostic.py
"""
Cycle-level diagnostic visualizations for the copula pairs trading dashboard.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from .figures import TRON_LAYOUT, fig_empty

# ── Palette ──────────────────────────────────────────────────────────────
CYAN       = "#00f0ff"
MAGENTA    = "#ff2eed"
GREEN      = "#00ff88"
RED        = "#ff3355"
ORANGE     = "#ff6f1a"
GOLD       = "#ffe01a"
PURPLE     = "#7b61ff"
SUBTEXT    = "#5a7a90"
PANEL_BG   = "rgba(10,22,40,0.95)"
GRID_COLOR = "rgba(0,240,255,0.06)"

ZONE_OPEN_LONG  = "rgba(0,255,136,0.12)"
ZONE_OPEN_SHORT = "rgba(255,51,85,0.12)"
ZONE_CLOSE      = "rgba(255,228,26,0.10)"


def _tron_layout(**overrides) -> dict:
    """
    Merge TRON_LAYOUT with overrides. Later keys win → no duplicate kwarg error.
    """
    out = dict(TRON_LAYOUT)
    out.update(overrides)
    return out


# ═════════════════════════════════════════════════════════════════════════
# 1. COPULA DECISION MAP
# ═════════════════════════════════════════════════════════════════════════

def fig_copula_decision_map(
    cop: Any,
    u_trades: np.ndarray,
    v_trades: np.ndarray,
    signals: np.ndarray,
    entry: float,
    exit_thr: float,
    copula_name: str = "",
    pair_label: str = "",
    Ngrid: int = 80,
) -> go.Figure:
    fig = go.Figure()

    eps = 1e-4
    grid = np.linspace(eps, 1 - eps, Ngrid)
    U, V = np.meshgrid(grid, grid)

    H12 = np.zeros_like(U)
    H21 = np.zeros_like(U)

    for i in range(Ngrid):
        for j in range(Ngrid):
            u, v = float(U[i, j]), float(V[i, j])
            try:
                if hasattr(cop, "conditional_cdf_u_given_v"):
                    H12[i, j] = float(cop.conditional_cdf_u_given_v(u, v))
                    H21[i, j] = float(cop.conditional_cdf_v_given_u(u, v))
                else:
                    H12[i, j] = 0.5
                    H21[i, j] = 0.5
            except Exception:
                H12[i, j] = 0.5
                H21[i, j] = 0.5

    Z = np.zeros_like(U)
    close_mask = (np.abs(H12 - 0.5) < exit_thr) & (np.abs(H21 - 0.5) < exit_thr)
    long_mask  = (H12 > 1 - entry) & (H21 < entry)
    short_mask = (H12 < entry) & (H21 > 1 - entry)

    Z[long_mask]  = 1
    Z[short_mask] = -1
    Z[close_mask] = 2

    zone_colorscale = [
        [0.0,  RED],
        [0.33, "rgba(10,22,40,0.95)"],
        [0.66, GREEN],
        [1.0,  GOLD],
    ]

    fig.add_trace(go.Heatmap(
        x=grid, y=grid, z=Z,
        colorscale=zone_colorscale, zmin=-1, zmax=2,
        showscale=False, opacity=0.35, hoverinfo="skip",
    ))

    try:
        Z_pdf = np.zeros_like(U)
        for i in range(Ngrid):
            for j in range(Ngrid):
                try:
                    Z_pdf[i, j] = float(cop.get_pdf(float(U[i, j]), float(V[i, j])))
                except Exception:
                    Z_pdf[i, j] = 0.0
        Z_pdf = np.clip(Z_pdf, 0, np.percentile(Z_pdf[Z_pdf > 0], 98) if (Z_pdf > 0).any() else 1)
        fig.add_trace(go.Contour(
            x=grid, y=grid, z=Z_pdf, ncontours=12, colorscale="Blues",
            showscale=False, opacity=0.20,
            line=dict(width=0.5, color="rgba(0,240,255,0.15)"), hoverinfo="skip",
        ))
    except Exception:
        pass

    fig.add_trace(go.Contour(
        x=grid, y=grid, z=H12,
        contours=dict(start=entry, end=entry, size=0.01, coloring="none"),
        line=dict(color=RED, width=1.5, dash="dot"),
        showscale=False, hoverinfo="skip", name=f"h12={entry:.2f}",
    ))
    fig.add_trace(go.Contour(
        x=grid, y=grid, z=H12,
        contours=dict(start=1 - entry, end=1 - entry, size=0.01, coloring="none"),
        line=dict(color=GREEN, width=1.5, dash="dot"),
        showscale=False, hoverinfo="skip", name=f"h12={1-entry:.2f}",
    ))
    fig.add_trace(go.Contour(
        x=grid, y=grid, z=H21,
        contours=dict(start=entry, end=entry, size=0.01, coloring="none"),
        line=dict(color=RED, width=1.5, dash="dash"),
        showscale=False, hoverinfo="skip", name=f"h21={entry:.2f}",
    ))
    fig.add_trace(go.Contour(
        x=grid, y=grid, z=H21,
        contours=dict(start=1 - entry, end=1 - entry, size=0.01, coloring="none"),
        line=dict(color=GREEN, width=1.5, dash="dash"),
        showscale=False, hoverinfo="skip", name=f"h21={1-entry:.2f}",
    ))

    for val in [0.5 - exit_thr, 0.5 + exit_thr]:
        fig.add_trace(go.Contour(
            x=grid, y=grid, z=H12,
            contours=dict(start=val, end=val, size=0.01, coloring="none"),
            line=dict(color=GOLD, width=1, dash="dot"),
            showscale=False, hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Contour(
            x=grid, y=grid, z=H21,
            contours=dict(start=val, end=val, size=0.01, coloring="none"),
            line=dict(color=GOLD, width=1, dash="dot"),
            showscale=False, hoverinfo="skip", showlegend=False,
        ))

    if len(u_trades) > 0:
        sig = np.asarray(signals)

        fig.add_trace(go.Scatter(
            x=u_trades, y=v_trades, mode="lines",
            line=dict(color="rgba(200,220,232,0.15)", width=0.8),
            hoverinfo="skip", showlegend=False,
        ))

        mask_hold = sig == 0
        if mask_hold.any():
            fig.add_trace(go.Scatter(
                x=u_trades[mask_hold], y=v_trades[mask_hold],
                mode="markers", name="Hold",
                marker=dict(size=4, color=SUBTEXT, opacity=0.4),
                hovertemplate="u=%{x:.3f} v=%{y:.3f}<extra>hold</extra>",
            ))

        mask_long = sig == 1
        if mask_long.any():
            fig.add_trace(go.Scatter(
                x=u_trades[mask_long], y=v_trades[mask_long],
                mode="markers", name="Open Long C1",
                marker=dict(size=10, color=GREEN, symbol="triangle-up",
                            line=dict(width=1, color="white")),
                hovertemplate="u=%{x:.3f} v=%{y:.3f}<extra>LONG coin1</extra>",
            ))

        mask_short = sig == -1
        if mask_short.any():
            fig.add_trace(go.Scatter(
                x=u_trades[mask_short], y=v_trades[mask_short],
                mode="markers", name="Open Short C1",
                marker=dict(size=10, color=RED, symbol="triangle-down",
                            line=dict(width=1, color="white")),
                hovertemplate="u=%{x:.3f} v=%{y:.3f}<extra>SHORT coin1</extra>",
            ))

    title = f"⬡  COPULA DECISION MAP — {copula_name}"
    if pair_label:
        title += f"  [{pair_label}]"

    fig.update_layout(**_tron_layout(
        height=520, width=560,
        title=title,
        xaxis=dict(title="u (ECDF spread₁)", range=[0, 1],
                   gridcolor=GRID_COLOR, tickfont=dict(color=SUBTEXT)),
        yaxis=dict(title="v (ECDF spread₂)", range=[0, 1],
                   gridcolor=GRID_COLOR, tickfont=dict(color=SUBTEXT),
                   scaleanchor="x", scaleratio=1),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(10,22,40,0.8)", font=dict(size=9)),
    ))
    return fig


# ═════════════════════════════════════════════════════════════════════════
# 2. H-FUNCTIONS TIME SERIES
# ═════════════════════════════════════════════════════════════════════════

def fig_h_functions_timeseries(
    timestamps: pd.DatetimeIndex,
    h12_series: np.ndarray,
    h21_series: np.ndarray,
    signals: np.ndarray,
    entry: float,
    exit_thr: float,
    pair_label: str = "",
) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=["h₁|₂ = P(U₁≤u₁ | U₂=u₂)", "h₂|₁ = P(U₂≤u₂ | U₁=u₁)"],
        vertical_spacing=0.08,
    )

    for row, (h_vals, name, color) in enumerate([
        (h12_series, "h₁|₂", CYAN),
        (h21_series, "h₂|₁", MAGENTA),
    ], 1):
        fig.add_trace(go.Scatter(
            x=timestamps, y=h_vals, name=name, mode="lines",
            line=dict(color=color, width=1.8),
        ), row=row, col=1)

        fig.add_hline(y=entry, line=dict(color=RED, width=1, dash="dash"),
                      annotation_text=f"α₁={entry}", row=row, col=1)
        fig.add_hline(y=1 - entry, line=dict(color=GREEN, width=1, dash="dash"),
                      annotation_text=f"1-α₁={1-entry:.2f}", row=row, col=1)
        fig.add_hrect(y0=0.5 - exit_thr, y1=0.5 + exit_thr,
                      fillcolor=ZONE_CLOSE, line_width=0,
                      annotation_text="close zone", row=row, col=1)
        fig.add_hline(y=0.5, line=dict(color=SUBTEXT, width=0.5, dash="dot"),
                      row=row, col=1)

    sig = np.asarray(signals)
    ts = np.asarray(timestamps)
    for mask, color, name, symbol in [
        (sig == 1, GREEN, "Long C1", "triangle-up"),
        (sig == -1, RED, "Short C1", "triangle-down"),
    ]:
        if mask.any():
            fig.add_trace(go.Scatter(
                x=ts[mask], y=h12_series[mask],
                mode="markers", name=name,
                marker=dict(size=9, color=color, symbol=symbol,
                            line=dict(width=1, color="white")),
                showlegend=True,
            ), row=1, col=1)

    title = f"⬡  H-FUNCTIONS — {pair_label}" if pair_label else "⬡  H-FUNCTIONS"

    # For subplots: only keep keys that don't clash with make_subplots axes
    base = {k: v for k, v in TRON_LAYOUT.items()
            if k not in ("xaxis", "yaxis", "legend", "margin")}
    fig.update_layout(**base, height=480, title=title,
                      margin=dict(l=35, r=25, t=55, b=35),
                      legend=dict(x=1.02, y=1, font=dict(size=9)))
    fig.update_yaxes(range=[-0.02, 1.02])
    return fig


# ═════════════════════════════════════════════════════════════════════════
# 3. SPREAD + TRADES TIME SERIES
# ═════════════════════════════════════════════════════════════════════════

def fig_spread_with_trades(
    spread1_form: pd.Series,
    spread2_form: pd.Series,
    spread1_trade: pd.Series,
    spread2_trade: pd.Series,
    trade_entries: List[Dict],
    trade_exits: List[Dict],
    pair_label: str = "",
    beta1: float = np.nan,
    beta2: float = np.nan,
) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=["Spread₁ = BTC − β₁·P₁", "Spread₂ = BTC − β₂·P₂"],
        vertical_spacing=0.08,
    )

    for row, (s_form, s_trade, color, beta, label) in enumerate([
        (spread1_form, spread1_trade, CYAN, beta1, "S₁"),
        (spread2_form, spread2_trade, MAGENTA, beta2, "S₂"),
    ], 1):
        if s_form is not None and len(s_form) > 0:
            fig.add_trace(go.Scatter(
                x=s_form.index, y=s_form.values,
                name=f"{label} (formation)", mode="lines",
                line=dict(color=color, width=1.2, dash="dot"), opacity=0.5,
            ), row=row, col=1)

        if s_trade is not None and len(s_trade) > 0:
            fig.add_trace(go.Scatter(
                x=s_trade.index, y=s_trade.values,
                name=f"{label} (trading)", mode="lines",
                line=dict(color=color, width=2),
            ), row=row, col=1)

            if s_form is not None and len(s_form) > 0:
                fig.add_vline(
                    x=s_trade.index[0], line=dict(color=GOLD, width=1.5, dash="dash"),
                    row=row, col=1,
                )

    for entry_d in trade_entries:
        t = pd.to_datetime(entry_d.get("time"))
        direction = entry_d.get("direction", "")
        color = GREEN if "LONG1" in direction else RED
        label = "▲ LONG" if "LONG1" in direction else "▼ SHORT"
        fig.add_vline(x=t, line=dict(color=color, width=1.5), row="all", col=1)
        fig.add_annotation(
            x=t, y=1.02, yref="paper", text=label, showarrow=False,
            font=dict(color=color, size=9, family="Share Tech Mono"),
        )

    for exit_d in trade_exits:
        t = pd.to_datetime(exit_d.get("time"))
        fig.add_vline(x=t, line=dict(color=GOLD, width=1, dash="dot"), row="all", col=1)

    title = f"⬡  SPREADS — {pair_label}" if pair_label else "⬡  SPREADS"
    if np.isfinite(beta1) and np.isfinite(beta2):
        title += f"  (β₁={beta1:.2f}, β₂={beta2:.2f})"

    base = {k: v for k, v in TRON_LAYOUT.items()
            if k not in ("xaxis", "yaxis", "legend", "margin")}
    fig.update_layout(**base, height=420, title=title,
                      margin=dict(l=35, r=25, t=55, b=35))
    return fig


# ═════════════════════════════════════════════════════════════════════════
# 4. CYCLE SUMMARY CARD
# ═════════════════════════════════════════════════════════════════════════

def fig_cycle_summary_card(
    cycle_id: int,
    pair: str,
    copula_name: str,
    beta1: float,
    beta2: float,
    q1: float,
    q2: float,
    n_trades: int,
    cycle_pnl: float,
    entry_threshold: float,
    exit_threshold: float,
    status: str = "OK",
) -> go.Figure:
    fig = go.Figure()

    lines = [
        f"<b>Cycle {cycle_id}</b>     Status: <b>{status}</b>",
        f"Pair: <b>{pair}</b>     Copula: <b>{copula_name}</b>",
        f"β₁ = {beta1:.4f}     β₂ = {beta2:.4f}",
        f"Q₁ = {q1:.2f}     Q₂ = {q2:.2f}",
        f"Trades: <b>{n_trades}</b>     PnL: <b style='color:{GREEN if cycle_pnl >= 0 else RED}'>{cycle_pnl:+,.0f} USDT</b>",
        f"Entry α₁ = {entry_threshold}     Exit α₂ = {exit_threshold}",
    ]
    text = "<br>".join(lines)

    fig.add_annotation(
        x=0.5, y=0.5, xref="paper", yref="paper",
        text=text, showarrow=False,
        font=dict(family="Share Tech Mono", size=13, color="#c8dce8"),
        align="left",
        bordercolor=CYAN, borderwidth=1, borderpad=16,
        bgcolor="rgba(10,22,40,0.95)",
    )

    fig.update_layout(**_tron_layout(
        height=200,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=10, b=10),
    ))
    return fig