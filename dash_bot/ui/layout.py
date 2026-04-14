# dash_bot/ui/layout.py
from __future__ import annotations

from typing import Dict, List
import pandas as pd

import dash_bootstrap_components as dbc
from dash import dcc, html

from dash_bot.config import (
    DEFAULT_USDT_SYMBOLS,
    DEFAULT_USDT_SYMBOLS_EXT,
    DEFAULT_REFERENCE,
    INTERVALS,
    STRATEGIES,
    RANK_METHODS,
    COPULA_PICK,
    COINTEGRATION_TESTS,
)

# Copulas list (depuis core)
from ..core.copula_engine import _get_all_copula_values


def _opts(vals: List[str]) -> List[Dict[str, str]]:
    return [{"label": v, "value": v} for v in vals]


ALL_COPULAS = _get_all_copula_values()

def build_layout():
    return dbc.Container(
        fluid=True,
        children=[
            # HEADER
            html.Div("Copula Trading Bot — Backtest Dashboard", className="tron-title"),
            html.Div(
                "REAL DATA BACKTEST  ·  CSV / YFINANCE / BINANCE  ·  DYNAMIC PAIR SELECTION  ·  TRANSACTION COSTS",
                className="tron-subtitle",
            ),

            dbc.Row(
                [
                    # SIDEBAR
                    dbc.Col(
                        width=3,
                        children=[
                            dbc.Card(
                                [
                                    dbc.CardHeader("⬡  PARAMETERS"),
                                    dbc.CardBody(
                                        [
                                            html.Label("Source de données"),
                                            dcc.Dropdown(
                                                id="data-source",
                                                options=[
                                                    {"label": "🟢 Binance SPOT — recommandé (cache local)", "value": "binance"},
                                                    {"label": "📁 CSV locaux (DataAnalysis/data/raw/)", "value": "csv"},
                                                ],
                                                value="binance",
                                                clearable=False,
                                            ),
                                            html.Div(id="data-source-warning", style={"marginTop": "6px", "color": "#ff6f1a"}),

                                            html.Hr(),

                                            html.Label("Strategy"),
                                            dcc.Dropdown(
                                                id="strategy",
                                                options=[{"label": lbl, "value": v} for v, lbl in STRATEGIES],
                                                value="reference_copula",
                                                clearable=False,
                                            ),

                                            html.Label("Interval"),
                                            dcc.Dropdown(
                                                id="interval",
                                                options=[{"label": a, "value": b} for a, b in INTERVALS],
                                                value="5m",
                                                clearable=False,
                                            ),

                                            html.Label("Start / End"),
                                            dcc.DatePickerRange(
                                                id="date-range",
                                                # start_date=(pd.Timestamp.today() - pd.Timedelta(days=365 * 2)).date(),
                                                # end_date=pd.Timestamp.today().date(),
                                                start_date=pd.Timestamp("2021-01-01").date(),
                                                end_date=pd.Timestamp("2023-01-19").date(),
                                                display_format="YYYY-MM-DD",
                                            ),

                                            html.Hr(),

                                            html.Label("Universe (symbols)"),
                                            dcc.Dropdown(
                                                id="symbols",
                                                options=_opts(sorted(set(DEFAULT_USDT_SYMBOLS_EXT))),
                                                value=DEFAULT_USDT_SYMBOLS,
                                                multi=True,
                                            ),
                                            html.Div(
                                                "Tip: en mode Binance/ccxt, tu peux ajouter plus de symbols.",
                                                className="tip-text",
                                            ),

                                            html.Label("Reference asset"),
                                            dcc.Dropdown(
                                                id="ref-asset",
                                                options=_opts(sorted(set(DEFAULT_USDT_SYMBOLS_EXT))),
                                                value=DEFAULT_REFERENCE,
                                                clearable=False,
                                            ),

                                            html.Hr(),

                                            html.Label("Formation / Trading / Step (weeks)"),
                                            dbc.Row(
                                                [
                                                    dbc.Col(dbc.Input(id="formation-weeks", type="number", value=3, min=1, step=1), width=4),
                                                    dbc.Col(dbc.Input(id="trading-weeks", type="number", value=1, min=1, step=1), width=4),
                                                    dbc.Col(dbc.Input(id="step-weeks", type="number", value=1, min=1, step=1), width=4),
                                                ],
                                                style={"marginBottom": "8px"},
                                            ),

                                            html.Label("ADF significance level (alpha)"),
                                            dbc.Input(id="adf-alpha", type="number", value=0.10, min=0.01, max=0.25, step=0.01),

                                            html.Label("Cointegration test strategy",
                                                       style={"marginTop": "8px"}),
                                            html.Div(
                                                "ADF = paper Strategy 1  ·  KSS = paper Strategy 2",
                                                className="tip-text",
                                            ),
                                            dcc.Dropdown(
                                                id="coint-test",
                                                options=[{"label": lbl, "value": v} for v, lbl in COINTEGRATION_TESTS],
                                                value="adf",
                                                clearable=False,
                                            ),

                                            html.Label("Min obs in formation (N_obs)"),
                                            dbc.Input(id="min-obs", type="number", value=200, min=50, step=10),

                                            html.Label("Min coverage per symbol"),
                                            dbc.Input(id="min-coverage", type="number", value=0.90, min=0.50, max=1.00, step=0.01),

                                            html.Hr(),

                                            html.Label("Ranking method"),
                                            dcc.Dropdown(
                                                id="rank-method",
                                                options=[{"label": lbl, "value": v} for v, lbl in RANK_METHODS],
                                                value="kendall_spread_pair",
                                                clearable=False,
                                            ),

                                            html.Label("Top-K coins (2 => 1 pair/week)"),
                                            dbc.Input(id="top-k", type="number", value=2, min=2, max=50, step=1),

                                            html.Hr(),

                                            html.Label("Copula selection"),
                                            dcc.Dropdown(
                                                id="copula-pick",
                                                options=[{"label": lbl, "value": v} for v, lbl in COPULA_PICK],
                                                value="best_score",
                                                clearable=False,
                                            ),
                                            html.Label("Manual copula (if manual)"),
                                            dcc.Dropdown(
                                                id="copula-manual",
                                                options=_opts(ALL_COPULAS),
                                                value="gaussian",
                                                clearable=False,
                                            ),

                                            dbc.Checklist(
                                                options=[{"label": " Suppress CMLE/fit logs", "value": "suppress"}],
                                                value=["suppress"],
                                                id="suppress-logs",
                                                style={"marginTop": "8px"},
                                            ),

                                            html.Hr(),

                                            html.Label("Entry / Exit thresholds"),
                                            dbc.Row(
                                                [
                                                    dbc.Col(dbc.Input(id="entry", type="number", value=0.20, min=0.01, max=0.49, step=0.01), width=6),
                                                    dbc.Col(dbc.Input(id="exit", type="number", value=0.10, min=0.01, max=0.49, step=0.01), width=6),
                                                ],
                                                style={"marginBottom": "8px"},
                                            ),

                                            dbc.Checklist(
                                                options=[{"label": " Flip on opposite signal", "value": "flip"}],
                                                value=[],
                                                id="flip",
                                            ),

                                            html.Hr(),

                                            html.Label("Capital & fees"),
                                            html.Div(
                                                "Cap per leg (USDT) — β-weighted: notional/leg = β·cap",
                                                className="tip-text", style={"marginTop": "4px"}
                                            ),
                                            dbc.Input(id="cap-per-leg", type="number", value=200000, min=100, step=1000),
                                            html.Div("Initial equity (USDT)", className="tip-text", style={"marginTop": "4px"}),
                                            dbc.Input(id="initial-equity", type="number", value=200000, min=100, step=1000),
                                            html.Div("Fee rate (taker ~ 0.0004)", className="tip-text", style={"marginTop": "4px"}),
                                            dbc.Input(id="fee-rate", type="number", value=0.0004, min=0.0, max=0.01, step=0.0001),

                                            html.Hr(),

                                            html.Label("⚠️ Risk Management / Stop-Loss",
                                                       style={"color": "#ff6f1a", "fontWeight": "bold"}),

                                            dbc.Checklist(
                                                options=[{"label": " Trade Stop-Loss (per-trade max loss)",
                                                          "value": "trade_sl"}],
                                                value=[],
                                                id="use-trade-sl",
                                                style={"marginTop": "4px"},
                                            ),
                                            html.Div("Max loss per trade (% du notionnel 2 legs)",
                                                     className="tip-text"),
                                            dbc.Input(id="trade-sl-pct", type="number", value=0.03, min=0.005, max=0.20,
                                                      step=0.005),

                                            dbc.Checklist(
                                                options=[{"label": " Daily Drawdown Limit", "value": "daily_dd"}],
                                                value=[],
                                                id="use-daily-dd",
                                                style={"marginTop": "6px"},
                                            ),
                                            html.Div("Max daily equity loss (%)", className="tip-text"),
                                            dbc.Input(id="daily-dd-pct", type="number", value=0.02, min=0.005, max=0.10,
                                                      step=0.005),

                                            dbc.Checklist(
                                                options=[{"label": " Max Portfolio Drawdown Stop", "value": "max_dd"}],
                                                value=[],
                                                id="use-max-dd-stop",
                                                style={"marginTop": "6px"},
                                            ),
                                            html.Div("Portfolio DD limit (% depuis HWM)", className="tip-text"),
                                            dbc.Input(id="max-dd-stop-pct", type="number", value=0.15, min=0.03,
                                                      max=0.50, step=0.01),

                                            dbc.Checklist(
                                                options=[{"label": " Trailing Stop on Profit", "value": "trail"}],
                                                value=[],
                                                id="use-trailing-stop",
                                                style={"marginTop": "6px"},
                                            ),
                                            html.Div("Trail % / Activation %", className="tip-text"),
                                            dbc.Row([
                                                dbc.Col(dbc.Input(id="trailing-pct", type="number", value=0.02,
                                                                  min=0.005, max=0.10, step=0.005), width=6),
                                                dbc.Col(dbc.Input(id="trailing-activation", type="number", value=0.01,
                                                                  min=0.005, max=0.10, step=0.005), width=6),
                                            ]),

                                            html.Hr(),

                                            html.Label("⬡ Position closing behaviour",
                                                       style={"fontWeight": "bold"}),
                                            dbc.Checklist(
                                                options=[{
                                                    "label": " Force close at week-end (paper behaviour)",
                                                    "value": "force_close"
                                                }],
                                                value=["force_close"],
                                                id="force-week-end-close",
                                                style={"marginTop": "4px"},
                                            ),
                                            html.Div(
                                                "Décoché : les positions ouvertes continuent "
                                                "au-delà de la semaine — le copula d'origine "
                                                "gère la sortie (multi-slot).",
                                                className="tip-text",
                                                style={"marginBottom": "8px"},
                                            ),

                                            dbc.Button("▶  RUN BACKTEST", id="run-btn", className="tron-run-btn w-100"),
                                            html.Div(id="run-status", className="tron-status", style={"marginTop": "10px"}),
                                        ]
                                    ),
                                ],
                                className="sidebar-panel",
                            ),
                        ],
                    ),

                    # MAIN CONTENT
                    dbc.Col(
                        width=9,
                        children=[
                            dbc.Row(
                                [
                                    dbc.Col(html.Div([html.Div("Total Net Return", className="metric-label"),
                                                     html.Div(id="m-total", children="—", className="metric-value")],
                                                    className="metric-card"), width=2),
                                    dbc.Col(html.Div([html.Div("Annual Net Return", className="metric-label"),
                                                     html.Div(id="m-ann", children="—", className="metric-value")],
                                                    className="metric-card"), width=2),
                                    dbc.Col(html.Div([html.Div("Sharpe Ratio", className="metric-label"),
                                                     html.Div(id="m-sharpe", children="—", className="metric-value")],
                                                    className="metric-card"), width=2),
                                    dbc.Col(html.Div([html.Div("Max Drawdown", className="metric-label"),
                                                     html.Div(id="m-mdd", children="—", className="metric-value",
                                                              style={"color": "#ff3355", "textShadow": "0 0 12px rgba(255,51,85,0.4)"})],
                                                    className="metric-card"), width=2),
                                    dbc.Col(html.Div([html.Div("Total Trades", className="metric-label"),
                                                     html.Div(id="m-trades", children="—", className="metric-value",
                                                              style={"color": "#ff2eed", "textShadow": "0 0 12px rgba(255,46,237,0.4)"})],
                                                    className="metric-card"), width=2),
                                    dbc.Col(html.Div([html.Div("Total Fees", className="metric-label"),
                                                     html.Div(id="m-fees", children="—", className="metric-value",
                                                              style={"color": "#ff6f1a", "textShadow": "0 0 12px rgba(255,111,26,0.4)"})],
                                                    className="metric-card"), width=2),
                                    dbc.Col(
                                        html.Div([
                                            html.Div("SL Triggers", className="metric-label"),
                                            html.Div(
                                                id="m-sl-count",
                                                children="—",
                                                className="metric-value",
                                                style={"color": "#ff6f1a",
                                                       "textShadow": "0 0 12px rgba(255,111,26,0.4)"}
                                            )
                                        ], className="metric-card"),
                                        width=2
                                    ),
                                ],
                                style={"marginBottom": "14px"},
                            ),

                            dcc.Tabs(
                                id="tabs",
                                value="tab-equity",
                                className="tab-container",
                                children=[
                                    dcc.Tab(label="⬡ Equity & Risk", value="tab-equity"),
                                    dcc.Tab(label="⬡ Weekly Selection", value="tab-weekly"),
                                    dcc.Tab(label="⬡ Trades", value="tab-trades"),
                                    dcc.Tab(label="⬡ Copula Stats", value="tab-copulas"),
                                    dcc.Tab(label="⬡ Cycle Diagnostic", value="tab-diagnostic"),
                                ],
                            ),

                            # Cycle selector (shown only on diagnostic tab)
                            html.Div(
                                id="diagnostic-controls",
                                style={"display": "none"},
                                children=[
                                    html.Label("SELECT CYCLE", style={
                                        "fontFamily": "Orbitron, sans-serif", "color": "#00f0ff",
                                        "fontSize": "0.75rem", "letterSpacing": "2px", "marginRight": "12px",
                                    }),
                                    dcc.Dropdown(id="diagnostic-cycle-selector",
                                                 placeholder="Select a trading cycle...",
                                                 style={"width": "450px", "backgroundColor": "#0a1628",
                                                        "color": "#c8dce8", "borderColor": "rgba(0,240,255,0.15)"}),
                                    html.Label("COPULA OVERLAY", style={
                                        "fontFamily": "Orbitron, sans-serif", "color": "#ff2eed",
                                        "fontSize": "0.75rem", "letterSpacing": "2px",
                                        "marginLeft": "20px", "marginRight": "12px",
                                    }),
                                    dcc.Dropdown(id="diagnostic-copula-overlay", placeholder="Best (selected)",
                                                 style={"width": "280px", "backgroundColor": "#0a1628",
                                                        "color": "#c8dce8", "borderColor": "rgba(0,240,255,0.15)"}),
                                ],
                            ),

                            html.Div(id="tab-content", style={"marginTop": "10px"}),

                            dcc.Store(id="store-results"),
                        ],
                    ),
                ]
            ),
        ],
        style={"maxWidth": "1650px", "position": "relative", "zIndex": "1"},
    )