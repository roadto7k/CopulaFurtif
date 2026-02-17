# dash_bot/ui/callbacks.py
from __future__ import annotations

import numpy as np
import pandas as pd

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table, Input, Output, State

from .serialization import serialize_results, _deserialize_series
from ..viz.figures import fig_equity, fig_drawdown, fig_monthly_heatmap, fig_copula_freq

# data (tu dis OK)
from ..data.sources import fetch_prices_yfinance, fetch_prices_binance_ccxt, load_prices_csv
from ..data.cleaning import clean_prices_basic

# core (tu dis OK)
from ..core.params import BacktestParams
from ..core.backtest import backtest_reference_copula
from ..core.metrics import _safe_pct

# Optional flags (si tu les exposes dans tes modules, sinon fallback)
try:
    from ..data.sources import HAS_YFINANCE, HAS_CCXT, DATA_PATH
except Exception:
    HAS_YFINANCE, HAS_CCXT, DATA_PATH = True, True, None

try:
    from ..core.copula_engine import HAS_COPULAFURTIF
except Exception:
    HAS_COPULAFURTIF = False
    
def register_callbacks(app):
    # --- Data source warning (comme avant) ---
    @app.callback(
        Output("data-source-warning", "children"),
        Input("data-source", "value"),
    )
    def data_source_warning(src: str):
        msgs = []
        if src == "csv":
            if not DATA_PATH:
                msgs.append("⚠️ DATA_PATH non trouvé. Renseigne-le ou utilise yfinance/ccxt.")
        if src == "yfinance" and not HAS_YFINANCE:
            msgs.append("⚠️ yfinance non installé: pip install yfinance")
        if src == "binance" and not HAS_CCXT:
            msgs.append("⚠️ ccxt non installé: pip install ccxt")
        if not HAS_COPULAFURTIF:
            msgs.append("⚠️ CopulaFurtif non disponible (optionnel).")
        return html.Div(msgs) if msgs else ""

    # --- RUN BACKTEST (store-results + run-status) ---
    @app.callback(
        Output("store-results", "data"),
        Output("run-status", "children"),
        Input("run-btn", "n_clicks"),
        State("data-source", "value"),
        State("strategy", "value"),
        State("interval", "value"),
        State("date-range", "start_date"),
        State("date-range", "end_date"),
        State("symbols", "value"),
        State("ref-asset", "value"),
        State("formation-weeks", "value"),
        State("trading-weeks", "value"),
        State("step-weeks", "value"),
        State("adf-alpha", "value"),
        State("use-kss", "value"),
        State("min-obs", "value"),
        State("min-coverage", "value"),
        State("rank-method", "value"),
        State("top-k", "value"),
        State("copula-pick", "value"),
        State("copula-manual", "value"),
        State("suppress-logs", "value"),
        State("entry", "value"),
        State("exit", "value"),
        State("flip", "value"),
        State("cap-per-leg", "value"),
        State("initial-equity", "value"),
        State("fee-rate", "value"),
        prevent_initial_call=True,
    )
    def run_backtest(
        n_clicks,
        data_source,
        strategy,
        interval,
        start_date,
        end_date,
        symbols,
        ref_asset,
        formation_weeks,
        trading_weeks,
        step_weeks,
        adf_alpha,
        use_kss,
        min_obs,
        min_coverage,
        rank_method,
        top_k,
        copula_pick,
        copula_manual,
        suppress_logs,
        entry,
        exit,
        flip,
        cap_per_leg,
        initial_equity,
        fee_rate,
    ):
        # ensure ref in universe
        if not symbols or ref_asset not in symbols:
            symbols = (symbols or []) + [ref_asset]

        # fetch data
        fetch_errors = {}
        try:
            if data_source == "yfinance":
                # adapte si ton fetcher a une autre signature
                prices = fetch_prices_yfinance(symbols, interval=interval, lookback_days=3650)
            elif data_source == "binance":
                # adapte si ton fetcher a une autre signature
                prices, fetch_errors = fetch_prices_binance_ccxt(symbols, timeframe=interval, lookback_days=3650)
            else:
                if not DATA_PATH:
                    raise RuntimeError("DATA_PATH manquant pour mode CSV.")
                prices = load_prices_csv(DATA_PATH)
                # slice dates
                prices = prices.loc[(prices.index >= pd.to_datetime(start_date)) & (prices.index < pd.to_datetime(end_date))]
                # resample best effort
                if interval != "1h":
                    rule_map = {"5m": "5T", "15m": "15T", "1h": "1H", "4h": "4H", "1d": "1D"}
                    rule = rule_map.get(interval)
                    if rule:
                        prices = prices.resample(rule).last()
        except Exception as e:
            return dash.no_update, f"❌ Data load error: {e}"

        # keep only requested
        prices = prices[[c for c in symbols if c in prices.columns]].dropna(how="all")
        if prices.empty:
            return dash.no_update, "❌ No price data returned for selected symbols/time range."

        # clean
        prices = clean_prices_basic(prices)

        # params (tu dis que tu as remis les bons args dans ton core)
        p = BacktestParams(
            strategy=strategy,
            interval=interval,
            start=str(start_date),
            end=str(end_date),
            ref=str(ref_asset),
            symbols=list(dict.fromkeys([s for s in symbols if s in prices.columns])),
            formation_weeks=int(formation_weeks or 3),
            trading_weeks=int(trading_weeks or 1),
            step_weeks=int(step_weeks or 1),
            adf_alpha=float(adf_alpha or 0.10),
            use_kss=("kss" in (use_kss or [])),
            min_obs=int(min_obs or 200),
            min_coverage=float(min_coverage or 0.90),
            suppress_fit_logs=("suppress" in (suppress_logs or ["suppress"])),
            rank_method=str(rank_method),
            top_k=int(top_k or 2),
            copula_pick=str(copula_pick),
            copula_manual=str(copula_manual),
            entry=float(entry or 0.10),
            exit=float(exit or 0.10),
            flip_on_opposite=("flip" in (flip or [])),
            cap_per_leg=float(cap_per_leg or 20000.0),
            initial_equity=float(initial_equity or 40000.0),
            fee_rate=float(fee_rate or 0.0004),
        )

        if getattr(p, "strategy", None) != "reference_copula":
            return dash.no_update, "⚠️ Pour l’instant, ce dashboard backteste uniquement 'reference_copula'."

        try:
            res = backtest_reference_copula(prices, p)
        except Exception as e:
            return dash.no_update, f"❌ Backtest error: {e}"

        payload = serialize_results(res)

        warns = []
        if fetch_errors:
            bad = ", ".join(list(fetch_errors.keys())[:8])
            extra = "" if len(fetch_errors) <= 8 else f" (+{len(fetch_errors)-8})"
            warns.append(f"⚠️ fetch issues: {bad}{extra}")

        status = "✅ Backtest terminé."
        if warns:
            status = status + "  " + "  |  ".join(warns)

        return payload, status

    # --- Metrics row (comme ton code) ---
    @app.callback(
        Output("m-total", "children"),
        Output("m-ann", "children"),
        Output("m-sharpe", "children"),
        Output("m-mdd", "children"),
        Output("m-trades", "children"),
        Output("m-fees", "children"),
        Input("store-results", "data"),
    )
    def update_metrics(store):
        if not store:
            return ("—", "—", "—", "—", "—", "—")

        m = store.get("metrics", {}) or {}
        tot = m.get("total_return", np.nan)
        ann = m.get("annual_return", np.nan)
        sh = m.get("sharpe", np.nan)
        mdd = m.get("max_drawdown", np.nan)

        trades = store.get("trades", []) or []
        fees = 0.0
        if trades:
            try:
                fees = float(np.nansum([t.get("fees", 0.0) for t in trades]))
            except Exception:
                fees = np.nan

        def f(x, pct=False):
            if x is None or not np.isfinite(x):
                return "—"
            return f"{_safe_pct(x):.1f}%" if pct else f"{x:.2f}"

        return (
            f(tot, pct=True),
            f(ann, pct=True),
            f(sh, pct=False),
            f(mdd, pct=True),
            str(len(trades) if trades else 0),
            (f"{fees:,.0f}" if np.isfinite(fees) else "—"),
        )

    # --- Tabs render (reprend ton code) ---
    @app.callback(
        Output("tab-content", "children"),
        Input("tabs", "value"),
        Input("store-results", "data"),
    )
    def render_tab(tab, store):
        if not store:
            return dbc.Alert(
                "▶  Click 'RUN BACKTEST' to launch the simulation.",
                className="alert-info",
                style={"fontFamily": "Share Tech Mono", "letterSpacing": "1px"},
            )

        equity = _deserialize_series(store.get("equity"))
        equity_g = _deserialize_series(store.get("equity_gross"))
        trades = pd.DataFrame(store.get("trades", []))
        weekly = pd.DataFrame(store.get("weekly", []))
        copfreq = pd.DataFrame(store.get("copula_freq", []))
        monthly = _deserialize_series(store.get("monthly_returns", {"index": [], "values": []}))

        tron_cell = {
            "backgroundColor": "#0a1628",
            "color": "#c8dce8",
            "border": "1px solid rgba(0,240,255,0.08)",
            "fontFamily": "Share Tech Mono, monospace",
            "fontSize": "11px",
            "padding": "6px 8px",
        }
        tron_header = {
            "backgroundColor": "#0d1f3c",
            "color": "#00f0ff",
            "fontWeight": "600",
            "fontFamily": "Orbitron, sans-serif",
            "fontSize": "10px",
            "letterSpacing": "1px",
            "textTransform": "uppercase",
            "border": "1px solid rgba(0,240,255,0.12)",
            "padding": "8px",
        }
        tron_data_cond = [
            {"if": {"state": "active"}, "backgroundColor": "rgba(0,240,255,0.08)", "border": "1px solid rgba(0,240,255,0.3)"},
            {"if": {"state": "selected"}, "backgroundColor": "rgba(0,240,255,0.06)", "border": "1px solid rgba(0,240,255,0.25)"},
        ]

        if tab == "tab-equity":
            return dbc.Row(
                [
                    dbc.Col(dcc.Graph(figure=fig_equity(equity, equity_g), style={"height": "420px"}, config={"responsive": True}), width=8),
                    dbc.Col(
                        [
                            dcc.Graph(figure=fig_drawdown(equity), style={"height": "300px"}, config={"responsive": True}),
                            dcc.Graph(figure=fig_monthly_heatmap(monthly), style={"height": "360px"}, config={"responsive": True}),
                        ],
                        width=4,
                    ),
                ]
            )

        if tab == "tab-weekly":
            cols = [{"name": c, "id": c} for c in weekly.columns] if not weekly.empty else []
            return dbc.Card(
                dbc.CardBody(
                    [
                        html.H4("⬡  Weekly Selection (Formation/Trading Cycles)"),
                        dash_table.DataTable(
                            data=weekly.to_dict("records") if not weekly.empty else [],
                            columns=cols,
                            page_size=12,
                            style_table={"overflowX": "auto"},
                            style_cell=tron_cell,
                            style_header=tron_header,
                            style_data_conditional=tron_data_cond,
                        ),
                    ]
                )
            )

        if tab == "tab-trades":
            cols = [{"name": c, "id": c} for c in trades.columns] if not trades.empty else []
            return dbc.Card(
                dbc.CardBody(
                    [
                        html.H4("⬡  Trades Log"),
                        dash_table.DataTable(
                            data=trades.to_dict("records") if not trades.empty else [],
                            columns=cols,
                            page_size=12,
                            sort_action="native",
                            filter_action="native",
                            style_table={"overflowX": "auto"},
                            style_cell=tron_cell,
                            style_header=tron_header,
                            style_data_conditional=tron_data_cond,
                        ),
                    ]
                )
            )

        if tab == "tab-copulas":
            return dbc.Row(
                [
                    dbc.Col(dcc.Graph(figure=fig_copula_freq(copfreq), style={"height": "380px"}, config={"responsive": True}), width=6),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4("⬡  System Diagnostics"),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Span("COPULA BACKEND: ", style={"color": "#5a7a90", "fontFamily": "Share Tech Mono", "fontSize": "0.75rem"}),
                                                    html.Span(
                                                        f"{'CopulaFurtif' if HAS_COPULAFURTIF else 'Not installed'}",
                                                        style={"color": "#00f0ff", "fontFamily": "Share Tech Mono", "fontSize": "0.75rem"},
                                                    ),
                                                ],
                                                style={"marginBottom": "8px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Span("CYCLES: ", style={"color": "#5a7a90", "fontFamily": "Share Tech Mono", "fontSize": "0.75rem"}),
                                                    html.Span(f"{len(weekly)}", style={"color": "#ff2eed", "fontFamily": "Orbitron", "fontSize": "0.85rem"}),
                                                ],
                                                style={"marginBottom": "8px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Span("TRADES: ", style={"color": "#5a7a90", "fontFamily": "Share Tech Mono", "fontSize": "0.75rem"}),
                                                    html.Span(f"{len(trades)}", style={"color": "#00ff88", "fontFamily": "Orbitron", "fontSize": "0.85rem"}),
                                                ],
                                                style={"marginBottom": "8px"},
                                            ),
                                            html.Div(
                                                "Positions are force-closed at the end of each trading week.",
                                                style={
                                                    "color": "#5a7a90",
                                                    "fontFamily": "Share Tech Mono",
                                                    "fontSize": "0.7rem",
                                                    "marginTop": "14px",
                                                    "borderTop": "1px solid rgba(0,240,255,0.1)",
                                                    "paddingTop": "10px",
                                                },
                                            ),
                                        ],
                                        style={"padding": "8px 0"},
                                    ),
                                ]
                            )
                        ),
                        width=6,
                    ),
                ]
            )

        return dbc.Alert("Unknown tab", color="warning")