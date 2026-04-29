# dash_bot/ui/callbacks.py
from __future__ import annotations

import numpy as np
import pandas as pd

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table, Input, Output, State

from .serialization import serialize_results, _deserialize_series
from ..viz.figures import fig_equity, fig_drawdown, fig_monthly_heatmap, fig_copula_freq, compute_copula_stats, fig_empty
from ..viz.figures_diagnostic import (
    fig_copula_decision_map,
    fig_h_functions_timeseries,
    fig_spread_with_trades,
    fig_cycle_summary_card,
)

from ..viz.figures_diagnostic import (
    fig_copula_decision_map,
    fig_h_functions_timeseries,
    fig_spread_with_trades,
    fig_cycle_summary_card,
    fig_pseudo_obs_scatter,
    fig_fit_ranking,
    fig_tail_dependence,
)

# data (tu dis OK)
from ..data.sources import fetch_prices_cached, load_prices_csv, load_article_5min, ARTICLE_5MIN_META
from ..data.cleaning import clean_prices_basic

# core (tu dis OK)
from ..core.params import BacktestParams
from ..core.backtest import backtest_reference_copula
from ..core.metrics import _safe_pct

# Optional flags (si tu les exposes dans tes modules, sinon fallback)
try:
    from ..data.sources import HAS_YFINANCE, HAS_CCXT, DATA_PATH
except Exception:
    HAS_YFINANCE, HAS_CCXT, DATA_PATH = False, False, None

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
                msgs.append("⚠️ DATA_PATH non trouvé. Renseigne-le ou passe sur Binance.")
        if src == "binance":
            msgs.append(
                "ℹ️ Binance SPOT — API REST directe, cache local. Seules les périodes manquantes sont retéléchargées.")
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
        State("coint-test", "value"),
        State("min-obs", "value"),
        State("min-coverage", "value"),
        State("copula-pick", "value"),
        State("copula-manual", "value"),
        State("suppress-logs", "value"),
        State("entry", "value"),
        State("exit", "value"),
        State("flip", "value"),
        State("cap-per-leg", "value"),
        State("initial-equity", "value"),
        State("fee-rate", "value"),
        State("use-trade-sl", "value"),
        State("trade-sl-pct", "value"),
        State("use-daily-dd", "value"),
        State("daily-dd-pct", "value"),
        State("use-max-dd-stop", "value"),
        State("max-dd-stop-pct", "value"),
        State("use-trailing-stop", "value"),
        State("trailing-pct", "value"),
        State("trailing-activation", "value"),
        State("force-week-end-close", "value"),
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
            cointegration_test,
            min_obs,
            min_coverage,
            copula_pick,
            copula_manual,
            suppress_logs,
            entry,
            exit,
            flip,
            cap_per_leg,
            initial_equity,
            fee_rate,
            use_trade_sl,
            trade_sl_pct,
            use_daily_dd,
            daily_dd_pct,
            use_max_dd_stop,
            max_dd_stop_pct,
            use_trailing_stop,
            trailing_pct,
            trailing_activation,
            force_week_end_close,
    ):
        # ensure ref in universe
        if not symbols or ref_asset not in symbols:
            symbols = (symbols or []) + [ref_asset]

        # fetch data
        fetch_errors = {}
        try:
            if data_source == "binance":
                prices, fetch_errors = fetch_prices_cached(
                    symbols, start_date, end_date, interval, source="binance"
                )
            elif data_source == "article_5min":
                prices, fetch_errors = load_article_5min()
                # override tous les params UI par ceux de l'article
                interval = "5m"
                start_date = ARTICLE_5MIN_META["start"]
                end_date = ARTICLE_5MIN_META["end"]
                ref_asset = ARTICLE_5MIN_META["ref"]
                symbols = ARTICLE_5MIN_META["symbols"]
            else:  # csv
                if not DATA_PATH:
                    raise RuntimeError("DATA_PATH manquant pour mode CSV.")
                prices = load_prices_csv(DATA_PATH)
                prices = prices.loc[
                    (prices.index >= pd.to_datetime(start_date)) &
                    (prices.index < pd.to_datetime(end_date))]
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
            cointegration_test=str(cointegration_test or "adf"),
            min_obs=int(min_obs or 200),
            min_coverage=float(min_coverage or 0.90),
            suppress_fit_logs=("suppress" in (suppress_logs or ["suppress"])),
            copula_pick=str(copula_pick),
            copula_manual=str(copula_manual),
            entry=float(entry or 0.10),
            exit=float(exit or 0.10),
            flip_on_opposite=("flip" in (flip or [])),
            cap_per_leg=float(cap_per_leg or 20000.0),
            initial_equity=float(initial_equity or 40000.0),
            fee_rate=float(fee_rate or 0.0004),
            use_trade_stop_loss=("trade_sl" in (use_trade_sl or [])),
            trade_stop_loss_pct=float(trade_sl_pct or 0.03),
            use_daily_drawdown_limit=("daily_dd" in (use_daily_dd or [])),
            daily_drawdown_limit_pct=float(daily_dd_pct or 0.02),
            use_max_drawdown_stop=("max_dd" in (use_max_dd_stop or [])),
            max_drawdown_stop_pct=float(max_dd_stop_pct or 0.15),
            use_trailing_stop=("trail" in (use_trailing_stop or [])),
            trailing_stop_pct=float(trailing_pct or 0.02),
            trailing_stop_activation=float(trailing_activation or 0.01),
            force_week_end_close=("force_close" in (force_week_end_close or [])),
        )

        if getattr(p, "strategy", None) != "reference_copula":
            return dash.no_update, "⚠️ Pour l'instant, ce dashboard backteste uniquement 'reference_copula'."

        try:
            res = backtest_reference_copula(prices, p)
        except Exception as e:
            return dash.no_update, f"❌ Backtest error: {e}"

        payload = serialize_results(res)

        warns = []
        if fetch_errors:
            bad = ", ".join(list(fetch_errors.keys())[:8])
            extra = "" if len(fetch_errors) <= 8 else f" (+{len(fetch_errors) - 8})"
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
        Output("m-sl-count", "children"),  # <-- AJOUT ICI
        Input("store-results", "data"),
    )
    def update_metrics(store):
        if not store:
            return ("—", "—", "—", "—", "—", "—", "—")

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

        sl_stats = store.get("stop_loss_stats", {})
        sl_count = sl_stats.get("total_stop_loss_trades", 0)

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
            str(sl_count),
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
            {"if": {"state": "active"}, "backgroundColor": "rgba(0,240,255,0.08)",
             "border": "1px solid rgba(0,240,255,0.3)"},
            {"if": {"state": "selected"}, "backgroundColor": "rgba(0,240,255,0.06)",
             "border": "1px solid rgba(0,240,255,0.25)"},
        ]

        if tab == "tab-equity":
            return dbc.Row(
                [
                    dbc.Col(dcc.Graph(figure=fig_equity(equity, equity_g), style={"height": "420px"},
                                      config={"responsive": True}), width=8),
                    dbc.Col(
                        [
                            dcc.Graph(figure=fig_drawdown(equity), style={"height": "300px"},
                                      config={"responsive": True}),
                            dcc.Graph(figure=fig_monthly_heatmap(monthly), style={"height": "360px"},
                                      config={"responsive": True}),
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
            cop_stats = compute_copula_stats(trades)
            cop_stats_cols = (
                [{"name": c, "id": c} for c in cop_stats.columns]
                if not cop_stats.empty else []
            )
            # Conditional formatting: green for positive PnL cells, red for negative
            pnl_cond = []
            for col in ["Avg PnL", "Total PnL", "Best", "Worst", "Avg Win", "Avg Loss"]:
                pnl_cond += [
                    {
                        "if": {"filter_query": f"{{{col}}} contains '+'", "column_id": col},
                        "color": "#00ff88",
                    },
                    {
                        "if": {"filter_query": f"{{{col}}} contains '-'", "column_id": col},
                        "color": "#ff3355",
                    },
                ]

            return dbc.Col(
                [
                    # Row 1 : frequency chart + diagnostics
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Graph(
                                    figure=fig_copula_freq(copfreq),
                                    style={"height": "340px"},
                                    config={"responsive": True},
                                ),
                                width=7,
                            ),
                            dbc.Col(
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.H5(
                                                "⬡  SYSTEM INFO",
                                                style={
                                                    "fontFamily": "Orbitron",
                                                    "color": "#00f0ff",
                                                    "fontSize": "0.8rem",
                                                    "letterSpacing": "2px",
                                                    "marginBottom": "16px",
                                                },
                                            ),
                                            html.Div(
                                                [
                                                    html.Span("BACKEND: ", style={"color": "#5a7a90", "fontFamily": "Share Tech Mono", "fontSize": "0.75rem"}),
                                                    html.Span(f"{'CopulaFurtif' if HAS_COPULAFURTIF else 'Not installed'}", style={"color": "#00f0ff", "fontFamily": "Share Tech Mono", "fontSize": "0.75rem"}),
                                                ],
                                                style={"marginBottom": "10px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Span("CYCLES: ", style={"color": "#5a7a90", "fontFamily": "Share Tech Mono", "fontSize": "0.75rem"}),
                                                    html.Span(f"{len(weekly)}", style={"color": "#ff2eed", "fontFamily": "Orbitron", "fontSize": "0.85rem"}),
                                                ],
                                                style={"marginBottom": "10px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Span("TRADES: ", style={"color": "#5a7a90", "fontFamily": "Share Tech Mono", "fontSize": "0.75rem"}),
                                                    html.Span(f"{len(trades)}", style={"color": "#00ff88", "fontFamily": "Orbitron", "fontSize": "0.85rem"}),
                                                ],
                                                style={"marginBottom": "10px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Span("UNIQUE COPULAS: ", style={"color": "#5a7a90", "fontFamily": "Share Tech Mono", "fontSize": "0.75rem"}),
                                                    html.Span(
                                                        f"{len(cop_stats)}" if not cop_stats.empty else "0",
                                                        style={"color": "#ffe01a", "fontFamily": "Orbitron", "fontSize": "0.85rem"},
                                                    ),
                                                ],
                                                style={"marginBottom": "10px"},
                                            ),
                                        ]
                                    ),
                                    style={"backgroundColor": "#0a1628", "border": "1px solid rgba(0,240,255,0.1)"},
                                ),
                                width=5,
                            ),
                        ],
                        className="mb-3",
                    ),
                    # Row 2 : per-copula stats table (full width)
                    dbc.Row(
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H5(
                                            "⬡  PERFORMANCE BY COPULA",
                                            style={
                                                "fontFamily": "Orbitron",
                                                "color": "#00f0ff",
                                                "fontSize": "0.8rem",
                                                "letterSpacing": "2px",
                                                "marginBottom": "12px",
                                            },
                                        ),
                                        dash_table.DataTable(
                                            data=cop_stats.to_dict("records") if not cop_stats.empty else [],
                                            columns=cop_stats_cols,
                                            page_size=20,
                                            sort_action="native",
                                            style_table={"overflowX": "auto"},
                                            style_cell={
                                                **tron_cell,
                                                "textAlign": "center",
                                                "minWidth": "70px",
                                            },
                                            style_cell_conditional=[
                                                {"if": {"column_id": "Copula"}, "textAlign": "left", "fontWeight": "600", "color": "#ff2eed"},
                                            ],
                                            style_header=tron_header,
                                            style_data_conditional=tron_data_cond + pnl_cond,
                                        ) if not cop_stats.empty else html.Div(
                                            "No trade data available for copula breakdown.",
                                            style={"color": "#5a7a90", "fontFamily": "Share Tech Mono", "fontSize": "0.8rem", "padding": "12px"},
                                        ),
                                    ]
                                ),
                                style={"backgroundColor": "#0a1628", "border": "1px solid rgba(0,240,255,0.1)"},
                            ),
                            width=12,
                        )
                    ),
                ]
            )

        if tab == "tab-diagnostic":
            return html.Div(
                "⬡  Select a cycle from the dropdown above to inspect its copula, signals, and trades.",
                style={"color": "#5a7a90", "fontFamily": "Share Tech Mono", "padding": "40px",
                       "textAlign": "center", "fontSize": "1rem"},
            )

        return dbc.Alert("Unknown tab", color="warning")

    # ── Toggle diagnostic controls visibility ──────────────────────
    @app.callback(
        Output("diagnostic-controls", "style"),
        Input("tabs", "value"),
    )
    def toggle_diag_controls(tab):
        if tab == "tab-diagnostic":
            return {"display": "flex", "alignItems": "center", "gap": "12px",
                    "marginTop": "10px", "marginBottom": "10px",
                    "padding": "8px 16px", "backgroundColor": "rgba(10,22,40,0.6)",
                    "borderRadius": "6px", "border": "1px solid rgba(0,240,255,0.08)"}
        return {"display": "none"}

    # ── Populate cycle selector ────────────────────────────────────
    @app.callback(
        Output("diagnostic-cycle-selector", "options"),
        Input("store-results", "data"),
    )
    def populate_cycle_selector(store):
        if not store:
            return []
        weekly = store.get("weekly", [])
        options = []
        for w in weekly:
            cycle = w.get("cycle", "?")
            pair = w.get("selected_pair") or "—"
            status = w.get("status", "")
            cop = w.get("copula") or "—"
            label = f"Cycle {cycle} │ {pair} │ {cop} │ {status}"
            options.append({"label": label, "value": cycle})
        return options

    # ── Render diagnostic detail for selected cycle ────────────────
    @app.callback(
        Output("tab-content", "children", allow_duplicate=True),
        Input("diagnostic-cycle-selector", "value"),
        Input("diagnostic-copula-overlay", "value"),
        State("store-results", "data"),
        State("tabs", "value"),
        prevent_initial_call=True,
    )
    def render_diagnostic_detail(selected_cycle, overlay_copula_name, store, current_tab):
        if current_tab != "tab-diagnostic" or selected_cycle is None or not store:
            raise dash.exceptions.PreventUpdate

        weekly = store.get("weekly", [])
        trades_all = pd.DataFrame(store.get("trades", []))

        # Find the selected cycle data
        cycle_data = None
        for w in weekly:
            if w.get("cycle") == selected_cycle:
                cycle_data = w
                break

        if cycle_data is None:
            return dbc.Alert(f"Cycle {selected_cycle}: no data found.", className="alert-warning")
        if cycle_data.get("status") != "OK":
            return dbc.Alert(
                f"Cycle {selected_cycle}: {cycle_data.get('status', 'no data')} — no diagnostic available.",
                className="alert-warning",
            )

        # Extract diagnostic bars
        diag_bars = cycle_data.get("diag_bars", [])
        if not diag_bars:
            return dbc.Alert("No diagnostic data for this cycle (no bars recorded).", className="alert-info")

        diag_df = pd.DataFrame(diag_bars)
        diag_df["ts"] = pd.to_datetime(diag_df["ts"])

        pair = cycle_data.get("selected_pair", "?")
        cop_name = cycle_data.get("copula", "?")
        beta1 = cycle_data.get("beta1", float("nan"))
        beta2 = cycle_data.get("beta2", float("nan"))
        q1 = cycle_data.get("q1", 0)
        q2 = cycle_data.get("q2", 0)
        cop_rotation = int(cycle_data.get("cop_rotation", 0))
        cop_params_list = cycle_data.get("cop_params", [])

        # Rebuild copula for the decision map
        cop = None
        try:
            from ..core.copula_engine import build_copula
            cop_type_str = cop_name.split(" R")[0] if " R" in str(cop_name) else str(cop_name)
            params = np.array(cop_params_list, dtype=float)
            cop = build_copula(cop_type_str, params)
            if cop_rotation != 0:
                from CopulaFurtif.core.copulas.adapters import RotatedCopula
                cop = RotatedCopula(cop, cop_rotation)
        except Exception as e:
            cop = None

        # Cycle trades
        cycle_trades = trades_all[trades_all["cycle"] == selected_cycle] if not trades_all.empty else pd.DataFrame()
        n_trades = len(cycle_trades)
        cycle_pnl = float(cycle_trades["net_pnl"].sum()) if not cycle_trades.empty and "net_pnl" in cycle_trades.columns else 0.0

        # Entry/exit thresholds from stored params
        params_dict = store.get("params", {})
        entry_thr = float(params_dict.get("entry", 0.20))
        exit_thr = float(params_dict.get("exit", 0.10))

        # ── Build figures ──

        # 1. Summary card
        card = fig_cycle_summary_card(
            cycle_id=selected_cycle, pair=pair, copula_name=str(cop_name),
            beta1=float(beta1) if np.isfinite(beta1) else 0.0,
            beta2=float(beta2) if np.isfinite(beta2) else 0.0,
            q1=float(q1), q2=float(q2),
            n_trades=n_trades, cycle_pnl=cycle_pnl,
            entry_threshold=entry_thr, exit_threshold=exit_thr,
        )

        # 2. Decision map
        fig_map = fig_empty("Decision map — copula could not be rebuilt")
        if cop is not None and len(diag_df) > 0:
            try:
                fig_map = fig_copula_decision_map(
                    cop=cop,
                    u_trades=diag_df["u"].values,
                    v_trades=diag_df["v"].values,
                    signals=diag_df["sig"].values,
                    entry=entry_thr, exit_thr=exit_thr,
                    copula_name=str(cop_name), pair_label=pair,
                )
            except Exception:
                pass

        # 3. H-functions time series
        fig_h = fig_h_functions_timeseries(
            timestamps=diag_df["ts"],
            h12_series=diag_df["h12"].values,
            h21_series=diag_df["h21"].values,
            signals=diag_df["sig"].values,
            entry=entry_thr, exit_thr=exit_thr,
            pair_label=pair,
        )

        # 4. Spreads with trade markers
        fig_spreads = fig_spread_with_trades(
            spread1_form=pd.Series(dtype=float),
            spread2_form=pd.Series(dtype=float),
            spread1_trade=pd.Series(diag_df["s1"].values, index=diag_df["ts"]),
            spread2_trade=pd.Series(diag_df["s2"].values, index=diag_df["ts"]),
            trade_entries=[
                {"time": t.get("entry_time"), "direction": t.get("direction", "")}
                for _, t in cycle_trades.iterrows()
            ] if not cycle_trades.empty else [],
            trade_exits=[
                {"time": t.get("exit_time"), "reason": t.get("exit_reason", "")}
                for _, t in cycle_trades.iterrows()
            ] if not cycle_trades.empty else [],
            pair_label=pair,
            beta1=float(beta1) if np.isfinite(beta1) else 0.0,
            beta2=float(beta2) if np.isfinite(beta2) else 0.0,
        )

        # 5. Pseudo-obs scatter with copula overlay
        pseudo_u = cycle_data.get("pseudo_u", [])
        pseudo_v = cycle_data.get("pseudo_v", [])
        fit_summary = cycle_data.get("fit_summary", [])

        # Build overlay copula (from dropdown selection)
        overlay_cop = None
        overlay_name = overlay_copula_name or cop_name
        if fit_summary and overlay_name:
            for fs in fit_summary:
                if fs["name"] == overlay_name and fs.get("evaluable", False):
                    try:
                        from ..core.copula_engine import build_copula
                        rot = int(fs.get("rotation", 0))
                        oc = build_copula(fs["name"].split(" R")[0] if " R" in fs["name"] else fs["name"],
                                          np.array(fs["params"], dtype=float))
                        if rot != 0:
                            from CopulaFurtif.core.copulas.adapters import RotatedCopula
                            oc = RotatedCopula(oc, rot)
                        overlay_cop = oc
                    except Exception:
                        pass
                    break

        fig_scatter = fig_pseudo_obs_scatter(
            pseudo_u=np.array(pseudo_u), pseudo_v=np.array(pseudo_v),
            overlay_cop=overlay_cop, overlay_name=overlay_name,
            selected_name=str(cop_name), pair_label=pair,
        )

        # 6. Fit ranking
        fig_rank = fig_fit_ranking(fit_summary, selected_name=str(cop_name))

        # 7. Tail dependence
        fig_tails = fig_tail_dependence(fit_summary, selected_name=str(cop_name))

        return dbc.Col([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=card, config={"responsive": True}), width=5),
                dbc.Col(dcc.Graph(figure=fig_map, config={"responsive": True}), width=7),
            ], className="mb-2"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_scatter, config={"responsive": True}), width=5),
                dbc.Col(dcc.Graph(figure=fig_rank, config={"responsive": True}), width=7),
            ], className="mb-2"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_h, config={"responsive": True}), width=12),
            ], className="mb-2"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_spreads, config={"responsive": True}), width=8),
                dbc.Col(dcc.Graph(figure=fig_tails, config={"responsive": True}), width=4),
            ]),
        ])
