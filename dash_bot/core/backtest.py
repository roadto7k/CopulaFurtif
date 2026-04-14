# # dash_bot/core/backtest.py
# import pandas as pd
# import numpy as np
# from typing import Dict, Any, List

# from .selection import select_stationary_spreads
# from .metrics import _to_datetime, performance_metrics
# from .selection import rank_coins
# from .copula_engine import fit_pair_copula, build_copula
# from .params import BacktestParams
# from .signals import generate_signals_reference_copula


# # ---------------------------------------------------------------------------
# # Slot helpers
# # ---------------------------------------------------------------------------

# def _serialize_fit_summary(df_fit):
#     """Extract JSON-safe summary from fit_copulas DataFrame for dashboard."""
#     if df_fit is None or df_fit.empty:
#         return []
#     rows = []
#     for _, r in df_fit.iterrows():
#         if not np.isfinite(r.get("aic", np.nan)):
#             continue
#         rows.append(dict(
#             name=str(r.get("name", "?")),
#             rotation=int(r.get("rotation", 0)),
#             aic=float(r["aic"]) if np.isfinite(r.get("aic", np.nan)) else None,
#             loglik=float(r["loglik"]) if np.isfinite(r.get("loglik", np.nan)) else None,
#             kt_err=float(r["kt_err"]) if np.isfinite(r.get("kt_err", np.nan)) else None,
#             tail_gap=float(r["tail_gap"]) if np.isfinite(r.get("tail_gap", np.nan)) else None,
#             score=float(r["score_stage1"]) if "score_stage1" in r and np.isfinite(r.get("score_stage1", np.nan)) else None,
#             tail_dep_L=float(r["tail_dep_L"]) if np.isfinite(r.get("tail_dep_L", np.nan)) else None,
#             tail_dep_U=float(r["tail_dep_U"]) if np.isfinite(r.get("tail_dep_U", np.nan)) else None,
#             params=[float(x) for x in np.atleast_1d(r["params"])] if r.get("params") is not None and len(np.atleast_1d(r["params"])) > 0 else [],
#             evaluable=bool(r.get("evaluable", False)),
#         ))
#     return rows

# def _make_slot(cycle_id, coin1, coin2, cop, best_name,
#                beta1, beta2, s1_sorted, s2_sorted,
#                q1, q2, total_notional):
#     """Crée un slot copula sans position ouverte."""
#     return dict(
#         cycle_id=cycle_id,
#         coin1=coin1, coin2=coin2,
#         best_name=best_name,
#         cop=cop,
#         beta1=beta1, beta2=beta2,
#         s1_sorted=s1_sorted, s2_sorted=s2_sorted,
#         q1=q1, q2=q2,
#         total_notional=total_notional,
#         pos=0,
#         entry_ts=None, entry_p1=None, entry_p2=None,
#         gross_pnl=0.0, net_pnl=0.0, fees_paid=0.0,
#         peak_trade_pnl=0.0,
#         pending_sig=0,
#         pending_close=False,
#         prev_p1=None, prev_p2=None,
#     )


# def _close_slot(slot, close_ts, close_p1, close_p2, reason,
#                 fee_rate, trades, stop_loss_events,
#                 current_equity, current_equity_gross, hwm):
#     q1, q2 = slot["q1"], slot["q2"]
#     notional = abs(q1 * close_p1) + abs(q2 * close_p2)
#     fee = fee_rate * abs(notional)
#     slot["fees_paid"] += fee
#     slot["net_pnl"]   -= fee

#     is_stop = reason not in ("signal", "week_end")

#     trades.append(dict(
#         cycle=slot["cycle_id"],
#         pair=f"{slot['coin1']}-{slot['coin2']}",
#         copula=slot["best_name"],
#         entry_time=str(slot["entry_ts"]),
#         exit_time=str(close_ts),
#         direction="LONG1_SHORT2" if slot["pos"] == +1 else "SHORT1_LONG2",
#         q1=q1, q2=q2,
#         beta1=slot["beta1"], beta2=slot["beta2"],
#         entry_p1=slot["entry_p1"], entry_p2=slot["entry_p2"],
#         exit_p1=close_p1, exit_p2=close_p2,
#         gross_pnl=slot["gross_pnl"],
#         net_pnl=slot["net_pnl"],
#         fees=slot["fees_paid"],
#         bars=int(
#             (pd.to_datetime(close_ts) - pd.to_datetime(slot["entry_ts"])
#              ).total_seconds() // 60
#         ) if slot["entry_ts"] else np.nan,
#         forced_week_end=(reason == "week_end"),
#         exit_reason=reason,
#     ))

#     if is_stop:
#         stop_loss_events.append(dict(
#             time=str(close_ts),
#             type=reason,
#             detail=f"PnL={slot['net_pnl']:.2f}",
#             equity=current_equity + slot["net_pnl"],
#         ))

#     current_equity       += slot["net_pnl"]
#     current_equity_gross += slot["gross_pnl"]
#     if current_equity > hwm:
#         hwm = current_equity

#     slot["pos"] = 0
#     slot["entry_ts"] = slot["entry_p1"] = slot["entry_p2"] = None
#     slot["gross_pnl"] = slot["net_pnl"] = slot["fees_paid"] = 0.0
#     slot["peak_trade_pnl"] = 0.0
#     slot["pending_sig"] = 0
#     slot["pending_close"] = False

#     return current_equity, current_equity_gross, hwm


# # ---------------------------------------------------------------------------
# # Main backtest
# # ---------------------------------------------------------------------------

# def backtest_reference_copula(prices: pd.DataFrame, p: BacktestParams) -> Dict[str, Any]:
#     """
#     Backtest de la strategie proposee (reference spread copula).

#     Architecture multi-slot :
#     - Chaque semaine un nouveau copula est fitte -> nouveau slot cree.
#     - force_week_end_close=True  : comportement papier, toutes les positions
#       fermees en fin de semaine.
#     - force_week_end_close=False : un slot avec position ouverte continue
#       au-dela de sa semaine. Il utilise son propre copula pour les signaux
#       de sortie. Seul le slot courant peut ouvrir de nouvelles positions.
#     - Un slot sans position en fin de cycle est purge (modele stale).
#     """
#     prices = prices.copy()
#     prices = prices.loc[
#         (prices.index >= _to_datetime(p.start)) &
#         (prices.index <  _to_datetime(p.end))
#     ]
#     prices = prices.sort_index()
#     prices = prices[p.symbols].dropna(how="all")

#     if p.ref not in prices.columns:
#         raise ValueError(f"Reference asset '{p.ref}' absent des donnees.")

#     start_dt = prices.index.min()
#     end_dt   = prices.index.max()
#     if start_dt is pd.NaT or end_dt is pd.NaT:
#         raise ValueError("Pas de donnees dans la fenetre demandee.")

#     formation_delta = pd.Timedelta(days=7 * p.formation_weeks)
#     trading_delta   = pd.Timedelta(days=7 * p.trading_weeks)
#     step_delta      = pd.Timedelta(days=7 * p.step_weeks)

#     equity       = pd.Series(index=prices.index, dtype=float)
#     equity_gross = pd.Series(index=prices.index, dtype=float)
#     equity.iloc[:]       = np.nan
#     equity_gross.iloc[:] = np.nan

#     trades:           list = []
#     weekly:           list = []
#     stop_loss_events: list = []

#     current_equity       = float(p.initial_equity)
#     current_equity_gross = float(p.initial_equity)
#     hwm                  = float(p.initial_equity)
#     portfolio_stopped    = False

#     active_slots: List[dict] = []

#     current_day        = None
#     daily_start_equity = None
#     daily_stopped      = False

#     t0       = start_dt
#     cycle_id = 0

#     # Pré-calcul du nombre total de cycles pour la progression
#     _t = start_dt
#     _total_cycles = 0
#     while True:
#         if _t + formation_delta + trading_delta > end_dt:
#             break
#         _total_cycles += 1
#         _t += step_delta

#     while True:
#         t_form_end = t0 + formation_delta
#         t_trade_end = t_form_end + trading_delta
#         if t_trade_end > end_dt:
#             break

#         print(
#             f"[Backtest] Cycle {cycle_id + 1}/{_total_cycles} │ "
#             f"Formation: {t0.strftime('%Y-%m-%d')} → {t_form_end.strftime('%Y-%m-%d')} │ "
#             f"Trading: {t_form_end.strftime('%Y-%m-%d')} → {t_trade_end.strftime('%Y-%m-%d')} │ "
#             f"Equity: {current_equity:,.0f} USDT",
#             flush=True,
#         )

#         # Max drawdown portfolio stop
#         if p.use_max_drawdown_stop and not portfolio_stopped:
#             dd_from_hwm = (current_equity - hwm) / hwm if hwm > 0 else 0.0
#             if dd_from_hwm < -abs(p.max_drawdown_stop_pct):
#                 portfolio_stopped = True
#                 stop_loss_events.append(dict(
#                     time=str(t_form_end),
#                     type="MAX_DRAWDOWN_STOP",
#                     detail=f"DD={dd_from_hwm:.4f} < -{p.max_drawdown_stop_pct:.4f}",
#                     equity=current_equity,
#                 ))

#         df_form  = prices.loc[(prices.index >= t0)         & (prices.index < t_form_end)]
#         df_trade = prices.loc[(prices.index >= t_form_end) & (prices.index < t_trade_end)]

#         # Fit this cycle's copula
#         new_slot    = None
#         skip_status = None
#         coin1 = coin2 = best_name = None
#         beta1 = beta2 = np.nan
#         q1 = q2 = 0.0
#         msgs = []
#         top  = []
#         pseudo_u = []
#         pseudo_v = []
#         df_fit = None

#         if portfolio_stopped:
#             skip_status = "STOPPED (max drawdown limit reached)"
#         else:
#             candidates = [s for s in p.symbols if s != p.ref and s in df_form.columns]
#             ref_ok     = df_form[p.ref].notna()
#             form_slice = df_form.loc[ref_ok, candidates]
#             coverage   = form_slice.notna().mean()
#             points     = form_slice.notna().sum()
#             eligible   = [
#                 s for s in candidates
#                 if float(coverage.get(s, 0.0)) >= float(p.min_coverage)
#                 and int(points.get(s, 0)) >= int(p.min_obs)
#             ]

#             summary, spreads, betas = select_stationary_spreads(
#                 prices=df_form, ref=p.ref, candidates=eligible,
#                 adf_alpha=p.adf_alpha, cointegration_test=p.cointegration_test,
#                 kss_crit=p.kss_crit, min_obs=p.min_obs,
#             )
#             accepted = (
#                 summary[summary["accepted"] == True]["coin"].tolist()
#                 if not summary.empty else []
#             )

#             if len(accepted) < 2 or df_trade.empty:
#                 skip_status = "SKIP (not enough stationary spreads)"
#             else:
#                 rank_df = rank_coins(df_form, spreads, p.ref, accepted, p.rank_method)
#                 top = (
#                     rank_df["coin"].tolist()[:max(2, p.top_k)]
#                     if not rank_df.empty else accepted[:max(2, p.top_k)]
#                 )
#                 if len(top) < 2:
#                     skip_status = "SKIP (ranking failed)"
#                 else:
#                     coin1, coin2 = top[0], top[1]
#                     s1_form = spreads[coin1]
#                     s2_form = spreads[coin2]
#                     beta1   = betas.get(coin1, np.nan)
#                     beta2   = betas.get(coin2, np.nan)

#                     best_name, best_params, df_fit, msgs = fit_pair_copula(
#                         s1_form, s2_form, p.copula_pick, p.copula_manual,
#                         suppress_logs=p.suppress_fit_logs,
#                     )
#                     if best_name is None:
#                         skip_status = "SKIP (copula fit failed)"
#                     else:
#                         cop       = build_copula(best_name, best_params)
#                         s1_sorted = np.sort(s1_form.dropna().values.astype(float))
#                         s2_sorted = np.sort(s2_form.dropna().values.astype(float))

#                         # Pseudo-observations for diagnostic scatter
#                         _s1_aligned, _s2_aligned = s1_form.align(s2_form, join="inner")
#                         _s1_clean = _s1_aligned.dropna()
#                         _s2_clean = _s2_aligned.loc[_s1_clean.index].dropna()
#                         _common = _s1_clean.index.intersection(_s2_clean.index)
#                         _n_po = len(_common)
#                         if _n_po > 10:
#                             from scipy.stats import rankdata
#                             pseudo_u = (rankdata(_s1_clean.loc[_common].values) / (_n_po + 1)).tolist()
#                             pseudo_v = (rankdata(_s2_clean.loc[_common].values) / (_n_po + 1)).tolist()
#                         else:
#                             pseudo_u, pseudo_v = [], []

#                         first_bar = df_trade[[coin1, coin2]].dropna().head(1)
#                         if first_bar.empty:
#                             skip_status = "SKIP (no trade prices)"
#                         else:
#                             p1_open = float(first_bar.iloc[0][coin1])
#                             p2_open = float(first_bar.iloc[0][coin2])

#                             q1 = (p.cap_per_leg / p1_open) if (np.isfinite(p1_open) and p1_open > 0) else 0.0
#                             q2 = (p.cap_per_leg / p2_open) if (np.isfinite(p2_open) and p2_open > 0) else 0.0

#                             new_slot = _make_slot(
#                                 cycle_id=cycle_id,
#                                 coin1=coin1, coin2=coin2,
#                                 cop=cop, best_name=best_name,
#                                 beta1=beta1, beta2=beta2,
#                                 s1_sorted=s1_sorted, s2_sorted=s2_sorted,
#                                 q1=q1, q2=q2,
#                                 total_notional=(q1 * p1_open) + (q2 * p2_open),
#                             )
#                             active_slots.append(new_slot)

#         # ── Diagnostic storage for the current cycle (must be BEFORE weekly.append) ──
#         cycle_diag_bars = []  # will collect per-bar (u, v, h12, h21, sig)

#         # Extract cop metadata for dashboard reconstruction
#         _cop_params = []
#         _cop_rotation = 0
#         _s1_sorted_list = []
#         _s2_sorted_list = []
#         if new_slot is not None:
#             try:
#                 _cop_params = list(float(x) for x in np.atleast_1d(cop.get_parameters()))
#             except Exception:
#                 _cop_params = []
#             _cop_rotation = getattr(cop, 'rotation', 0)
#             _s1_sorted_list = s1_sorted.tolist() if s1_sorted is not None else []
#             _s2_sorted_list = s2_sorted.tolist() if s2_sorted is not None else []

#         weekly.append(dict(
#             cycle=cycle_id,
#             formation_start=str(t0), formation_end=str(t_form_end),
#             trade_start=str(t_form_end), trade_end=str(t_trade_end),
#             status=skip_status if skip_status else "OK",
#             selected_pair=f"{coin1}-{coin2}" if coin1 else None,
#             copula=best_name,
#             beta1=float(beta1) if coin1 and np.isfinite(beta1) else np.nan,
#             beta2=float(beta2) if coin1 and np.isfinite(beta2) else np.nan,
#             q1=q1, q2=q2,
#             top_list=",".join(top) if top else "",
#             fit_messages="; ".join(msgs) if msgs else "",
#             open_slots_start=len(active_slots),
#             cop_rotation=_cop_rotation,
#             cop_params=_cop_params,
#             s1_sorted=_s1_sorted_list,
#             s2_sorted=_s2_sorted_list,
#             diag_bars=cycle_diag_bars,  # will be filled during bar loop below
#             pseudo_u=pseudo_u if new_slot is not None else [],
#             pseudo_v=pseudo_v if new_slot is not None else [],
#             fit_summary=_serialize_fit_summary(df_fit) if new_slot is not None and df_fit is not None else [],
#         ))

#         # Bar loop : all active slots
#         needed_coins = {p.ref}
#         for slot in active_slots:
#             needed_coins.add(slot["coin1"])
#             needed_coins.add(slot["coin2"])
#         needed_cols = [c for c in needed_coins if c in df_trade.columns]
#         trade_bars  = df_trade[needed_cols].dropna(subset=[p.ref])

#         for ts, row in trade_bars.iterrows():
#             pref = float(row[p.ref])

#             ts_dt   = pd.to_datetime(ts) if not isinstance(ts, pd.Timestamp) else ts
#             ts_date = ts_dt.date()
#             if current_day != ts_date:
#                 current_day        = ts_date
#                 unrealized_all     = sum(s["net_pnl"] for s in active_slots)
#                 daily_start_equity = current_equity + unrealized_all
#                 daily_stopped      = False

#             for slot in active_slots:
#                 c1, c2 = slot["coin1"], slot["coin2"]
#                 if c1 not in row.index or c2 not in row.index:
#                     continue
#                 p1_raw, p2_raw = row[c1], row[c2]
#                 if pd.isna(p1_raw) or pd.isna(p2_raw):
#                     continue
#                 p1, p2   = float(p1_raw), float(p2_raw)
#                 _q1, _q2 = slot["q1"], slot["q2"]
#                 _tn      = slot["total_notional"]

#                 # 1. Mark-to-market
#                 if slot["prev_p1"] is not None and slot["pos"] != 0:
#                     dp1  = p1 - float(slot["prev_p1"])
#                     dp2  = p2 - float(slot["prev_p2"])
#                     step = (_q1 * dp1 - _q2 * dp2) if slot["pos"] == +1 else (-_q1 * dp1 + _q2 * dp2)
#                     slot["gross_pnl"] += step
#                     slot["net_pnl"]   += step

#                 # 2. Stop-loss checks
#                 stop_triggered = False
#                 is_current     = (slot is new_slot)

#                 if slot["pos"] != 0:
#                     # A) Trade stop-loss
#                     if p.use_trade_stop_loss:
#                         if slot["net_pnl"] < -abs(p.trade_stop_loss_pct) * _tn:
#                             current_equity, current_equity_gross, hwm = _close_slot(
#                                 slot, ts, p1, p2, "TRADE_STOP_LOSS", p.fee_rate,
#                                 trades, stop_loss_events,
#                                 current_equity, current_equity_gross, hwm,
#                             )
#                             stop_triggered = True

#                     # B) Trailing stop
#                     if p.use_trailing_stop and not stop_triggered and slot["pos"] != 0:
#                         if slot["net_pnl"] > slot["peak_trade_pnl"]:
#                             slot["peak_trade_pnl"] = slot["net_pnl"]
#                         if slot["peak_trade_pnl"] > abs(p.trailing_stop_activation) * _tn:
#                             trail_thr = slot["peak_trade_pnl"] - abs(p.trailing_stop_pct) * _tn
#                             if slot["net_pnl"] < trail_thr:
#                                 current_equity, current_equity_gross, hwm = _close_slot(
#                                     slot, ts, p1, p2, "TRAILING_STOP", p.fee_rate,
#                                     trades, stop_loss_events,
#                                     current_equity, current_equity_gross, hwm,
#                                 )
#                                 stop_triggered = True

#                     # C) Daily drawdown limit
#                     if p.use_daily_drawdown_limit and not stop_triggered and slot["pos"] != 0 and daily_start_equity:
#                         unrealized_all = sum(s["net_pnl"] for s in active_slots)
#                         intraday_eq    = current_equity + unrealized_all
#                         if daily_start_equity > 0:
#                             daily_dd = (intraday_eq - daily_start_equity) / daily_start_equity
#                             if daily_dd < -abs(p.daily_drawdown_limit_pct):
#                                 current_equity, current_equity_gross, hwm = _close_slot(
#                                     slot, ts, p1, p2, "DAILY_DD_LIMIT", p.fee_rate,
#                                     trades, stop_loss_events,
#                                     current_equity, current_equity_gross, hwm,
#                                 )
#                                 stop_triggered = True
#                                 daily_stopped  = True

#                 if stop_triggered or daily_stopped:
#                     slot["pending_sig"]   = 0
#                     slot["pending_close"] = False
#                     slot["prev_p1"] = p1
#                     slot["prev_p2"] = p2
#                     continue

#                 # 3. Execute pending orders
#                 trade_notional = abs(_q1 * p1) + abs(_q2 * p2)

#                 if slot["pos"] != 0 and slot["pending_close"]:
#                     current_equity, current_equity_gross, hwm = _close_slot(
#                         slot, ts, p1, p2, "signal", p.fee_rate,
#                         trades, stop_loss_events,
#                         current_equity, current_equity_gross, hwm,
#                     )
#                 slot["pending_close"] = False

#                 if slot["pending_sig"] != 0:
#                     if slot["pos"] == 0 and is_current:
#                         fee = p.fee_rate * abs(trade_notional)
#                         slot["fees_paid"] += fee
#                         slot["net_pnl"]   -= fee
#                         slot["pos"]        = int(slot["pending_sig"])
#                         slot["entry_ts"]   = ts
#                         slot["entry_p1"]   = p1
#                         slot["entry_p2"]   = p2
#                         slot["peak_trade_pnl"] = 0.0
#                     elif (slot["pos"] != 0 and is_current
#                           and p.flip_on_opposite
#                           and int(slot["pending_sig"]) != int(slot["pos"])):
#                         current_equity, current_equity_gross, hwm = _close_slot(
#                             slot, ts, p1, p2, "signal", p.fee_rate,
#                             trades, stop_loss_events,
#                             current_equity, current_equity_gross, hwm,
#                         )
#                         fee = p.fee_rate * abs(trade_notional)
#                         slot["fees_paid"] += fee
#                         slot["net_pnl"]   -= fee
#                         slot["pos"]        = int(slot["pending_sig"])
#                         slot["entry_ts"]   = ts
#                         slot["entry_p1"]   = p1
#                         slot["entry_p2"]   = p2
#                         slot["peak_trade_pnl"] = 0.0
#                 slot["pending_sig"] = 0

#                 # 4. Genere le signal pour la prochaine barre
#                 b1, b2 = slot["beta1"], slot["beta2"]
#                 s1_val = float(pref - b1 * p1) if np.isfinite(b1) else np.nan
#                 s2_val = float(pref - b2 * p2) if np.isfinite(b2) else np.nan

#                 if np.isfinite(s1_val) and np.isfinite(s2_val):
#                     sig, det = generate_signals_reference_copula(
#                         cop=slot["cop"],
#                         sorted_s1=slot["s1_sorted"], sorted_s2=slot["s2_sorted"],
#                         s1_val=s1_val, s2_val=s2_val,
#                         entry=p.entry, exit=p.exit,
#                     )
#                     close_now = bool(det.get("close", 0.0) > 0.5)

#                     if slot["pos"] != 0 and close_now:
#                         slot["pending_close"] = True
#                     elif slot["pos"] == 0 and sig != 0 and is_current:
#                         slot["pending_sig"] = int(sig)
#                     elif slot["pos"] != 0 and sig != 0 and is_current:
#                         slot["pending_sig"] = int(sig)
#                     # ── Diagnostic capture ──
#                     if slot is new_slot:
#                         cycle_diag_bars.append(dict(
#                             ts=ts, u=det["u"], v=det["v"],
#                             h12=det["h12"], h21=det["h21"],
#                             sig=sig, close=det["close"],
#                             p1=p1, p2=p2, pref=pref,
#                             s1=s1_val, s2=s2_val,
#                         ))

#                 slot["prev_p1"] = p1
#                 slot["prev_p2"] = p2

#             unrealized       = sum(s["net_pnl"]   for s in active_slots)
#             unrealized_gross = sum(s["gross_pnl"] for s in active_slots)
#             equity.loc[ts]       = current_equity       + unrealized
#             equity_gross.loc[ts] = current_equity_gross + unrealized_gross

#         # Fin du cycle
#         if p.force_week_end_close:
#             # Ferme TOUTES les positions ouvertes (comportement papier)
#             for slot in active_slots:
#                 if slot["pos"] != 0:
#                     c1, c2 = slot["coin1"], slot["coin2"]
#                     lb = df_trade[[c1, c2]].dropna().tail(1) if not df_trade.empty else pd.DataFrame()
#                     if not lb.empty:
#                         ts_end = lb.index[0]
#                         current_equity, current_equity_gross, hwm = _close_slot(
#                             slot, ts_end,
#                             float(lb.iloc[0][c1]), float(lb.iloc[0][c2]),
#                             "week_end", p.fee_rate,
#                             trades, stop_loss_events,
#                             current_equity, current_equity_gross, hwm,
#                         )
#             active_slots.clear()
#         else:
#             # Garde uniquement les slots avec position ouverte
#             # Les slots sans position sont des modeles stales -> purges
#             active_slots = [s for s in active_slots if s["pos"] != 0]

#         t0 = t0 + step_delta
#         cycle_id += 1

#     # Finalise
#     equity       = equity.ffill().dropna()
#     equity_gross = equity_gross.ffill().dropna()

#     trades_df = pd.DataFrame(trades)
#     weekly_df = pd.DataFrame(weekly)

#     metrics = performance_metrics(equity, interval=p.interval)
#     metrics_g = performance_metrics(equity_gross, interval=p.interval)

#     monthly = pd.Series(dtype=float)
#     if len(equity) > 10:
#         monthly = equity.resample("ME").last().pct_change().dropna()

#     cop_freq = pd.DataFrame(columns=["copula", "count"])
#     if not weekly_df.empty and "copula" in weekly_df.columns:
#         cop_freq = weekly_df["copula"].value_counts().reset_index()
#         cop_freq.columns = ["copula", "count"]

#     sl_stats: dict = {}
#     if not trades_df.empty and "exit_reason" in trades_df.columns:
#         reason_counts = trades_df["exit_reason"].value_counts().to_dict()
#         sl_stats["reason_counts"] = {str(k): int(v) for k, v in reason_counts.items()}
#         sl_types  = ["TRADE_STOP_LOSS", "TRAILING_STOP", "DAILY_DD_LIMIT"]
#         sl_trades = trades_df[trades_df["exit_reason"].isin(sl_types)]
#         sl_stats["total_stop_loss_trades"] = int(sl_trades.shape[0])
#         sl_stats["total_sl_pnl"] = float(sl_trades["net_pnl"].sum()) if not sl_trades.empty else 0.0
#     else:
#         sl_stats["reason_counts"] = {}
#         sl_stats["total_stop_loss_trades"] = 0
#         sl_stats["total_sl_pnl"] = 0.0
#     sl_stats["events"]            = stop_loss_events
#     sl_stats["portfolio_stopped"] = portfolio_stopped

#     return dict(
#         equity=equity,
#         equity_gross=equity_gross,
#         trades=trades_df,
#         weekly=weekly_df,
#         metrics=metrics,
#         metrics_gross=metrics_g,
#         monthly_returns=monthly,
#         copula_freq=cop_freq,
#         params=p,
#         stop_loss_stats=sl_stats,
#     )


# dash_bot/core/backtest.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List

from .selection import select_stationary_spreads
from .metrics import _to_datetime, performance_metrics
from .selection import rank_coins
from .copula_engine import fit_pair_copula, build_copula
from .params import BacktestParams
from .signals import generate_signals_reference_copula


# ---------------------------------------------------------------------------
# Slot helpers
# ---------------------------------------------------------------------------

def _serialize_fit_summary(df_fit):
    """Extract JSON-safe summary from fit_copulas DataFrame for dashboard."""
    if df_fit is None or df_fit.empty:
        return []
    rows = []
    for _, r in df_fit.iterrows():
        if not np.isfinite(r.get("aic", np.nan)):
            continue
        rows.append(dict(
            name=str(r.get("name", "?")),
            rotation=int(r.get("rotation", 0)),
            aic=float(r["aic"]) if np.isfinite(r.get("aic", np.nan)) else None,
            loglik=float(r["loglik"]) if np.isfinite(r.get("loglik", np.nan)) else None,
            kt_err=float(r["kt_err"]) if np.isfinite(r.get("kt_err", np.nan)) else None,
            tail_gap=float(r["tail_gap"]) if np.isfinite(r.get("tail_gap", np.nan)) else None,
            score=float(r["score_stage1"]) if "score_stage1" in r and np.isfinite(r.get("score_stage1", np.nan)) else None,
            tail_dep_L=float(r["tail_dep_L"]) if np.isfinite(r.get("tail_dep_L", np.nan)) else None,
            tail_dep_U=float(r["tail_dep_U"]) if np.isfinite(r.get("tail_dep_U", np.nan)) else None,
            params=[float(x) for x in np.atleast_1d(r["params"])] if r.get("params") is not None and len(np.atleast_1d(r["params"])) > 0 else [],
            evaluable=bool(r.get("evaluable", False)),
        ))
    return rows

def _make_slot(cycle_id, coin1, coin2, cop, best_name,
               beta1, beta2, s1_sorted, s2_sorted,
               q1, q2, total_notional, marginal1=None, marginal2=None):
    """Crée un slot copula sans position ouverte."""
    return dict(
        cycle_id=cycle_id,
        coin1=coin1, coin2=coin2,
        best_name=best_name,
        cop=cop,
        beta1=beta1, beta2=beta2,
        s1_sorted=s1_sorted, s2_sorted=s2_sorted,
        marginal1=marginal1, marginal2=marginal2,
        q1=q1, q2=q2,
        total_notional=total_notional,
        pos=0,
        entry_ts=None, entry_p1=None, entry_p2=None,
        gross_pnl=0.0, net_pnl=0.0, fees_paid=0.0,
        peak_trade_pnl=0.0,
        pending_sig=0,
        pending_close=False,
        prev_p1=None, prev_p2=None,
    )


def _close_slot(slot, close_ts, close_p1, close_p2, reason,
                fee_rate, trades, stop_loss_events,
                current_equity, current_equity_gross, hwm):
    q1, q2 = slot["q1"], slot["q2"]
    notional = abs(q1 * close_p1) + abs(q2 * close_p2)
    fee = fee_rate * abs(notional)
    slot["fees_paid"] += fee
    slot["net_pnl"]   -= fee

    is_stop = reason not in ("signal", "week_end")

    trades.append(dict(
        cycle=slot["cycle_id"],
        pair=f"{slot['coin1']}-{slot['coin2']}",
        copula=slot["best_name"],
        entry_time=str(slot["entry_ts"]),
        exit_time=str(close_ts),
        direction="LONG1_SHORT2" if slot["pos"] == +1 else "SHORT1_LONG2",
        q1=q1, q2=q2,
        beta1=slot["beta1"], beta2=slot["beta2"],
        entry_p1=slot["entry_p1"], entry_p2=slot["entry_p2"],
        exit_p1=close_p1, exit_p2=close_p2,
        gross_pnl=slot["gross_pnl"],
        net_pnl=slot["net_pnl"],
        fees=slot["fees_paid"],
        bars=int(
            (pd.to_datetime(close_ts) - pd.to_datetime(slot["entry_ts"])
             ).total_seconds() // 60
        ) if slot["entry_ts"] else np.nan,
        forced_week_end=(reason == "week_end"),
        exit_reason=reason,
    ))

    if is_stop:
        stop_loss_events.append(dict(
            time=str(close_ts),
            type=reason,
            detail=f"PnL={slot['net_pnl']:.2f}",
            equity=current_equity + slot["net_pnl"],
        ))

    current_equity       += slot["net_pnl"]
    current_equity_gross += slot["gross_pnl"]
    if current_equity > hwm:
        hwm = current_equity

    slot["pos"] = 0
    slot["entry_ts"] = slot["entry_p1"] = slot["entry_p2"] = None
    slot["gross_pnl"] = slot["net_pnl"] = slot["fees_paid"] = 0.0
    slot["peak_trade_pnl"] = 0.0
    slot["pending_sig"] = 0
    slot["pending_close"] = False

    return current_equity, current_equity_gross, hwm


# ---------------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------------

def backtest_reference_copula(prices: pd.DataFrame, p: BacktestParams) -> Dict[str, Any]:
    """
    Backtest de la strategie proposee (reference spread copula).

    Architecture multi-slot :
    - Chaque semaine un nouveau copula est fitte -> nouveau slot cree.
    - force_week_end_close=True  : comportement papier, toutes les positions
      fermees en fin de semaine.
    - force_week_end_close=False : un slot avec position ouverte continue
      au-dela de sa semaine. Il utilise son propre copula pour les signaux
      de sortie. Seul le slot courant peut ouvrir de nouvelles positions.
    - Un slot sans position en fin de cycle est purge (modele stale).
    """
    prices = prices.copy()
    prices = prices.loc[
        (prices.index >= _to_datetime(p.start)) &
        (prices.index <  _to_datetime(p.end))
    ]
    prices = prices.sort_index()
    prices = prices[p.symbols].dropna(how="all")

    if p.ref not in prices.columns:
        raise ValueError(f"Reference asset '{p.ref}' absent des donnees.")

    start_dt = prices.index.min()
    end_dt   = prices.index.max()
    if start_dt is pd.NaT or end_dt is pd.NaT:
        raise ValueError("Pas de donnees dans la fenetre demandee.")

    formation_delta = pd.Timedelta(days=7 * p.formation_weeks)
    trading_delta   = pd.Timedelta(days=7 * p.trading_weeks)
    step_delta      = pd.Timedelta(days=7 * p.step_weeks)

    equity       = pd.Series(index=prices.index, dtype=float)
    equity_gross = pd.Series(index=prices.index, dtype=float)
    equity.iloc[:]       = np.nan
    equity_gross.iloc[:] = np.nan

    trades:           list = []
    weekly:           list = []
    stop_loss_events: list = []

    current_equity       = float(p.initial_equity)
    current_equity_gross = float(p.initial_equity)
    hwm                  = float(p.initial_equity)
    portfolio_stopped    = False

    active_slots: List[dict] = []

    current_day        = None
    daily_start_equity = None
    daily_stopped      = False

    t0       = start_dt
    cycle_id = 0

    # Pré-calcul du nombre total de cycles pour la progression
    _t = start_dt
    _total_cycles = 0
    while True:
        if _t + formation_delta + trading_delta > end_dt:
            break
        _total_cycles += 1
        _t += step_delta

    while True:
        t_form_end = t0 + formation_delta
        t_trade_end = t_form_end + trading_delta
        if t_trade_end > end_dt:
            break

        print(
            f"[Backtest] Cycle {cycle_id + 1}/{_total_cycles} │ "
            f"Formation: {t0.strftime('%Y-%m-%d')} → {t_form_end.strftime('%Y-%m-%d')} │ "
            f"Trading: {t_form_end.strftime('%Y-%m-%d')} → {t_trade_end.strftime('%Y-%m-%d')} │ "
            f"Equity: {current_equity:,.0f} USDT",
            flush=True,
        )

        # Max drawdown portfolio stop
        if p.use_max_drawdown_stop and not portfolio_stopped:
            dd_from_hwm = (current_equity - hwm) / hwm if hwm > 0 else 0.0
            if dd_from_hwm < -abs(p.max_drawdown_stop_pct):
                portfolio_stopped = True
                stop_loss_events.append(dict(
                    time=str(t_form_end),
                    type="MAX_DRAWDOWN_STOP",
                    detail=f"DD={dd_from_hwm:.4f} < -{p.max_drawdown_stop_pct:.4f}",
                    equity=current_equity,
                ))

        df_form  = prices.loc[(prices.index >= t0)         & (prices.index < t_form_end)]
        df_trade = prices.loc[(prices.index >= t_form_end) & (prices.index < t_trade_end)]

        # Fit this cycle's copula
        new_slot    = None
        skip_status = None
        coin1 = coin2 = best_name = None
        beta1 = beta2 = np.nan
        q1 = q2 = 0.0
        msgs = []
        top  = []
        pseudo_u = []
        pseudo_v = []
        df_fit = None

        if portfolio_stopped:
            skip_status = "STOPPED (max drawdown limit reached)"
        else:
            candidates = [s for s in p.symbols if s != p.ref and s in df_form.columns]
            ref_ok     = df_form[p.ref].notna()
            form_slice = df_form.loc[ref_ok, candidates]
            coverage   = form_slice.notna().mean()
            points     = form_slice.notna().sum()
            eligible   = [
                s for s in candidates
                if float(coverage.get(s, 0.0)) >= float(p.min_coverage)
                and int(points.get(s, 0)) >= int(p.min_obs)
            ]

            summary, spreads, betas = select_stationary_spreads(
                prices=df_form, ref=p.ref, candidates=eligible,
                adf_alpha=p.adf_alpha, cointegration_test=p.cointegration_test,
                kss_crit=p.kss_crit, min_obs=p.min_obs,
            )
            accepted = (
                summary[summary["accepted"] == True]["coin"].tolist()
                if not summary.empty else []
            )

            if len(accepted) < 2 or df_trade.empty:
                skip_status = "SKIP (not enough stationary spreads)"
            else:
                rank_df = rank_coins(df_form, spreads, p.ref, accepted, p.rank_method)
                top = (
                    rank_df["coin"].tolist()[:max(2, p.top_k)]
                    if not rank_df.empty else accepted[:max(2, p.top_k)]
                )
                if len(top) < 2:
                    skip_status = "SKIP (ranking failed)"
                else:
                    coin1, coin2 = top[0], top[1]
                    s1_form = spreads[coin1]
                    s2_form = spreads[coin2]
                    beta1   = betas.get(coin1, np.nan)
                    beta2   = betas.get(coin2, np.nan)

                    best_name, best_params, df_fit, msgs = fit_pair_copula(
                        s1_form, s2_form, p.copula_pick, p.copula_manual,
                        suppress_logs=p.suppress_fit_logs,
                    )
                    if best_name is None:
                        skip_status = "SKIP (copula fit failed)"
                    else:
                        cop       = build_copula(best_name, best_params)
                        s1_sorted = np.sort(s1_form.dropna().values.astype(float))
                        s2_sorted = np.sort(s2_form.dropna().values.astype(float))
                        marginal1 = df_fit.attrs.get("marginal1") if df_fit is not None else None
                        marginal2 = df_fit.attrs.get("marginal2") if df_fit is not None else None

                        # PIT pseudo-observations for diagnostic scatter (paper-style marginals)
                        _s1_aligned, _s2_aligned = s1_form.align(s2_form, join="inner")
                        _s1_clean = _s1_aligned.dropna()
                        _s2_clean = _s2_aligned.loc[_s1_clean.index].dropna()
                        _common = _s1_clean.index.intersection(_s2_clean.index)
                        _n_po = len(_common)
                        if _n_po > 10 and marginal1 is not None and marginal2 is not None:
                            from .copula_engine import marginal_cdf_array
                            pseudo_u = marginal_cdf_array(marginal1, _s1_clean.loc[_common].values).tolist()
                            pseudo_v = marginal_cdf_array(marginal2, _s2_clean.loc[_common].values).tolist()
                        elif _n_po > 10:
                            from scipy.stats import rankdata
                            pseudo_u = (rankdata(_s1_clean.loc[_common].values) / (_n_po + 1)).tolist()
                            pseudo_v = (rankdata(_s2_clean.loc[_common].values) / (_n_po + 1)).tolist()
                        else:
                            pseudo_u, pseudo_v = [], []

                        first_bar = df_trade[[coin1, coin2]].dropna().head(1)
                        if first_bar.empty:
                            skip_status = "SKIP (no trade prices)"
                        else:
                            p1_open = float(first_bar.iloc[0][coin1])
                            p2_open = float(first_bar.iloc[0][coin2])

                            q1 = (p.cap_per_leg / p1_open) if (np.isfinite(p1_open) and p1_open > 0) else 0.0
                            q2 = (p.cap_per_leg / p2_open) if (np.isfinite(p2_open) and p2_open > 0) else 0.0

                            new_slot = _make_slot(
                                cycle_id=cycle_id,
                                coin1=coin1, coin2=coin2,
                                cop=cop, best_name=best_name,
                                beta1=beta1, beta2=beta2,
                                s1_sorted=s1_sorted, s2_sorted=s2_sorted,
                                q1=q1, q2=q2,
                                total_notional=(q1 * p1_open) + (q2 * p2_open),
                                marginal1=marginal1, marginal2=marginal2,
                            )
                            active_slots.append(new_slot)

        # ── Diagnostic storage for the current cycle (must be BEFORE weekly.append) ──
        cycle_diag_bars = []  # will collect per-bar (u, v, h12, h21, sig)

        # Extract cop metadata for dashboard reconstruction
        _cop_params = []
        _cop_rotation = 0
        _s1_sorted_list = []
        _s2_sorted_list = []
        if new_slot is not None:
            try:
                _cop_params = list(float(x) for x in np.atleast_1d(cop.get_parameters()))
            except Exception:
                _cop_params = []
            _cop_rotation = getattr(cop, 'rotation', 0)
            _s1_sorted_list = s1_sorted.tolist() if s1_sorted is not None else []
            _s2_sorted_list = s2_sorted.tolist() if s2_sorted is not None else []

        weekly.append(dict(
            cycle=cycle_id,
            formation_start=str(t0), formation_end=str(t_form_end),
            trade_start=str(t_form_end), trade_end=str(t_trade_end),
            status=skip_status if skip_status else "OK",
            selected_pair=f"{coin1}-{coin2}" if coin1 else None,
            copula=best_name,
            beta1=float(beta1) if coin1 and np.isfinite(beta1) else np.nan,
            beta2=float(beta2) if coin1 and np.isfinite(beta2) else np.nan,
            q1=q1, q2=q2,
            top_list=",".join(top) if top else "",
            fit_messages="; ".join(msgs) if msgs else "",
            open_slots_start=len(active_slots),
            cop_rotation=_cop_rotation,
            cop_params=_cop_params,
            s1_sorted=_s1_sorted_list,
            s2_sorted=_s2_sorted_list,
            marginal1=(df_fit.attrs.get("marginal1_serialized") if new_slot is not None and df_fit is not None else None),
            marginal2=(df_fit.attrs.get("marginal2_serialized") if new_slot is not None and df_fit is not None else None),
            diag_bars=cycle_diag_bars,  # will be filled during bar loop below
            pseudo_u=pseudo_u if new_slot is not None else [],
            pseudo_v=pseudo_v if new_slot is not None else [],
            fit_summary=_serialize_fit_summary(df_fit) if new_slot is not None and df_fit is not None else [],
        ))

        # Bar loop : all active slots
        needed_coins = {p.ref}
        for slot in active_slots:
            needed_coins.add(slot["coin1"])
            needed_coins.add(slot["coin2"])
        needed_cols = [c for c in needed_coins if c in df_trade.columns]
        trade_bars  = df_trade[needed_cols].dropna(subset=[p.ref])

        for ts, row in trade_bars.iterrows():
            pref = float(row[p.ref])

            ts_dt   = pd.to_datetime(ts) if not isinstance(ts, pd.Timestamp) else ts
            ts_date = ts_dt.date()
            if current_day != ts_date:
                current_day        = ts_date
                unrealized_all     = sum(s["net_pnl"] for s in active_slots)
                daily_start_equity = current_equity + unrealized_all
                daily_stopped      = False

            for slot in active_slots:
                c1, c2 = slot["coin1"], slot["coin2"]
                if c1 not in row.index or c2 not in row.index:
                    continue
                p1_raw, p2_raw = row[c1], row[c2]
                if pd.isna(p1_raw) or pd.isna(p2_raw):
                    continue
                p1, p2   = float(p1_raw), float(p2_raw)
                _q1, _q2 = slot["q1"], slot["q2"]
                _tn      = slot["total_notional"]

                # 1. Mark-to-market
                if slot["prev_p1"] is not None and slot["pos"] != 0:
                    dp1  = p1 - float(slot["prev_p1"])
                    dp2  = p2 - float(slot["prev_p2"])
                    step = (_q1 * dp1 - _q2 * dp2) if slot["pos"] == +1 else (-_q1 * dp1 + _q2 * dp2)
                    slot["gross_pnl"] += step
                    slot["net_pnl"]   += step

                # 2. Stop-loss checks
                stop_triggered = False
                is_current     = (slot is new_slot)

                if slot["pos"] != 0:
                    # A) Trade stop-loss
                    if p.use_trade_stop_loss:
                        if slot["net_pnl"] < -abs(p.trade_stop_loss_pct) * _tn:
                            current_equity, current_equity_gross, hwm = _close_slot(
                                slot, ts, p1, p2, "TRADE_STOP_LOSS", p.fee_rate,
                                trades, stop_loss_events,
                                current_equity, current_equity_gross, hwm,
                            )
                            stop_triggered = True

                    # B) Trailing stop
                    if p.use_trailing_stop and not stop_triggered and slot["pos"] != 0:
                        if slot["net_pnl"] > slot["peak_trade_pnl"]:
                            slot["peak_trade_pnl"] = slot["net_pnl"]
                        if slot["peak_trade_pnl"] > abs(p.trailing_stop_activation) * _tn:
                            trail_thr = slot["peak_trade_pnl"] - abs(p.trailing_stop_pct) * _tn
                            if slot["net_pnl"] < trail_thr:
                                current_equity, current_equity_gross, hwm = _close_slot(
                                    slot, ts, p1, p2, "TRAILING_STOP", p.fee_rate,
                                    trades, stop_loss_events,
                                    current_equity, current_equity_gross, hwm,
                                )
                                stop_triggered = True

                    # C) Daily drawdown limit
                    if p.use_daily_drawdown_limit and not stop_triggered and slot["pos"] != 0 and daily_start_equity:
                        unrealized_all = sum(s["net_pnl"] for s in active_slots)
                        intraday_eq    = current_equity + unrealized_all
                        if daily_start_equity > 0:
                            daily_dd = (intraday_eq - daily_start_equity) / daily_start_equity
                            if daily_dd < -abs(p.daily_drawdown_limit_pct):
                                current_equity, current_equity_gross, hwm = _close_slot(
                                    slot, ts, p1, p2, "DAILY_DD_LIMIT", p.fee_rate,
                                    trades, stop_loss_events,
                                    current_equity, current_equity_gross, hwm,
                                )
                                stop_triggered = True
                                daily_stopped  = True

                if stop_triggered or daily_stopped:
                    slot["pending_sig"]   = 0
                    slot["pending_close"] = False
                    slot["prev_p1"] = p1
                    slot["prev_p2"] = p2
                    continue

                # 3. Execute pending orders
                trade_notional = abs(_q1 * p1) + abs(_q2 * p2)

                if slot["pos"] != 0 and slot["pending_close"]:
                    current_equity, current_equity_gross, hwm = _close_slot(
                        slot, ts, p1, p2, "signal", p.fee_rate,
                        trades, stop_loss_events,
                        current_equity, current_equity_gross, hwm,
                    )
                slot["pending_close"] = False

                if slot["pending_sig"] != 0:
                    if slot["pos"] == 0 and is_current:
                        fee = p.fee_rate * abs(trade_notional)
                        slot["fees_paid"] += fee
                        slot["net_pnl"]   -= fee
                        slot["pos"]        = int(slot["pending_sig"])
                        slot["entry_ts"]   = ts
                        slot["entry_p1"]   = p1
                        slot["entry_p2"]   = p2
                        slot["peak_trade_pnl"] = 0.0
                    elif (slot["pos"] != 0 and is_current
                          and p.flip_on_opposite
                          and int(slot["pending_sig"]) != int(slot["pos"])):
                        current_equity, current_equity_gross, hwm = _close_slot(
                            slot, ts, p1, p2, "signal", p.fee_rate,
                            trades, stop_loss_events,
                            current_equity, current_equity_gross, hwm,
                        )
                        fee = p.fee_rate * abs(trade_notional)
                        slot["fees_paid"] += fee
                        slot["net_pnl"]   -= fee
                        slot["pos"]        = int(slot["pending_sig"])
                        slot["entry_ts"]   = ts
                        slot["entry_p1"]   = p1
                        slot["entry_p2"]   = p2
                        slot["peak_trade_pnl"] = 0.0
                slot["pending_sig"] = 0

                # 4. Genere le signal pour la prochaine barre
                b1, b2 = slot["beta1"], slot["beta2"]
                s1_val = float(pref - b1 * p1) if np.isfinite(b1) else np.nan
                s2_val = float(pref - b2 * p2) if np.isfinite(b2) else np.nan

                if np.isfinite(s1_val) and np.isfinite(s2_val):
                    sig, det = generate_signals_reference_copula(
                        cop=slot["cop"],
                        sorted_s1=slot["s1_sorted"], sorted_s2=slot["s2_sorted"],
                        s1_val=s1_val, s2_val=s2_val,
                        entry=p.entry, exit=p.exit,
                        marginal1=slot.get("marginal1"), marginal2=slot.get("marginal2"),
                    )
                    close_now = bool(det.get("close", 0.0) > 0.5)

                    if slot["pos"] != 0 and close_now:
                        slot["pending_close"] = True
                    elif slot["pos"] == 0 and sig != 0 and is_current:
                        slot["pending_sig"] = int(sig)
                    elif slot["pos"] != 0 and sig != 0 and is_current:
                        slot["pending_sig"] = int(sig)
                    # ── Diagnostic capture ──
                    if slot is new_slot:
                        cycle_diag_bars.append(dict(
                            ts=ts, u=det["u"], v=det["v"],
                            h12=det["h12"], h21=det["h21"],
                            sig=sig, close=det["close"],
                            p1=p1, p2=p2, pref=pref,
                            s1=s1_val, s2=s2_val,
                        ))

                slot["prev_p1"] = p1
                slot["prev_p2"] = p2

            unrealized       = sum(s["net_pnl"]   for s in active_slots)
            unrealized_gross = sum(s["gross_pnl"] for s in active_slots)
            equity.loc[ts]       = current_equity       + unrealized
            equity_gross.loc[ts] = current_equity_gross + unrealized_gross

        # Fin du cycle
        if p.force_week_end_close:
            # Ferme TOUTES les positions ouvertes (comportement papier)
            for slot in active_slots:
                if slot["pos"] != 0:
                    c1, c2 = slot["coin1"], slot["coin2"]
                    lb = df_trade[[c1, c2]].dropna().tail(1) if not df_trade.empty else pd.DataFrame()
                    if not lb.empty:
                        ts_end = lb.index[0]
                        current_equity, current_equity_gross, hwm = _close_slot(
                            slot, ts_end,
                            float(lb.iloc[0][c1]), float(lb.iloc[0][c2]),
                            "week_end", p.fee_rate,
                            trades, stop_loss_events,
                            current_equity, current_equity_gross, hwm,
                        )
            active_slots.clear()
        else:
            # Garde uniquement les slots avec position ouverte
            # Les slots sans position sont des modeles stales -> purges
            active_slots = [s for s in active_slots if s["pos"] != 0]

        t0 = t0 + step_delta
        cycle_id += 1

    # Finalise
    equity       = equity.ffill().dropna()
    equity_gross = equity_gross.ffill().dropna()

    trades_df = pd.DataFrame(trades)
    weekly_df = pd.DataFrame(weekly)

    metrics = performance_metrics(equity, interval=p.interval)
    metrics_g = performance_metrics(equity_gross, interval=p.interval)

    monthly = pd.Series(dtype=float)
    if len(equity) > 10:
        monthly = equity.resample("ME").last().pct_change().dropna()

    cop_freq = pd.DataFrame(columns=["copula", "count"])
    if not weekly_df.empty and "copula" in weekly_df.columns:
        cop_freq = weekly_df["copula"].value_counts().reset_index()
        cop_freq.columns = ["copula", "count"]

    sl_stats: dict = {}
    if not trades_df.empty and "exit_reason" in trades_df.columns:
        reason_counts = trades_df["exit_reason"].value_counts().to_dict()
        sl_stats["reason_counts"] = {str(k): int(v) for k, v in reason_counts.items()}
        sl_types  = ["TRADE_STOP_LOSS", "TRAILING_STOP", "DAILY_DD_LIMIT"]
        sl_trades = trades_df[trades_df["exit_reason"].isin(sl_types)]
        sl_stats["total_stop_loss_trades"] = int(sl_trades.shape[0])
        sl_stats["total_sl_pnl"] = float(sl_trades["net_pnl"].sum()) if not sl_trades.empty else 0.0
    else:
        sl_stats["reason_counts"] = {}
        sl_stats["total_stop_loss_trades"] = 0
        sl_stats["total_sl_pnl"] = 0.0
    sl_stats["events"]            = stop_loss_events
    sl_stats["portfolio_stopped"] = portfolio_stopped

    return dict(
        equity=equity,
        equity_gross=equity_gross,
        trades=trades_df,
        weekly=weekly_df,
        metrics=metrics,
        metrics_gross=metrics_g,
        monthly_returns=monthly,
        copula_freq=cop_freq,
        params=p,
        stop_loss_stats=sl_stats,
    )