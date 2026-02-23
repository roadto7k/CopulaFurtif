# dash_bot/core/backtest.py
import pandas as pd
import numpy as np
from typing import Dict, Any

from .selection import select_stationary_spreads
from .metrics import _to_datetime, performance_metrics
from .selection import rank_coins
from .copula_engine import fit_pair_copula, build_copula
from .params import BacktestParams
from .signals import generate_signals_reference_copula

def backtest_reference_copula(prices: pd.DataFrame, p: BacktestParams) -> Dict[str, Any]:
    """
    Backtest de la stratégie proposée (reference spread copula).
    Avec risk management: trade stop-loss, daily DD limit, max DD stop, trailing stop.
    """
    prices = prices.copy()
    prices = prices.loc[(prices.index >= _to_datetime(p.start)) & (prices.index < _to_datetime(p.end))]
    prices = prices.sort_index()
    prices = prices[p.symbols].dropna(how="all")

    if p.ref not in prices.columns:
        raise ValueError(f"Reference asset '{p.ref}' absent des données.")

    # cycle windows in calendar time
    start_dt = prices.index.min()
    end_dt = prices.index.max()
    if start_dt is pd.NaT or end_dt is pd.NaT:
        raise ValueError("Pas de données dans la fenêtre demandée.")

    formation_delta = pd.Timedelta(days=7 * p.formation_weeks)
    trading_delta = pd.Timedelta(days=7 * p.trading_weeks)
    step_delta = pd.Timedelta(days=7 * p.step_weeks)

    # outputs
    equity = pd.Series(index=prices.index, dtype=float)
    equity_gross = pd.Series(index=prices.index, dtype=float)
    equity.iloc[:] = np.nan
    equity_gross.iloc[:] = np.nan

    trades = []
    weekly = []

    # --- RISK MANAGEMENT: tracking ---
    stop_loss_events = []

    current_equity = float(p.initial_equity)
    current_equity_gross = float(p.initial_equity)

    # High Water Mark pour max drawdown stop
    hwm = float(p.initial_equity)
    portfolio_stopped = False  # si True, on ne trade plus du tout

    # iterate cycles
    t0 = start_dt
    cycle_id = 0

    while True:
        t_form_end = t0 + formation_delta
        t_trade_end = t_form_end + trading_delta
        if t_trade_end > end_dt:
            break

        # --- RISK MANAGEMENT: check max drawdown stop AVANT le cycle ---
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

        if portfolio_stopped:
            df_trade = prices.loc[(prices.index >= t_form_end) & (prices.index < t_trade_end)]
            for ts in df_trade.index:
                equity.loc[ts] = current_equity
                equity_gross.loc[ts] = current_equity_gross
            weekly.append(dict(
                cycle=cycle_id,
                formation_start=str(t0), formation_end=str(t_form_end),
                trade_start=str(t_form_end), trade_end=str(t_trade_end),
                status="STOPPED (max drawdown limit reached)",
                selected_pair=None, copula=None,
            ))
            t0 = t0 + step_delta
            cycle_id += 1
            continue

        df_form = prices.loc[(prices.index >= t0) & (prices.index < t_form_end)]
        df_trade = prices.loc[(prices.index >= t_form_end) & (prices.index < t_trade_end)]

        # stationarity filter on formation
        candidates = [s for s in p.symbols if (s != p.ref and s in df_form.columns)]

        ref_ok = df_form[p.ref].notna()
        form_slice = df_form.loc[ref_ok, candidates]

        coverage = form_slice.notna().mean()
        points = form_slice.notna().sum()

        eligible = [
            s for s in candidates
            if float(coverage.get(s, 0.0)) >= float(p.min_coverage)
               and int(points.get(s, 0)) >= int(p.min_obs)
        ]

        summary, spreads, betas = select_stationary_spreads(
            prices=df_form,
            ref=p.ref,
            candidates=eligible,
            adf_alpha=p.adf_alpha,
            use_kss=p.use_kss,
            kss_crit=p.kss_crit,
            min_obs=p.min_obs,
        )
        accepted = summary[summary["accepted"] == True]["coin"].tolist() if not summary.empty else []
        if len(accepted) < 2 or df_trade.empty:
            weekly.append(dict(
                cycle=cycle_id,
                formation_start=str(t0), formation_end=str(t_form_end),
                trade_start=str(t_form_end), trade_end=str(t_trade_end),
                status="SKIP (not enough stationary spreads)",
                selected_pair=None,
                copula=None,
            ))
            for ts in df_trade.index:
                equity.loc[ts] = current_equity
                equity_gross.loc[ts] = current_equity_gross
            t0 = t0 + step_delta
            cycle_id += 1
            continue

        # rank
        rank_df = rank_coins(df_form, spreads, p.ref, accepted, p.rank_method)
        top = rank_df["coin"].tolist()[:max(2, p.top_k)] if not rank_df.empty else accepted[:max(2, p.top_k)]
        if len(top) < 2:
            weekly.append(dict(
                cycle=cycle_id, formation_start=str(t0), formation_end=str(t_form_end),
                trade_start=str(t_form_end), trade_end=str(t_trade_end),
                status="SKIP (ranking failed)",
                selected_pair=None, copula=None,
            ))
            for ts in df_trade.index:
                equity.loc[ts] = current_equity
                equity_gross.loc[ts] = current_equity_gross
            t0 = t0 + step_delta
            cycle_id += 1
            continue

        # only 1 pair (top2) for now
        coin1, coin2 = top[0], top[1]
        s1_form = spreads[coin1]
        s2_form = spreads[coin2]
        beta1 = betas.get(coin1, np.nan)
        beta2 = betas.get(coin2, np.nan)

        best_name, best_params, df_fit, msgs = fit_pair_copula(
            s1_form, s2_form, p.copula_pick, p.copula_manual, suppress_logs=p.suppress_fit_logs
        )
        if best_name is None:
            weekly.append(dict(
                cycle=cycle_id, formation_start=str(t0), formation_end=str(t_form_end),
                trade_start=str(t_form_end), trade_end=str(t_trade_end),
                status="SKIP (copula fit failed)",
                selected_pair=f"{coin1}-{coin2}", copula=None,
                messages="; ".join(msgs),
            ))
            for ts in df_trade.index:
                equity.loc[ts] = current_equity
                equity_gross.loc[ts] = current_equity_gross
            t0 = t0 + step_delta
            cycle_id += 1
            continue

        cop = build_copula(best_name, best_params)

        # prepare ECDF on formation spreads
        s1_sorted = np.sort(s1_form.dropna().values.astype(float))
        s2_sorted = np.sort(s2_form.dropna().values.astype(float))

        # quantities fixed for the week
        first_bar = df_trade[[coin1, coin2]].dropna().head(1)
        if first_bar.empty:
            weekly.append(dict(
                cycle=cycle_id, formation_start=str(t0), formation_end=str(t_form_end),
                trade_start=str(t_form_end), trade_end=str(t_trade_end),
                status="SKIP (no trade prices)",
                selected_pair=f"{coin1}-{coin2}", copula=best_name
            ))
            for ts in df_trade.index:
                equity.loc[ts] = current_equity
                equity_gross.loc[ts] = current_equity_gross
            t0 = t0 + step_delta
            cycle_id += 1
            continue

        p1_open = float(first_bar.iloc[0][coin1])
        p2_open = float(first_bar.iloc[0][coin2])
        q1 = (p.cap_per_leg / p1_open) if (np.isfinite(p1_open) and p1_open > 0) else 0.0
        q2 = (p.cap_per_leg / p2_open) if (np.isfinite(p2_open) and p2_open > 0) else 0.0

        # --- RISK MANAGEMENT: notionnel total pour seuils ---
        total_notional = p.cap_per_leg * 2.0

        pos = 0
        entry_ts = None
        entry_p1 = entry_p2 = None
        entry_fee = 0.0
        gross_pnl = 0.0
        net_pnl = 0.0
        fees_paid = 0.0

        # --- RISK MANAGEMENT: état intra-cycle ---
        peak_trade_pnl = 0.0        # Peak unrealized PnL du trade en cours
        daily_start_equity = None    # Equity début de journée
        current_day = None           # Date courante
        daily_stopped = False        # Flag: stop pour la journée

        prev_ts = None
        prev_p1 = prev_p2 = None

        pending_sig: int = 0
        pending_close: bool = False

        def _fee(notional: float) -> float:
            return float(p.fee_rate) * float(abs(notional))

        # -------------------------------------------------------
        # HELPER: ferme la position (factorisé)
        # -------------------------------------------------------
        def _close_position(close_ts, close_p1, close_p2, reason="signal"):
            nonlocal pos, entry_ts, entry_p1, entry_p2
            nonlocal gross_pnl, net_pnl, fees_paid
            nonlocal current_equity, current_equity_gross
            nonlocal hwm, peak_trade_pnl

            notional = abs(q1 * close_p1) + abs(q2 * close_p2)
            fee = _fee(notional)
            fees_paid += fee
            net_pnl -= fee

            is_stop = reason not in ("signal", "week_end")

            trades.append(dict(
                cycle=cycle_id,
                pair=f"{coin1}-{coin2}",
                copula=best_name,
                entry_time=str(entry_ts),
                exit_time=str(close_ts),
                direction=("LONG1_SHORT2" if pos == +1 else "SHORT1_LONG2"),
                q1=q1, q2=q2,
                entry_p1=entry_p1, entry_p2=entry_p2,
                exit_p1=close_p1, exit_p2=close_p2,
                gross_pnl=gross_pnl,
                net_pnl=net_pnl,
                fees=fees_paid,
                bars=int((pd.to_datetime(close_ts) - pd.to_datetime(
                    entry_ts)).total_seconds() // 60) if entry_ts else np.nan,
                forced_week_end=(reason == "week_end"),
                exit_reason=reason,
            ))

            if is_stop:
                stop_loss_events.append(dict(
                    time=str(close_ts),
                    type=reason,
                    detail=f"PnL={net_pnl:.2f}",
                    equity=current_equity + net_pnl,
                ))

            current_equity = float(current_equity + net_pnl)
            current_equity_gross = float(current_equity_gross + gross_pnl)

            # Update HWM
            if current_equity > hwm:
                hwm = current_equity

            pos = 0
            entry_ts = None
            entry_p1 = entry_p2 = None
            gross_pnl = 0.0
            net_pnl = 0.0
            fees_paid = 0.0
            peak_trade_pnl = 0.0

        # -------------------------------------------------------
        # Boucle de trading (bar par bar)
        # -------------------------------------------------------
        trade_bars = df_trade[[p.ref, coin1, coin2]].dropna()
        for ts, row in trade_bars.iterrows():
            pref = float(row[p.ref])
            p1 = float(row[coin1])
            p2 = float(row[coin2])

            # --- RISK MANAGEMENT: Daily DD tracking ---
            ts_dt = pd.to_datetime(ts) if not isinstance(ts, pd.Timestamp) else ts
            ts_date = ts_dt.date()
            if current_day != ts_date:
                current_day = ts_date
                daily_start_equity = current_equity + net_pnl
                daily_stopped = False

            # 1) mark-to-market PnL since prev bar
            if prev_ts is not None and pos != 0:
                dp1 = p1 - float(prev_p1)
                dp2 = p2 - float(prev_p2)
                if pos == +1:
                    step_pnl = q1 * dp1 - q2 * dp2
                else:
                    step_pnl = -q1 * dp1 + q2 * dp2
                gross_pnl += step_pnl
                net_pnl += step_pnl

            # ===================================================
            # RISK MANAGEMENT: vérifications stop-loss
            # ===================================================
            stop_triggered = False

            if pos != 0:
                # A) Trade-level stop-loss
                if p.use_trade_stop_loss and not stop_triggered:
                    sl_threshold = -abs(p.trade_stop_loss_pct) * total_notional
                    if net_pnl < sl_threshold:
                        _close_position(ts, p1, p2, reason="TRADE_STOP_LOSS")
                        stop_triggered = True
                        pending_close = False
                        pending_sig = 0

                # B) Trailing stop
                if p.use_trailing_stop and not stop_triggered and pos != 0:
                    if net_pnl > peak_trade_pnl:
                        peak_trade_pnl = net_pnl
                    activation_level = abs(p.trailing_stop_activation) * total_notional
                    if peak_trade_pnl > activation_level:
                        trail_threshold = peak_trade_pnl - abs(p.trailing_stop_pct) * total_notional
                        if net_pnl < trail_threshold:
                            _close_position(ts, p1, p2, reason="TRAILING_STOP")
                            stop_triggered = True
                            pending_close = False
                            pending_sig = 0

                # C) Daily drawdown limit
                if p.use_daily_drawdown_limit and not stop_triggered and daily_start_equity is not None and pos != 0:
                    intraday_equity = current_equity + net_pnl
                    daily_dd = (intraday_equity - daily_start_equity) / daily_start_equity if daily_start_equity > 0 else 0.0
                    if daily_dd < -abs(p.daily_drawdown_limit_pct):
                        _close_position(ts, p1, p2, reason="DAILY_DD_LIMIT")
                        stop_triggered = True
                        daily_stopped = True
                        pending_close = False
                        pending_sig = 0

            # Si daily stopped => pas de nouveaux trades
            if daily_stopped:
                pending_sig = 0

            # 2) execute pending orders from previous bar (si pas de stop trigger)
            if not stop_triggered:
                notional1 = q1 * p1
                notional2 = q2 * p2
                trade_notional = abs(notional1) + abs(notional2)

                # pending close first
                if pos != 0 and pending_close:
                    _close_position(ts, p1, p2, reason="signal")

                pending_close = False

                # pending open/flip
                if pending_sig != 0:
                    if pos == 0:
                        # open at current prices
                        fee = _fee(trade_notional)
                        fees_paid += fee
                        net_pnl -= fee
                        pos = int(pending_sig)
                        entry_ts = ts
                        entry_p1, entry_p2 = p1, p2
                        peak_trade_pnl = 0.0
                    else:
                        # flip at current prices (close then open) if allowed
                        if p.flip_on_opposite and int(pending_sig) != int(pos):
                            _close_position(ts, p1, p2, reason="signal")

                            # open new position
                            fee_open = _fee(trade_notional)
                            fees_paid += fee_open
                            net_pnl -= fee_open
                            pos = int(pending_sig)
                            entry_ts = ts
                            entry_p1, entry_p2 = p1, p2
                            peak_trade_pnl = 0.0

                pending_sig = 0

            # 3) compute current spreads + signal for next bar
            s1_val = float(pref - beta1 * p1) if np.isfinite(beta1) else np.nan
            s2_val = float(pref - beta2 * p2) if np.isfinite(beta2) else np.nan
            if not np.isfinite(s1_val) or not np.isfinite(s2_val):
                sig = 0
                close_now = False
            else:
                sig, det = generate_signals_reference_copula(
                    cop=cop, sorted_s1=s1_sorted, sorted_s2=s2_sorted,
                    s1_val=s1_val, s2_val=s2_val,
                    entry=p.entry, exit=p.exit
                )
                close_now = bool(det.get("close", 0.0) > 0.5)

            # schedule execution at next bar (sauf si daily stopped)
            if not daily_stopped:
                if pos != 0 and close_now:
                    pending_close = True
                elif sig != 0:
                    pending_sig = int(sig)

            # 4) update equity marks
            equity.loc[ts] = current_equity + net_pnl
            equity_gross.loc[ts] = current_equity_gross + gross_pnl

            prev_ts = ts
            prev_p1, prev_p2 = p1, p2

        # --- End of week: force-close ---
        last_bar = df_trade[[coin1, coin2]].dropna().tail(1)
        if not last_bar.empty:
            ts_end = last_bar.index[0]
            p1_end = float(last_bar.iloc[0][coin1])
            p2_end = float(last_bar.iloc[0][coin2])

            if pos != 0:
                _close_position(ts_end, p1_end, p2_end, reason="week_end")

            # fill equity series for bars not set
            for ts in df_trade.index:
                if pd.isna(equity.loc[ts]):
                    equity.loc[ts] = current_equity
                if pd.isna(equity_gross.loc[ts]):
                    equity_gross.loc[ts] = current_equity_gross

        weekly.append(dict(
            cycle=cycle_id,
            formation_start=str(t0), formation_end=str(t_form_end),
            trade_start=str(t_form_end), trade_end=str(t_trade_end),
            status="OK",
            selected_pair=f"{coin1}-{coin2}",
            copula=best_name,
            beta1=float(beta1) if np.isfinite(beta1) else np.nan,
            beta2=float(beta2) if np.isfinite(beta2) else np.nan,
            q1=q1, q2=q2,
            top_list=",".join(top),
            fit_messages="; ".join(msgs) if msgs else "",
        ))

        t0 = t0 + step_delta
        cycle_id += 1

    equity = equity.ffill().dropna()
    equity_gross = equity_gross.ffill().dropna()

    trades_df = pd.DataFrame(trades)
    weekly_df = pd.DataFrame(weekly)

    # perf metrics
    metrics = performance_metrics(equity, p.interval)
    metrics_g = performance_metrics(equity_gross, p.interval)

    # monthly returns from net equity
    monthly = None
    if len(equity) > 10:
        eq_m = equity.resample("M").last()
        monthly = eq_m.pct_change().dropna()
    else:
        monthly = pd.Series(dtype=float)

    # copula frequency
    cop_freq = None
    if not weekly_df.empty and "copula" in weekly_df.columns:
        cop_freq = weekly_df["copula"].value_counts().reset_index()
        cop_freq.columns = ["copula", "count"]
    else:
        cop_freq = pd.DataFrame(columns=["copula", "count"])

    # --- RISK MANAGEMENT: stop-loss stats ---
    sl_stats = {}
    if trades_df is not None and not trades_df.empty and "exit_reason" in trades_df.columns:
        reason_counts = trades_df["exit_reason"].value_counts().to_dict()
        sl_stats["reason_counts"] = {str(k): int(v) for k, v in reason_counts.items()}
        sl_types = ["TRADE_STOP_LOSS", "TRAILING_STOP", "DAILY_DD_LIMIT"]
        sl_trades = trades_df[trades_df["exit_reason"].isin(sl_types)]
        sl_stats["total_stop_loss_trades"] = int(sl_trades.shape[0])
        sl_stats["total_sl_pnl"] = float(sl_trades["net_pnl"].sum()) if not sl_trades.empty else 0.0
    else:
        sl_stats["reason_counts"] = {}
        sl_stats["total_stop_loss_trades"] = 0
        sl_stats["total_sl_pnl"] = 0.0
    sl_stats["events"] = stop_loss_events
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