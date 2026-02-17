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

    current_equity = float(p.initial_equity)
    current_equity_gross = float(p.initial_equity)

    # iterate cycles
    t0 = start_dt
    cycle_id = 0

    while True:
        t_form_end = t0 + formation_delta
        t_trade_end = t_form_end + trading_delta
        if t_trade_end > end_dt:
            break

        df_form = prices.loc[(prices.index >= t0) & (prices.index < t_form_end)]
        df_trade = prices.loc[(prices.index >= t_form_end) & (prices.index < t_trade_end)]

        # stationarity filter on formation
        # 1) candidates: coins dispo dans ce df_form
        candidates = [s for s in p.symbols if (s != p.ref and s in df_form.columns)]

        # 2) calcule coverage / points SUR LA FORMATION WINDOW uniquement
        ref_ok = df_form[p.ref].notna()
        form_slice = df_form.loc[ref_ok, candidates]

        coverage = form_slice.notna().mean()  # % de points non-NaN
        points = form_slice.notna().sum()  # nb de points non-NaN

        # 3) garde seulement ceux qui passent les seuils
        eligible = [
            s for s in candidates
            if float(coverage.get(s, 0.0)) >= float(p.min_coverage)
               and int(points.get(s, 0)) >= int(p.min_obs)
        ]

        # 4) stationarity test uniquement sur eligible
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
            # write equity (flat) over trading window
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

        # quantities fixed for the week using opening prices (paper: <= 20k USDT each coin)
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

        pos = 0  # +1 long coin1 short coin2 ; -1 short coin1 long coin2 ; 0 flat
        entry_ts = None
        entry_p1 = entry_p2 = None
        entry_fee = 0.0
        gross_pnl = 0.0
        net_pnl = 0.0
        fees_paid = 0.0

        prev_ts = None
        prev_p1 = prev_p2 = None

        # run through trading window bars (signals at t, execution at t+1 -> apply pending orders)
        pending_sig: int = 0
        pending_close: bool = False

        def _fee(notional: float) -> float:
            return float(p.fee_rate) * float(abs(notional))

        # run through trading window bars (only where ref/coin1/coin2 prices exist)
        trade_bars = df_trade[[p.ref, coin1, coin2]].dropna()
        for ts, row in trade_bars.iterrows():
            pref = float(row[p.ref])
            p1 = float(row[coin1])
            p2 = float(row[coin2])

            # 1) mark-to-market PnL since prev bar (position held during prev->ts)
            if prev_ts is not None and pos != 0:
                dp1 = p1 - float(prev_p1)  # type: ignore
                dp2 = p2 - float(prev_p2)  # type: ignore
                if pos == +1:
                    step_pnl = q1 * dp1 - q2 * dp2
                else:
                    step_pnl = -q1 * dp1 + q2 * dp2
                gross_pnl += step_pnl
                net_pnl += step_pnl

            # 2) execute pending orders from previous bar at *current* prices
            notional1 = q1 * p1
            notional2 = q2 * p2
            trade_notional = abs(notional1) + abs(notional2)

            # pending close first
            if pos != 0 and pending_close:
                fee = _fee(trade_notional)
                fees_paid += fee
                net_pnl -= fee
                trades.append(dict(
                    cycle=cycle_id,
                    pair=f"{coin1}-{coin2}",
                    copula=best_name,
                    entry_time=str(entry_ts),
                    exit_time=str(ts),
                    direction=("LONG1_SHORT2" if pos == +1 else "SHORT1_LONG2"),
                    q1=q1, q2=q2,
                    entry_p1=entry_p1, entry_p2=entry_p2,
                    exit_p1=p1, exit_p2=p2,
                    gross_pnl=gross_pnl,
                    net_pnl=net_pnl,
                    fees=fees_paid,
                    bars=int(
                        (pd.to_datetime(ts) - pd.to_datetime(entry_ts)).total_seconds() // 60) if entry_ts else np.nan,
                    forced_week_end=False,
                ))
                # realize PnL at execution time
                current_equity = float(current_equity + net_pnl)
                current_equity_gross = float(current_equity_gross + gross_pnl)
                pos = 0
                entry_ts = None
                entry_p1 = entry_p2 = None
                gross_pnl = 0.0
                net_pnl = 0.0
                fees_paid = 0.0

            pending_close = False  # consumed

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
                else:
                    # flip at current prices (close then open) if allowed
                    if p.flip_on_opposite and int(pending_sig) != int(pos):
                        fee_close = _fee(trade_notional)
                        fees_paid += fee_close
                        net_pnl -= fee_close
                        trades.append(dict(
                            cycle=cycle_id,
                            pair=f"{coin1}-{coin2}",
                            copula=best_name,
                            entry_time=str(entry_ts),
                            exit_time=str(ts),
                            direction=("LONG1_SHORT2" if pos == +1 else "SHORT1_LONG2"),
                            q1=q1, q2=q2,
                            entry_p1=entry_p1, entry_p2=entry_p2,
                            exit_p1=p1, exit_p2=p2,
                            gross_pnl=gross_pnl,
                            net_pnl=net_pnl,
                            fees=fees_paid,
                            bars=int((pd.to_datetime(ts) - pd.to_datetime(
                                entry_ts)).total_seconds() // 60) if entry_ts else np.nan,
                            forced_week_end=False,
                        ))
                        # realize PnL at flip-close
                        current_equity = float(current_equity + net_pnl)
                        current_equity_gross = float(current_equity_gross + gross_pnl)

                        # reset and open new
                        pos = 0
                        entry_ts = None
                        entry_p1 = entry_p2 = None
                        gross_pnl = 0.0
                        net_pnl = 0.0
                        fees_paid = 0.0

                        fee_open = _fee(trade_notional)
                        fees_paid += fee_open
                        net_pnl -= fee_open
                        pos = int(pending_sig)
                        entry_ts = ts
                        entry_p1, entry_p2 = p1, p2

            pending_sig = 0  # consumed

            # 3) compute current spreads (using betas from formation) + signal for next bar
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

            # schedule execution at next bar
            if pos != 0 and close_now:
                pending_close = True
            elif sig != 0:
                pending_sig = int(sig)

            # 4) update equity marks (equity series is global portfolio)
            equity.loc[ts] = current_equity + net_pnl  # net_pnl inclut fees/unrealized
            equity_gross.loc[ts] = current_equity_gross + gross_pnl

            prev_ts = ts
            prev_p1, prev_p2 = p1, p2

        last_bar = df_trade[[coin1, coin2]].dropna().tail(1)
        if not last_bar.empty:
            ts_end = last_bar.index[0]
            p1_end = float(last_bar.iloc[0][coin1])
            p2_end = float(last_bar.iloc[0][coin2])

            if pos != 0:
                notional_end = abs(q1 * p1_end) + abs(q2 * p2_end)
                fee = float(p.fee_rate) * notional_end
                fees_paid += fee
                net_pnl -= fee
                trades.append(dict(
                    cycle=cycle_id,
                    pair=f"{coin1}-{coin2}",
                    copula=best_name,
                    entry_time=str(entry_ts),
                    exit_time=str(ts_end),
                    direction=("LONG1_SHORT2" if pos == +1 else "SHORT1_LONG2"),
                    q1=q1, q2=q2,
                    entry_p1=entry_p1, entry_p2=entry_p2,
                    exit_p1=p1_end, exit_p2=p2_end,
                    gross_pnl=gross_pnl,
                    net_pnl=net_pnl,
                    fees=fees_paid,
                    bars=int((pd.to_datetime(ts_end) - pd.to_datetime(entry_ts)).total_seconds() // 60) if entry_ts else np.nan,
                    forced_week_end=True
                ))
                pos = 0

            # update equity at week end and roll portfolio equity forward
            current_equity = float(current_equity + net_pnl)
            current_equity_gross = float(current_equity_gross + gross_pnl)

            # fill equity series for bars not set (if any)
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
    )