# dash_bot/ui/serialization.py
from __future__ import annotations

from typing import Any, Dict
import numpy as np
import pandas as pd


def _serialize_dataframe(df: pd.DataFrame | None) -> dict:
    if df is None:
        return {"columns": [], "index": [], "data": []}
    df = pd.DataFrame(df).copy().replace([np.inf, -np.inf], np.nan)

    idx = df.index
    if isinstance(idx, pd.DatetimeIndex):
        index = [x.isoformat() if not pd.isna(x) else None for x in idx.to_pydatetime()]
    else:
        index = [None if x is None else str(x) for x in idx.tolist()]

    return {"columns": list(df.columns), "index": index, "data": df.to_numpy().tolist()}


def _deserialize_dataframe(payload: dict | None) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame()
    cols = payload.get("columns", [])
    idx = payload.get("index", [])
    data = payload.get("data", [])

    df = pd.DataFrame(data, columns=cols)
    try:
        df.index = pd.to_datetime(idx, errors="raise")
    except Exception:
        df.index = idx
    return df


def _serialize_series(s: pd.Series | None) -> dict:
    if s is None:
        return {"index": [], "values": []}
    s = pd.Series(s).copy().replace([np.inf, -np.inf], np.nan)

    idx = s.index
    if isinstance(idx, pd.DatetimeIndex):
        index = [x.isoformat() if not pd.isna(x) else None for x in idx.to_pydatetime()]
    else:
        index = [None if x is None else str(x) for x in idx.tolist()]

    values = [None if pd.isna(v) else float(v) for v in s.values]
    return {"index": index, "values": values}

def _deserialize_series(obj: Dict[str, Any]) -> pd.Series:
    idx = pd.to_datetime(obj.get("index", []))
    vals = obj.get("values", [])
    s = pd.Series(vals, index=idx)
    s = s[~s.index.isna()]
    return s.sort_index()

# def _deserialize_series(payload: dict | None) -> pd.Series:
#     if not payload:
#         return pd.Series(dtype="float64")
#     idx = payload.get("index", [])
#     vals = payload.get("values", [])

#     try:
#         index = pd.to_datetime(idx, errors="raise")
#     except Exception:
#         index = idx

#     s = pd.Series(vals, index=index, dtype="float64")
#     s = s.replace([np.inf, -np.inf], np.nan)
#     if isinstance(s.index, pd.DatetimeIndex):
#         s = s[~s.index.isna()].sort_index()
#     return s


# def serialize_results(results: dict) -> dict:
#     out: Dict[str, Any] = {}

#     out["metrics"] = results.get("metrics", {}) or {}
#     out["trades"] = results.get("trades", []) or []
#     out["weekly"] = results.get("weekly", []) or []
#     out["copula_freq"] = results.get("copula_freq", []) or []

#     out["equity"] = _serialize_series(results.get("equity"))
#     out["equity_gross"] = _serialize_series(results.get("equity_gross"))
#     out["monthly_returns"] = _serialize_series(results.get("monthly_returns"))

#     out["params"] = results.get("params", {})  # optionnel

#     return out

def _serialize_weekly(weekly):
    """Serialize weekly DataFrame/list with diagnostic data for JSON storage."""
    if isinstance(weekly, pd.DataFrame):
        records = weekly.to_dict("records")
    elif isinstance(weekly, list):
        records = weekly
    else:
        return []

    for rec in records:
        # Convert numpy arrays to lists
        for key in ("s1_sorted", "s2_sorted", "cop_params"):
            val = rec.get(key)
            if val is not None:
                if isinstance(val, np.ndarray):
                    rec[key] = val.tolist()
                elif not isinstance(val, list):
                    try:
                        rec[key] = list(val)
                    except Exception:
                        rec[key] = []

        # Convert diag_bars timestamps to strings and ensure JSON-safe types
        diag = rec.get("diag_bars")
        if diag and isinstance(diag, list):
            for bar in diag:
                ts = bar.get("ts")
                if ts is not None and not isinstance(ts, str):
                    bar["ts"] = str(ts)
                for k, v in list(bar.items()):
                    if isinstance(v, (np.floating, np.integer)):
                        bar[k] = float(v)
                    elif isinstance(v, np.bool_):
                        bar[k] = bool(v)

    return records


def serialize_results(res: Dict[str, Any]) -> Dict[str, Any]:
    def ser_series(s: pd.Series) -> Dict[str, Any]:
        return {"index": s.index.astype(str).tolist(), "values": s.values.tolist()}

    out = {
        "equity": ser_series(res["equity"]),
        "equity_gross": ser_series(res["equity_gross"]),
        "metrics": res["metrics"],
        "metrics_gross": res["metrics_gross"],
        "trades": res["trades"].to_dict("records") if isinstance(res["trades"], pd.DataFrame) else [],
        "weekly": _serialize_weekly(res["weekly"]),
        "copula_freq": res["copula_freq"].to_dict("records") if isinstance(res["copula_freq"], pd.DataFrame) else [],
        "monthly_returns": ser_series(res["monthly_returns"]) if isinstance(res["monthly_returns"], pd.Series) else {"index": [], "values": []},
        "params": res["params"].__dict__,
    }

    out["stop_loss_stats"] = res.get("stop_loss_stats", {})
    return out

def deserialize_results(payload: dict) -> dict:
    if not payload:
        return {}
    out = dict(payload)
    out["equity"] = _deserialize_series(payload.get("equity"))
    out["equity_gross"] = _deserialize_series(payload.get("equity_gross"))
    out["monthly_returns"] = _deserialize_series(payload.get("monthly_returns"))
    return out