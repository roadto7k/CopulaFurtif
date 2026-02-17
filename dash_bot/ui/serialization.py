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

def serialize_results(res: Dict[str, Any]) -> Dict[str, Any]:
    def ser_series(s: pd.Series) -> Dict[str, Any]:
        return {"index": s.index.astype(str).tolist(), "values": s.values.tolist()}
    out = {
        "equity": ser_series(res["equity"]),
        "equity_gross": ser_series(res["equity_gross"]),
        "metrics": res["metrics"],
        "metrics_gross": res["metrics_gross"],
        "trades": res["trades"].to_dict("records") if isinstance(res["trades"], pd.DataFrame) else [],
        "weekly": res["weekly"].to_dict("records") if isinstance(res["weekly"], pd.DataFrame) else [],
        "copula_freq": res["copula_freq"].to_dict("records") if isinstance(res["copula_freq"], pd.DataFrame) else [],
        "monthly_returns": ser_series(res["monthly_returns"]) if isinstance(res["monthly_returns"], pd.Series) else {"index": [], "values": []},
        "params": res["params"].__dict__,
    }
    return out

def deserialize_results(payload: dict) -> dict:
    if not payload:
        return {}
    out = dict(payload)
    out["equity"] = _deserialize_series(payload.get("equity"))
    out["equity_gross"] = _deserialize_series(payload.get("equity_gross"))
    out["monthly_returns"] = _deserialize_series(payload.get("monthly_returns"))
    return out
