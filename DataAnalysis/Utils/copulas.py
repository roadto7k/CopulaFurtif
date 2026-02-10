"""DataAnalysis copula fitting helpers.

Used by the interactive dashboards and by :
- dash_bot.py
- DataAnalysis/dash_copula_app.py
- DataAnalysis/main_plotly*.py

Key change
----------
We used to select the "best" copula by AIC only.
We now support a multi-metric ranking that matches the project demo:
`copulafurtif/CopulaFurtif/debugging/demo_fitting_tau_then_cmle.py`.

Pipeline (default = fast enough for dashboards/backtests)
--------------------------------------------------------
Stage 1:
  1) Fit each candidate with `fit_tau` (moments init; very fast)
  2) Compute copula-only log-likelihood on pseudo-observations to enable AIC
  3) Compute diagnostics via `copula.gof_summary(...)`:
       - AIC
       - Kendall tau error (KT_err)
       - Huang tail dependence mismatch (TailGap)
       - Optional Rosenblatt PIT KS statistic (PIT)
  4) Combine metrics into a robust z-score score (lower = better)

Stage 2 (optional):
  - Refine the top-K candidates with a quick CMLE pass and re-score.

The public API stays compatible with the old dashboards:
`fit_copulas(x, y) -> (df, msgs)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# CopulaFurtif
from CopulaFurtif.copulas import CopulaFactory, CopulaType
from CopulaFurtif.copulas import CopulaFitter
from CopulaFurtif.core.copulas.domain.estimation.estimation import pseudo_obs as _pseudo_obs_pair
# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

HAS_COPULAFURTIF = True


@dataclass(frozen=True)
class CopulaSelectionConfig:
    """Config controlling speed/precision of the ranking."""

    # Candidate family whitelist (keep stable / implemented ones)
    candidates: Sequence[Tuple[str, CopulaType]] = (
        ("Gaussian", CopulaType.GAUSSIAN),
        ("Student-t", CopulaType.STUDENT),
        ("Clayton", CopulaType.CLAYTON),
        ("Gumbel", CopulaType.GUMBEL),
        ("Frank", CopulaType.FRANK),
        ("Galambos", CopulaType.GALAMBOS),
        ("Joe", CopulaType.JOE),
        ("Plackett", CopulaType.PLACKETT),
    )

    # Selection controls
    include_pit: bool = False
    pit_m: int = 250
    refine_topk: int = 0  # 0 = no CMLE refinement

    # Scoring weights
    w_aic: float = 1.00
    w_kt: float = 0.75
    w_tail: float = 1.00
    w_pit: float = 1.00

    # Numerical stability
    eps_uv: float = 1e-10
    eps_pdf: float = 1e-12


# -----------------------------------------------------------------------------
# Backwards-compatible helpers (used by some Plotly dashboards)
# -----------------------------------------------------------------------------
def pseudo_obs(x: Sequence[float]) -> np.ndarray:
    """1D pseudo-observations: rank(x)/(n+1), mapped to (0,1)."""
    x = np.asarray(list(x), dtype=float).ravel()
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return np.asarray([], dtype=float)
    ranks = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    return ranks / (n + 1.0)


def aic_val(loglik: float, k: int) -> float:
    """AIC = 2k - 2 loglik. Returns NaN if loglik is not finite."""
    return float(2 * k - 2 * loglik) if np.isfinite(loglik) else float('nan')


# -----------------------------------------------------------------------------
# Internals
# -----------------------------------------------------------------------------


def _flatten_params(theta: Any) -> Tuple[float, ...]:
    try:
        arr = np.ravel(theta).astype(float)
        return tuple(float(x) for x in arr)
    except Exception:
        return (float("nan"),)


def _call_pdf(copula: Any, u: np.ndarray, v: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Compat: some models accept get_pdf(u,v,theta), others only get_pdf(u,v)."""
    try:
        return copula.get_pdf(u, v, theta)
    except TypeError:
        return copula.get_pdf(u, v)


def _compute_copula_only_loglik(
    copula: Any,
    data_xy: Tuple[np.ndarray, np.ndarray],
    *,
    eps_uv: float,
    eps_pdf: float,
) -> Tuple[float, int]:
    """log L_c(theta) = sum log c_theta(u_i, v_i) on pseudo-observations."""
    u, v = _pseudo_obs_pair(data_xy)
    u = np.clip(u, eps_uv, 1.0 - eps_uv)
    v = np.clip(v, eps_uv, 1.0 - eps_uv)

    theta = np.asarray(copula.get_parameters(), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        pdf = _call_pdf(copula, u, v, theta)

    pdf = np.asarray(pdf, dtype=float)
    pdf = np.clip(pdf, eps_pdf, None)
    ll = float(np.sum(np.log(pdf)))
    return ll, int(len(u))


def _tail_gap_from_summary(summ: Dict[str, Any]) -> float:
    """Scalar tail mismatch used for ranking."""
    a = summ.get("lambdaL_emp_huang", np.nan)
    b = summ.get("lambdaL_model", np.nan)
    c = summ.get("lambdaU_emp_huang", np.nan)
    d = summ.get("lambdaU_model", np.nan)

    vals = [abs(a - b), abs(c - d)]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.sum(vals)) if vals else np.nan


def _robust_z(x: np.ndarray) -> np.ndarray:
    """Robust z-score using median/MAD; NaN-safe."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad < 1e-12:
        return x - med
    return (x - med) / (1.4826 * mad)


def _fill_missing(x: np.ndarray, *, mode: str) -> np.ndarray:
    """Fill NaN in a metric vector.

    mode="neutral": replace by median (doesn't help/hurt much)
    mode="penalize": replace by (max + margin) (hurts)
    """
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    if not ok.any():
        return np.zeros_like(x)

    out = x.copy()
    if mode == "neutral":
        fill = float(np.nanmedian(x[ok]))
    else:
        fill = float(np.nanmax(x[ok]) + 0.05)

    out[~ok] = fill
    return out


def _score_stage(
    df_ok: pd.DataFrame,
    *,
    include_pit: bool,
    w_aic: float,
    w_kt: float,
    w_tail: float,
    w_pit: float,
    prefix: str,
) -> pd.Series:
    """Compute a robust score from columns.

    Expected columns:
      - {prefix}aic
      - {prefix}kt_err
      - {prefix}tail_gap
      - {prefix}pit (optional)

    Lower score is better.
    """
    aic = df_ok[f"{prefix}aic"].astype(float).to_numpy()
    kt = df_ok[f"{prefix}kt_err"].astype(float).to_numpy()
    tail = df_ok[f"{prefix}tail_gap"].astype(float).to_numpy()

    if include_pit and f"{prefix}pit" in df_ok.columns:
        pit = df_ok[f"{prefix}pit"].astype(float).to_numpy()
    else:
        pit = np.full(len(df_ok), np.nan, dtype=float)

    aic_f = _fill_missing(aic, mode="neutral")
    kt_f = _fill_missing(kt, mode="neutral")
    tail_f = _fill_missing(tail, mode="neutral")
    pit_f = _fill_missing(pit, mode="penalize") if include_pit else np.zeros_like(aic_f)

    z_aic = _robust_z(aic_f)
    z_kt = _robust_z(kt_f)
    z_tail = _robust_z(tail_f)
    z_pit = _robust_z(pit_f) if include_pit else np.zeros_like(z_aic)

    score = w_aic * z_aic + w_kt * z_kt + w_tail * z_tail + (w_pit * z_pit if include_pit else 0.0)
    return pd.Series(score, index=df_ok.index)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def fit_copulas(
    x: np.ndarray,
    y: np.ndarray,
    *,
    selection: str = "score",  # {"score","aic"}
    include_pit: bool = False,
    pit_m: int = 250,
    refine_topk: int = 0,
    candidates: Optional[Sequence[Tuple[str, CopulaType]]] = None,
    # weights
    w_aic: float = 1.00,
    w_kt: float = 0.75,
    w_tail: float = 1.00,
    w_pit: float = 1.00,
    # numerical
    eps_uv: float = 1e-10,
    eps_pdf: float = 1e-12,
) -> Tuple[pd.DataFrame, List[str]]:
    """Fit and rank bivariate copulas on raw samples (x,y).

    Parameters
    ----------
    selection:
      - "score": multi-metric robust score (AIC + tau + tails [+ optional PIT])
      - "aic"  : AIC only

    Returns
    -------
    df, msgs
      df includes at least: name, type, params, loglik, aic
      plus: kt_err, tail_gap, pit, score_stage1, ...
    """

    msgs: List[str] = []
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    if x.size == 0 or y.size == 0:
        return (
            pd.DataFrame(columns=["name", "type", "params", "loglik", "aic", "kt_err", "tail_gap", "pit", "score_stage1"]),
            ["Pas de données."],
        )

    if candidates is None:
        candidates = CopulaSelectionConfig().candidates

    fitter = CopulaFitter()

    rows: List[Dict[str, Any]] = []

    # ---- Stage 1: tau init + diagnostics
    for name, ctype in candidates:
        try:
            cop = CopulaFactory.create(ctype)

            # init params
            fitter.fit_tau((x, y), copula=cop)

            # copula-only loglik for AIC
            ll, n_obs = _compute_copula_only_loglik(cop, (x, y), eps_uv=eps_uv, eps_pdf=eps_pdf)
            cop.log_likelihood_ = ll
            cop.n_obs = n_obs

            # gof bundle
            summ = cop.gof_summary(
                (x, y),
                include_aic=True,
                include_bic=False,
                include_tau=True,
                include_tails=True,
                include_ad=False,
                include_iad=False,
                include_pit=bool(include_pit),
                pit_m=int(pit_m),
                pit_seed=0,
                pit_q_tail=0.10,
                pit_tail_frac=0.33,
            )

            aic = float(summ.get("AIC", np.nan))
            kt_err = float(summ.get("KT_err", np.nan))
            tail_gap = _tail_gap_from_summary(summ)
            pit = float(summ.get("PIT", np.nan)) if include_pit else np.nan

            # store canonical tail deps too (for display)
            tdL = float(summ.get("lambdaL_model", np.nan))
            tdU = float(summ.get("lambdaU_model", np.nan))

            row = {
                "name": name,
                "type": ctype,
                "params": np.asarray(cop.get_parameters(), dtype=float),
                "params_tuple": _flatten_params(cop.get_parameters()),
                "loglik": ll,
                "aic": aic,
                "kt_err": kt_err,
                "tail_gap": tail_gap,
                "pit": pit,
                "tail_dep_L": tdL,
                "tail_dep_U": tdU,
            }

            # also keep the raw gof keys (helpful for debugging in dashboards)
            for k in (
                "lambdaL_emp_huang",
                "lambdaU_emp_huang",
                "lambdaL_model",
                "lambdaU_model",
                "PIT_ks_D",
                "PIT_ks_pvalue",
                "PIT_indep_tau",
                "PIT_loU_ks_D",
                "PIT_hiU_ks_D",
            ):
                if k in summ:
                    row[k] = summ.get(k)

            rows.append(row)

        except Exception as e:
            msgs.append(f"{name} fit failed: {e}")
            rows.append({
                "name": name,
                "type": ctype,
                "params": np.array([], dtype=float),
                "params_tuple": (),
                "loglik": np.nan,
                "aic": np.nan,
                "kt_err": np.nan,
                "tail_gap": np.nan,
                "pit": np.nan,
                "tail_dep_L": np.nan,
                "tail_dep_U": np.nan,
                "error": str(e),
            })

    df = pd.DataFrame(rows)

    # keep only successful rows for ranking
    ok_mask = df["aic"].apply(lambda v: np.isfinite(v))
    df_ok = df[ok_mask].copy()

    if df_ok.empty:
        return df, msgs + ["Aucune copula n'a pu être évaluée (AIC/loglik manquant)."]

    # Stage1 score
    df_ok["score_stage1"] = _score_stage(
        df_ok.rename(
            columns={
                "aic": "s1_aic",
                "kt_err": "s1_kt_err",
                "tail_gap": "s1_tail_gap",
                "pit": "s1_pit",
            }
        ),
        include_pit=bool(include_pit),
        w_aic=w_aic,
        w_kt=w_kt,
        w_tail=w_tail,
        w_pit=w_pit,
        prefix="s1_",
    )

    # merge back to df
    df = df.merge(df_ok[["name", "score_stage1"]], on="name", how="left")

    # ---- Stage 2: optional CMLE refinement for top-k
    if refine_topk and int(refine_topk) > 0 and HAS_COPULAFURTIF:
        k = min(int(refine_topk), len(df_ok))

        # candidates to refine (best stage1)
        df_ok_sorted = df_ok.sort_values("score_stage1", kind="mergesort").reset_index(drop=True)
        refine_names = df_ok_sorted.loc[: k - 1, "name"].tolist()

        refined_rows: Dict[str, Dict[str, Any]] = {}

        for nm in refine_names:
            ctype = df.loc[df["name"] == nm, "type"].iloc[0]
            try:
                cop = CopulaFactory.create(ctype)
                fitter.fit_tau((x, y), copula=cop)

                # quick CMLE refine
                fitter.fit_cmle((x, y), copula=cop, quick=True, verbose=False, return_metrics=False)

                ll2, n_obs2 = _compute_copula_only_loglik(cop, (x, y), eps_uv=eps_uv, eps_pdf=eps_pdf)
                cop.log_likelihood_ = ll2
                cop.n_obs = n_obs2

                summ2 = cop.gof_summary(
                    (x, y),
                    include_aic=True,
                    include_bic=False,
                    include_tau=True,
                    include_tails=True,
                    include_ad=False,
                    include_iad=False,
                    include_pit=bool(include_pit),
                    pit_m=int(pit_m),
                    pit_seed=0,
                    pit_q_tail=0.10,
                    pit_tail_frac=0.33,
                )

                refined_rows[nm] = {
                    "params_refined": np.asarray(cop.get_parameters(), dtype=float),
                    "loglik_refined": ll2,
                    "aic_refined": float(summ2.get("AIC", np.nan)),
                    "kt_err_refined": float(summ2.get("KT_err", np.nan)),
                    "tail_gap_refined": _tail_gap_from_summary(summ2),
                    "pit_refined": float(summ2.get("PIT", np.nan)) if include_pit else np.nan,
                }
            except Exception as e:
                refined_rows[nm] = {
                    "params_refined": None,
                    "loglik_refined": np.nan,
                    "aic_refined": np.nan,
                    "kt_err_refined": np.nan,
                    "tail_gap_refined": np.nan,
                    "pit_refined": np.nan,
                    "refine_error": str(e),
                }

        # attach refined columns
        for nm, rr in refined_rows.items():
            for col, val in rr.items():
                df.loc[df["name"] == nm, col] = [val]

        # compute stage2 score on refined subset that succeeded
        ref_ok = df["aic_refined"].apply(lambda v: np.isfinite(v)) if "aic_refined" in df.columns else pd.Series(False)
        df_ref = df[ref_ok].copy()
        if not df_ref.empty:
            tmp = df_ref.rename(
                columns={
                    "aic_refined": "s2_aic",
                    "kt_err_refined": "s2_kt_err",
                    "tail_gap_refined": "s2_tail_gap",
                    "pit_refined": "s2_pit",
                }
            )
            df_ref["score_stage2"] = _score_stage(
                tmp,
                include_pit=bool(include_pit),
                w_aic=w_aic,
                w_kt=w_kt,
                w_tail=w_tail,
                w_pit=w_pit,
                prefix="s2_",
            )
            df = df.merge(df_ref[["name", "score_stage2"]], on="name", how="left")

    # ---- Final sort
    sel = str(selection).lower().strip()
    if sel == "aic":
        sort_cols = ["aic"]
    else:
        # prefer stage2 when available
        sort_cols = ["score_stage2", "score_stage1"] if "score_stage2" in df.columns else ["score_stage1"]

    # stable sort, NaNs at bottom
    for c in reversed(sort_cols):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values(sort_cols, ascending=True, kind="mergesort", na_position="last").reset_index(drop=True)

    return df, msgs
