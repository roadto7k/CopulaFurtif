"""
Demo: Copula fitting (fast tau init) + diagnostics + optional CMLE refinement.

What this script does:
1) Fit a whitelist of copulas using fit_tau (very fast)
2) Compute copula-only log-likelihood on pseudo-observations so AIC works
3) Compute diagnostics: AIC, tau mismatch, tail mismatch, optional PIT (Rosenblatt KS)
4) Optionally refine TOP-K candidates with a quick CMLE pass, then recompute diagnostics
5) Print a readable final summary that explains everything

Run:
    python copulafurtif/CopulaFurtif/debugging/demo_fitting_tau_then_cmle.py
"""

import os
import sys
import inspect
import numpy as np
import pandas as pd

# -----------------------
# Ensure import path
# -----------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# .../copulafurtif/CopulaFurtif/debugging -> go up two -> .../copulafurtif
_COPULAFURTIF_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _COPULAFURTIF_ROOT not in sys.path:
    sys.path.insert(0, _COPULAFURTIF_ROOT)

from CopulaFurtif.core.copulas.domain.copula_type import CopulaType
from CopulaFurtif.core.copulas.domain.factories.copula_factory import CopulaFactory
from CopulaFurtif.core.copulas.application.services.fit_copula import CopulaFitter
from CopulaFurtif.core.copulas.domain.estimation.estimation import pseudo_obs
from scipy.stats import kendalltau


# -----------------------
# Utilities
# -----------------------

def _flatten_params(theta):
    """Return parameters as a stable tuple[float,...] for logging/dataframes."""
    try:
        arr = np.ravel(theta).astype(float)
        return tuple(float(x) for x in arr)
    except Exception:
        # last resort: string
        return (str(theta),)


def _call_pdf(copula, u, v, theta):
    """
    Compat: some copulas define get_pdf(u,v,theta),
    others define get_pdf(u,v) using internal parameters.
    """
    try:
        return copula.get_pdf(u, v, theta)
    except TypeError:
        return copula.get_pdf(u, v)


def _compute_copula_only_loglik(copula, data_xy, *, eps_uv=1e-10):
    """
    Compute copula-only log-likelihood on pseudo-observations:
        log L_c = sum log c_theta(u_i, v_i)
    Clip u,v away from 0/1 for numerical stability.
    """
    u, v = pseudo_obs(data_xy)

    # crucial: avoid 0/1 (Galambos, etc. can explode with u^{-d}, (u/v)^d ...)
    u = np.clip(u, eps_uv, 1.0 - eps_uv)
    v = np.clip(v, eps_uv, 1.0 - eps_uv)

    theta = copula.get_parameters()

    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        pdf = _call_pdf(copula, u, v, theta)

    pdf = np.asarray(pdf, float)
    pdf = np.clip(pdf, 1e-12, None)
    ll = float(np.sum(np.log(pdf)))
    return ll, int(len(u))


def _gof_summary_with_optional_pit(copula, data_xy, *, include_pit=True, pit_m=400):
    """
    Call copula.gof_summary(...) and include PIT args if your mixins supports it.
    If not supported, it falls back to summary without PIT.
    """
    sig = inspect.signature(copula.gof_summary)

    kwargs = dict(
        include_aic=True,
        include_bic=False,
        include_tau=True,
        include_tails=True,
        include_ad=False,
        include_iad=False,
    )

    if include_pit and ("include_pit" in sig.parameters):
        kwargs.update(dict(
            include_pit=True,
            pit_m=pit_m,
            pit_seed=0,
            pit_q_tail=0.10,
            pit_tail_frac=0.33,
        ))

    return copula.gof_summary(data_xy, **kwargs)


def _robust_z(x):
    """Robust z-score using median/MAD; NaN-safe."""
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad < 1e-12:
        return x - med
    return (x - med) / (1.4826 * mad)


def _safe_fit_cmle(fitter: CopulaFitter, data_xy, copula):
    """
    Call fitter.fit_cmle(...) in a signature-compatible way.
    We try to run a "quick" CMLE refinement if supported.
    """
    sig = inspect.signature(fitter.fit_cmle)
    kwargs = {}

    # the project may differ: quick / verbose / return_metrics exist or not
    if "quick" in sig.parameters:
        kwargs["quick"] = True
    if "verbose" in sig.parameters:
        kwargs["verbose"] = False
    if "return_metrics" in sig.parameters:
        kwargs["return_metrics"] = False

    return fitter.fit_cmle(data_xy, copula=copula, **kwargs)


def _tail_gap_from_summary(summ: dict) -> float:
    """
    Scalar tail mismatch:
        |lambdaL_emp_huang - lambdaL_model| + |lambdaU_emp_huang - lambdaU_model|
    """
    a = summ.get("lambdaL_emp_huang", np.nan)
    b = summ.get("lambdaL_model", np.nan)
    c = summ.get("lambdaU_emp_huang", np.nan)
    d = summ.get("lambdaU_model", np.nan)
    vals = [abs(a - b), abs(c - d)]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.sum(vals)) if vals else np.nan


# -----------------------
# Demo data (synthetic)
# -----------------------

def sample_uv(true_type: CopulaType, true_params, n=6000, seed=123):
    """
    Sample synthetic (U,V) from a chosen "true" copula.
    NOTE: this requires the copula class to implement sample().
    """
    cop = CopulaFactory.create(true_type)
    try:
        rng = np.random.default_rng(seed)
        uv = cop.sample(n, param=true_params, rng=rng)
    except TypeError:
        uv = cop.sample(n, param=true_params)
    uv = np.asarray(uv, float)
    return uv[:, 0], uv[:, 1]


# -----------------------
# Main ranking function
# -----------------------

def fit_and_rank_candidates(
    x, y,
    *,
    candidate_types,
    refine_topk=3,
    pit_m=400,
):
    fitter = CopulaFitter()

    rows = []
    copulas_stage1 = []

    # ---- Stage 1: fit_tau for all candidates + fast metrics
    for ct in candidate_types:
        cop = CopulaFactory.create(ct)

        # 1) fit_tau (very fast init)
        try:
            fitter.fit_tau((x, y), copula=cop)
        except Exception as e:
            rows.append({
                "CopulaType": ct, "CopulaName": getattr(cop, "name", str(ct)),
                "FitOK": False, "Error": f"fit_tau failed: {e}"
            })
            continue

        # 2) copula-only loglik (so AIC works)
        try:
            ll, n_obs = _compute_copula_only_loglik(cop, (x, y))
            cop.log_likelihood_ = ll
            cop.n_obs = n_obs
        except Exception as e:
            rows.append({
                "CopulaType": ct, "CopulaName": getattr(cop, "name", str(ct)),
                "FitOK": False, "Error": f"loglik failed: {e}"
            })
            continue

        # 3) summary: AIC + tau + tails + optional PIT
        try:
            summ = _gof_summary_with_optional_pit(cop, (x, y), include_pit=True, pit_m=pit_m)
        except Exception as e:
            summ = {"AIC": np.nan, "KT_err": np.nan, "Error": f"gof_summary failed: {e}"}

        # sanitize and add extras
        summ["CopulaType"] = ct
        summ["CopulaName"] = getattr(cop, "name", str(ct))
        summ["FitOK"] = True
        summ["Params"] = _flatten_params(cop.get_parameters())
        summ["TailGap"] = _tail_gap_from_summary(summ)

        # ensure PIT scalar exists if present
        if "PIT" not in summ and "PIT_ks_D" in summ:
            summ["PIT"] = summ["PIT_ks_D"]

        rows.append(summ)
        copulas_stage1.append(cop)

    df = pd.DataFrame(rows)

    df_ok = df[df["FitOK"] == True].copy()
    if len(df_ok) == 0:
        return None, df

    # ---- Stage 1 scoring (robust z-score combo)
    # ---- Stage 1 scoring (robust z-score combo) with NaN handling
    def _fill_missing(x, mode="neutral"):
        """
        mode="neutral": remplace NaN par la médiane (neutre)
        mode="penalize": remplace NaN par (max + marge) => pénalise le modèle
        """
        x = np.asarray(x, float)
        ok = np.isfinite(x)
        if not ok.any():
            return np.zeros_like(x)

        out = x.copy()
        if mode == "neutral":
            fill = np.nanmedian(x[ok])
        else:
            fill = np.nanmax(x[ok]) + 0.05  # marge
        out[~ok] = fill
        return out

    aic = df_ok["AIC"].astype(float).to_numpy()
    kt = df_ok["KT_err"].astype(float).to_numpy()
    tail = df_ok["TailGap"].astype(float).to_numpy()

    if "PIT" in df_ok.columns:
        pit = df_ok["PIT"].astype(float).to_numpy()
    else:
        pit = np.full(len(df_ok), np.nan, dtype=float)

    # Remplissage NaN:
    aic_f = _fill_missing(aic, mode="neutral")
    kt_f = _fill_missing(kt, mode="neutral")
    tail_f = _fill_missing(tail, mode="neutral")

    # PIT: je recommande "penalize" (si PIT manquant, on pénalise un peu)
    pit_f = _fill_missing(pit, mode="penalize")

    z_aic = _robust_z(aic_f)
    z_kt = _robust_z(kt_f)
    z_tail = _robust_z(tail_f)
    z_pit = _robust_z(pit_f)

    # weights
    w_aic, w_kt, w_tail, w_pit = 1.00, 0.75, 1.00, 1.00

    score1 = w_aic * z_aic + w_kt * z_kt + w_tail * z_tail + w_pit * z_pit
    df_ok["Score_stage1"] = score1

    # stable sort
    df_ok = df_ok.sort_values("Score_stage1", kind="mergesort").reset_index(drop=True)

    # ---- Stage 2: optional CMLE refine top-k (expensive, so only do a few)
    if refine_topk and refine_topk > 0:
        k = min(int(refine_topk), len(df_ok))

        # ensure object-typed columns exist for storing tuples/strings
        df_ok["AIC_refined"] = np.nan
        df_ok["KT_err_refined"] = np.nan
        df_ok["TailGap_refined"] = np.nan
        df_ok["PIT_refined"] = np.nan
        df_ok["Params_refined"] = None
        df_ok["RefineError"] = None

        for i in range(k):
            ct = df_ok.loc[i, "CopulaType"]
            cop = CopulaFactory.create(ct)

            try:
                # start again from tau init
                fitter.fit_tau((x, y), copula=cop)

                # quick CMLE refine
                _safe_fit_cmle(fitter, (x, y), cop)

                # recompute loglik + summary
                ll2, n_obs2 = _compute_copula_only_loglik(cop, (x, y))
                cop.log_likelihood_ = ll2
                cop.n_obs = n_obs2

                summ2 = _gof_summary_with_optional_pit(cop, (x, y), include_pit=True, pit_m=pit_m)
                summ2["TailGap"] = _tail_gap_from_summary(summ2)
                if "PIT" not in summ2 and "PIT_ks_D" in summ2:
                    summ2["PIT"] = summ2["PIT_ks_D"]

                df_ok.loc[i, "AIC_refined"] = float(summ2.get("AIC", np.nan))
                df_ok.loc[i, "KT_err_refined"] = float(summ2.get("KT_err", np.nan))
                df_ok.loc[i, "TailGap_refined"] = float(summ2.get("TailGap", np.nan))
                df_ok.loc[i, "PIT_refined"] = float(summ2.get("PIT", np.nan))
                df_ok.at[i, "Params_refined"] = _flatten_params(cop.get_parameters())

            except Exception as e:
                df_ok.at[i, "RefineError"] = str(e)

    # Return best candidate (stage1 best, already fit_tau params in df_ok)
    best_type = df_ok.loc[0, "CopulaType"]
    best = CopulaFactory.create(best_type)
    fitter.fit_tau((x, y), copula=best)

    return best, df_ok


# -----------------------
# Pretty final print
# -----------------------

def pretty_print_final(best, ranking: pd.DataFrame, candidates, refine_topk, pit_m):
    n = len(ranking)
    has_pit = ("PIT" in ranking.columns) or ("PIT_ks_D" in ranking.columns)

    # dataset stats
    try:
        tau_emp, _ = kendalltau(*pseudo_obs((X_U, Y_V)))  # uses globals from main()
    except Exception:
        tau_emp = np.nan

    print("\n" + "=" * 86)
    print("COPULAFURTIF — DEMO FIT + DIAGNOSTICS (tau init → optional CMLE refine)")
    print("=" * 86)
    print(f"Candidates (whitelist): {[getattr(CopulaFactory.create(ct), 'name', str(ct)) for ct in candidates]}")
    print(f"Ranking rows (fit OK): {n}")
    print(f"Refine top-k (CMLE quick): {refine_topk}")
    print(f"PIT/Rosenblatt enabled in gof_summary: {has_pit}  |  PIT subsample m={pit_m}")
    if np.isfinite(tau_emp):
        print(f"Empirical Kendall tau (on pseudo-obs): {tau_emp: .4f}")
    print("-" * 86)

    if best is None:
        print("No valid copula candidate fitted.")
        print("=" * 86)
        return

    print("BEST (stage1 winner):")
    print(f"  • Copula:  {getattr(best, 'name', type(best).__name__)}")
    print(f"  • Params:  {_flatten_params(best.get_parameters())}")
    print("-" * 86)

    # Table
    show_cols = []
    for c in ["CopulaName", "AIC", "KT_err", "TailGap", "PIT", "Score_stage1"]:
        if c in ranking.columns:
            show_cols.append(c)

    top = ranking[show_cols].head(min(10, len(ranking))).copy()

    # nice formatting (avoid trying to float-format tuples)
    def _fmt(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "nan"
        if isinstance(x, (float, np.floating)):
            return f"{x: .6f}"
        return str(x)

    print("TOP RANKING (stage1): (lower is better for all numeric columns)")
    for idx, row in top.iterrows():
        parts = []
        for c in show_cols:
            parts.append(f"{c}={_fmt(row[c])}")
        print(f"  {idx+1:>2}. " + " | ".join(parts))

    # refined section
    if "AIC_refined" in ranking.columns:
        print("-" * 86)
        print("REFINEMENT (CMLE quick) — ONLY for the first candidates (top-k):")
        cols2 = [c for c in ["CopulaName", "AIC_refined", "KT_err_refined", "TailGap_refined", "PIT_refined", "RefineError"] if c in ranking.columns]
        top2 = ranking[cols2].head(min(refine_topk, len(ranking))).copy()
        for idx, row in top2.iterrows():
            parts = []
            for c in cols2:
                parts.append(f"{c}={_fmt(row[c])}")
            print(f"  {idx+1:>2}. " + " | ".join(parts))

    print("-" * 86)
    print("METRICS EXPLAINED (quick cheat sheet):")
    print("  • AIC       : 2k - 2 logL_c  (copula-only, computed on pseudo-observations). Lower = better.")
    print("  • KT_err    : |tau_emp - tau_model| (fast dependence shape check). Lower = better.")
    print("  • TailGap   : |lambdaL_emp - lambdaL_model| + |lambdaU_emp - lambdaU_model|. Lower = better.")
    print("  • PIT       : KS D-stat on z2 from Rosenblatt transform (z2 should be Uniform). Lower = better.")
    print("  • Score_stage1 : robust z-score combination of (AIC, KT_err, TailGap, PIT) with weights.")
    print("NOTE:")
    print("  - PIT p-values are heuristic when parameters are estimated; use KS D-stat for ranking.")
    print("  - u,v are clipped away from 0/1 to avoid numerical explosions in some families (Galambos etc.).")
    print("=" * 86 + "\n")


# -----------------------
# Main
# -----------------------

if __name__ == "__main__":
    # ---- Choose a ground-truth scenario (synthetic demo)
    true_type = CopulaType.STUDENT
    true_params = [0.65, 7.0]  # e.g. rho=0.65, nu=7

    X_U, Y_V = sample_uv(true_type, true_params, n=6000, seed=123)

    # ---- Your predefined candidate whitelist (only include what you trust)
    candidates = [
        CopulaType.GAUSSIAN,
        CopulaType.STUDENT,
        CopulaType.CLAYTON,
        CopulaType.GUMBEL,
        CopulaType.FRANK,
        CopulaType.PLACKETT,
        CopulaType.JOE,
        CopulaType.GALAMBOS,  # can be numerically spicy -> clipping helps
        # add BB1/BB7 etc only if stable in your current implementation
    ]

    refine_topk = 3    # set 0 to disable CMLE refine
    pit_m = 400

    best, ranking = fit_and_rank_candidates(
        X_U, Y_V,
        candidate_types=candidates,
        refine_topk=refine_topk,
        pit_m=pit_m,
    )

    pretty_print_final(best, ranking, candidates, refine_topk, pit_m)
