# dash_bot/core/copula_engine.py
import numpy as np
from typing import Any, Tuple, Optional, List
import pandas as pd
from CopulaFurtif.copulas import CopulaFactory, CopulaType
from DataAnalysis.Utils.copulas import fit_copulas

HAS_COPULAFURTIF = True

def _nudge_params(p: np.ndarray) -> np.ndarray:
    """
    Évite les paramètres exactement sur des bornes ouvertes (ex: theta=0.01 alors que (0.01,100)).
    On pousse très légèrement vers l'intérieur.
    """
    p = np.array(p, dtype=float).copy()
    # petits epsilons relatifs
    def nudge_val(x: float) -> float:
        # bornes fréquentes vues dans tes logs
        if np.isclose(x, 0.01):
            return 0.0101
        if np.isclose(x, 0.0001):
            return 0.0001001
        if np.isclose(x, 1.01):
            return 1.0101
        # corr (gaussian / student)
        if abs(x) >= 1.0:
            return float(np.clip(x, -0.999, 0.999))
        return x

    return np.array([nudge_val(float(x)) for x in p], dtype=float)

def _get_all_copula_values():
    if HAS_COPULAFURTIF and CopulaType is not None:
        return [ct.value for ct in list(CopulaType)]
    return ["gaussian", "student", "clayton", "frank", "gumbel"]

def _copula_type_from_value(val: str):
    if not (HAS_COPULAFURTIF and CopulaType is not None):
        return None
    for ct in list(CopulaType):
        if ct.value.lower() == str(val).lower() or ct.name.lower() == str(val).lower():
            return ct
    return None

def build_copula(name: str, params: Any):
    """
    Reconstruit un objet copula à partir du nom + params via CopulaFurtif.
    """
    nm_raw = str(name)
    nm = nm_raw.strip()
    p = _nudge_params(np.atleast_1d(params).astype(float))

    def _set_params_furtif(cop, p):
        if hasattr(cop, "set_parameters"):
            cop.set_parameters(p)
            return
        if hasattr(cop, "parameters"):
            cop.parameters = p
            return
        if hasattr(cop, "_parameters"):
            cop._parameters = p
            return
        raise AttributeError("No parameter setter found for CopulaFurtif object.")

    def _to_copulatype(nm: str):
        n = nm.lower().replace("_", "-").strip()
        if n in ("t", "student", "studentt", "student-t", "student-t-copula"):
            return CopulaType.STUDENT
        if n in ("gaussian", "normal", "gauss"):
            return CopulaType.GAUSSIAN
        ct = _copula_type_from_value(nm)
        return ct

    if not HAS_COPULAFURTIF or CopulaFactory is None:
        raise RuntimeError("CopulaFurtif non disponible. Installe CopulaFurtif pour utiliser ce dashboard.")

    ct = _to_copulatype(nm)
    if ct is None:
        raise ValueError(f"CopulaType introuvable pour '{nm_raw}'")
    cop = CopulaFactory.create(ct)
    _set_params_furtif(cop, p)
    return cop

def _call_cdf(cop, u: float, v: float) -> float:
    """
    Évalue C(u,v) via CopulaFurtif.
    """
    u = float(np.clip(u, 1e-10, 1 - 1e-10))
    v = float(np.clip(v, 1e-10, 1 - 1e-10))

    def _as_float(out):
        return float(np.asarray(out).ravel()[0])

    # CopulaFurtif API : get_cdf(u, v)
    if hasattr(cop, "get_cdf"):
        try:
            return _as_float(cop.get_cdf(u, v))
        except Exception:
            pass

    # Variantes d'API CopulaFurtif
    for attr in ("get_CDF", "cdf", "CDF"):
        if hasattr(cop, attr):
            try:
                fn = getattr(cop, attr)
                return _as_float(fn(u, v))
            except Exception:
                pass

    # Dernier recours : objet callable
    if callable(cop):
        try:
            return _as_float(cop(u, v))
        except Exception:
            pass

    raise RuntimeError("Impossible d'évaluer la CDF du copula via CopulaFurtif.")

def copula_h_funcs(cop, u: float, v: float, eps: float = 1e-4) -> Tuple[float, float]:
    """
    h1|2 = P(U<=u | V=v), h2|1 = P(V<=v | U=u)
    - Si la copule CopulaFurtif fournit des conditionnelles built-in -> on les utilise
    - Sinon -> dérivées numériques via CDF
    """
    u = float(np.clip(u, eps, 1 - eps))
    v = float(np.clip(v, eps, 1 - eps))

    # 1) Built-in conditionnelles CopulaFurtif
    if hasattr(cop, "conditional_cdf_u_given_v") and hasattr(cop, "conditional_cdf_v_given_u"):
        try:
            h12 = float(cop.conditional_cdf_u_given_v(u, v))
            h21 = float(cop.conditional_cdf_v_given_u(u, v))
            return float(np.clip(h12, 0.0, 1.0)), float(np.clip(h21, 0.0, 1.0))
        except Exception:
            pass

    # 2) Dérivées numériques via la CDF
    dv = min(eps, v - 1e-8, 1 - 1e-8 - v)
    du = min(eps, u - 1e-8, 1 - 1e-8 - u)
    dv = max(dv, 1e-6)
    du = max(du, 1e-6)

    c_up = _call_cdf(cop, u, min(v + dv, 1 - 1e-8))
    c_dn = _call_cdf(cop, u, max(v - dv, 1e-8))
    h12 = (c_up - c_dn) / (2.0 * dv)

    c_up2 = _call_cdf(cop, min(u + du, 1 - 1e-8), v)
    c_dn2 = _call_cdf(cop, max(u - du, 1e-8), v)
    h21 = (c_up2 - c_dn2) / (2.0 * du)

    return float(np.clip(h12, 0.0, 1.0)), float(np.clip(h21, 0.0, 1.0))

def ecdf_value(sorted_x: np.ndarray, x: float) -> float:
    """
    Empirical CDF (CML style): F_hat(x) = rank/(n+1).
    """
    n = len(sorted_x)
    if n == 0:
        return np.nan
    # number <= x
    k = int(np.searchsorted(sorted_x, x, side="right"))
    return float(np.clip(k / (n + 1.0), 1e-6, 1 - 1e-6))

def is_copula_evaluable(name: str, params: Any) -> bool:
    """Teste si on peut calculer h-functions (donc trading) avec ce copula."""
    try:
        cop = build_copula(name, params)
        h12, h21 = copula_h_funcs(cop, 0.37, 0.61)
        return bool(np.isfinite(h12) and np.isfinite(h21))
    except Exception:
        return False

def fit_pair_copula(
    s1: pd.Series,
    s2: pd.Series,
    pick_mode: str,
    manual_name: str,
    suppress_logs: bool = True,
) -> Tuple[Optional[str], Optional[Any], pd.DataFrame, List[str]]:
    """
    Fit copulas sur pseudo-observations (CML) via Utils.fit_copulas.
    Retourne (best_name, best_params, df_fit, messages)
    """
    s1, s2 = s1.align(s2, join="inner")
    if len(s1) < 50:
        return None, None, pd.DataFrame(), ["Pas assez d'obs pour fitter le copula."]
    x = s1.to_numpy()
    y = s2.to_numpy()

    # Decide selection logic (fast/medium) based on the dashboard choice
    pick = str(pick_mode or "").lower().strip()
    if pick == "best_aic":
        fit_kwargs = dict(selection="aic", include_pit=False, refine_topk=0)
    elif pick == "best_score_pit":
        # Most expensive option: adds Rosenblatt-PIT and refines top candidates with a quick CMLE.
        fit_kwargs = dict(selection="score", include_pit=True, pit_m=250, refine_topk=2)
    else:
        # Default: robust score using AIC + Kendall-tau error + tail mismatch (no PIT).
        fit_kwargs = dict(selection="score", include_pit=False, refine_topk=0)

    # Fit (optionally silence logs to avoid CMLE/diagnostic spam)
    import io, contextlib
    if suppress_logs:
        _buf = io.StringIO()
        with contextlib.redirect_stdout(_buf), contextlib.redirect_stderr(_buf):
            df_fit, msgs = fit_copulas(x, y, **fit_kwargs)
        _fit_log = _buf.getvalue().strip()
        # keep captured logs only if nothing else is reported
        if (not msgs) and _fit_log:
            msgs = [_fit_log]
    else:
        df_fit, msgs = fit_copulas(x, y, **fit_kwargs)

    if df_fit is None or df_fit.empty:
        return None, None, pd.DataFrame(), (msgs or ["fit_copulas: aucun résultat."])

    # marquer les copulas évaluables (CDF/h-functions disponibles)
    if "evaluable" not in df_fit.columns:
        df_fit = df_fit.copy()
        df_fit["evaluable"] = [is_copula_evaluable(r["name"], r["params"]) for _, r in df_fit.iterrows()]

    if pick == "manual":
        target = str(manual_name).lower()
        row = df_fit[df_fit["name"].astype(str).str.lower() == target]
        if row.empty:
            msgs = (msgs or []) + [f"Copula '{manual_name}' introuvable dans le fit => fallback meilleur modèle évaluable."]
        else:
            cand = row.iloc[0]
            if bool(cand.get("evaluable", True)):
                return str(cand["name"]), cand["params"], df_fit, msgs
            msgs = (msgs or []) + [f"Copula '{manual_name}' non-évaluable (CDF/h-functions indisponibles) => fallback."]

    # pick best evaluable
    df_ok = df_fit[df_fit.get("evaluable", True) == True]  # noqa: E712
    if df_ok.empty:
        return None, None, df_fit, (msgs or []) + ["Aucune copula évaluable pour le trading (CDF/h-functions indisponibles). "
                                                  "Choisis une famille supportée (Gaussian/Student/Clayton/Frank/Gumbel) "
                                                  "ou étends _call_cdf pour CopulaFurtif."]
    best = df_ok.iloc[0]
    return str(best["name"]), best["params"], df_fit, msgs
