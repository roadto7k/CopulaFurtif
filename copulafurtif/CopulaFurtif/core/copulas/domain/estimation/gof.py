import numpy as np
from scipy.stats import kendalltau
from scipy.stats import kstest

from CopulaFurtif.core.copulas.domain.estimation.estimation import pseudo_obs, huang_lambda


def compute_aic(copula):
    """
    AIC = 2k - 2 logL

    IMPORTANT (math):
    - If copula.log_likelihood_ is COPULA-ONLY: k = #copula params
    - If copula.log_likelihood_ is JOINT (copula+marginals): k must include marginal params too.
      In that case, set copula.n_params_total accordingly.
    """
    if hasattr(copula, "n_params_total") and copula.n_params_total is not None:
        k = int(copula.n_params_total)
    else:
        k = len(copula.get_bounds())
    return 2 * k - 2 * copula.log_likelihood_


def compute_bic(copula):
    """
    BIC = k log(n) - 2 logL

    IMPORTANT (math): same comment as compute_aic regarding k.
    """
    if not hasattr(copula, "n_obs") or copula.n_obs is None:
        raise ValueError("n_obs (number of observations) is missing in copula object.")

    if hasattr(copula, "n_params_total") and copula.n_params_total is not None:
        k = int(copula.n_params_total)
    else:
        k = len(copula.get_bounds())

    return k * np.log(copula.n_obs) - 2 * copula.log_likelihood_


def compute_iad_score(copula, data, *, max_n=250, seed=0, q_tail=0.10, tail_frac=0.33):
    eps = 1e-10  # <-- moved up

    u, v = data
    u, v = _prepare_uv(u, v, max_n=max_n, seed=seed, q_tail=q_tail, tail_frac=tail_frac)

    n = len(u)
    params = copula.get_parameters()

    u_sorted = np.sort(u)
    v_sorted = np.sort(v)

    below_u = u[:, None] <= u_sorted[None, :]
    below_v = v[:, None] <= v_sorted[None, :]
    C_empirical = (below_u.T @ below_v) / n
    C_empirical = np.clip(C_empirical, eps, 1 - eps)

    uu, vv = np.meshgrid(u_sorted, v_sorted, indexing="ij")
    C_model = copula.get_cdf(uu.ravel(), vv.ravel(), params).reshape(n, n)

    C_model = np.clip(C_model, eps, 1 - eps)
    denom = np.clip(C_model * (1 - C_model), eps, None)

    iad_score = np.mean(((C_empirical - C_model) ** 2) / denom)
    return float(iad_score)


def AD_score(copula, data, *, max_n=300, seed=0, q_tail=0.10, tail_frac=0.33):
    params = copula.get_parameters()

    u, v = data
    u, v = _prepare_uv(u, v, max_n=max_n, seed=seed, q_tail=q_tail, tail_frac=tail_frac)

    n = len(u)

    u_sorted = np.sort(u)
    v_sorted = np.sort(v)

    below_u = u[:, None] <= u_sorted[None, :]
    below_v = v[:, None] <= v_sorted[None, :]
    C_empirical = (below_u.T @ below_v) / n

    uu, vv = np.meshgrid(u_sorted, v_sorted, indexing="ij")
    C_model = copula.get_cdf(uu.ravel(), vv.ravel(), params).reshape(n, n)

    eps = 1e-10
    C_model = np.clip(C_model, eps, 1 - eps)
    weights = np.clip(C_model * (1 - C_model), eps, None)

    # ton AD actuel = max weighted deviation (plus proche d’un “sup” score)
    ad_score = np.max(((C_empirical - C_model) ** 2) / weights)
    return float(ad_score)

def subsample_uv_tail_aware(u, v, m=300, seed=0, q_tail=0.10, tail_frac=0.33):
    """
    Subsample (u,v) to m points, over-representing tails.
    tail_frac ~ fraction of points forced in tails.
    """
    u = np.asarray(u).ravel()
    v = np.asarray(v).ravel()
    n = len(u)
    if n <= m:
        return u, v

    rng = np.random.default_rng(seed)

    lo_u = u <= q_tail
    hi_u = u >= 1 - q_tail
    lo_v = v <= q_tail
    hi_v = v >= 1 - q_tail

    tail_mask = (lo_u & lo_v) | (hi_u & hi_v) | (lo_u & hi_v) | (hi_u & lo_v)
    tail_idx = np.flatnonzero(tail_mask)
    mid_idx  = np.flatnonzero(~tail_mask)

    m_tail = min(len(tail_idx), int(round(m * tail_frac)))
    m_mid  = m - m_tail

    pick = []
    if m_tail > 0:
        pick.append(rng.choice(tail_idx, size=m_tail, replace=False))
    if m_mid > 0:
        pick.append(rng.choice(mid_idx, size=m_mid, replace=False))

    if not pick:
        idx = rng.choice(n, size=m, replace=False)
    else:
        idx = np.concatenate(pick)

    rng.shuffle(idx)
    return u[idx], v[idx]


def _prepare_uv(u, v, *, max_n=None, seed=0, q_tail=0.10, tail_frac=0.33, eps=1e-10):
    u = np.asarray(u).ravel()
    v = np.asarray(v).ravel()
    if len(u) != len(v):
        raise ValueError("Mismatch: len(u) != len(v)")

    # keep strictly inside (0,1) for weights
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)

    if (max_n is not None) and (len(u) > int(max_n)):
        u, v = subsample_uv_tail_aware(u, v, m=int(max_n), seed=seed, q_tail=q_tail, tail_frac=tail_frac)
    return u, v


def AD_score_subsampled(copula, x, y, m=300, seed=0):
    u, v = pseudo_obs([x, y])
    return AD_score(copula, (u, v), max_n=m, seed=seed)

def IAD_score_subsampled(copula, x, y, m=250, seed=0):
    u, v = pseudo_obs([x, y])
    return compute_iad_score(copula, (u, v), max_n=m, seed=seed)

def AD_score_bootstrap(copula, x, y, m=300, n_boot=3, seed=0):
    u, v = pseudo_obs([x, y])
    vals = []
    for b in range(n_boot):
        vals.append(float(AD_score(copula, (u, v), max_n=m, seed=seed + b)))
    return float(np.mean(vals)), float(np.std(vals))

def kendall_tau_distance(copula, data):
    """
    Compute the absolute distance between empirical and theoretical Kendall’s tau for a copula.

    Args:
        copula (object): Copula model with optional method `kendall_tau(params)`.
        data (Sequence[array-like, array-like]): Two-element sequence [X, Y] of observed samples.

    Returns:
        float: Absolute difference between empirical and theoretical Kendall’s tau, or NaN if not implemented.
    """

    if not hasattr(copula, "kendall_tau"):
        print(f"[WARNING] Kendall's tau formula not implemented for copula '{copula.name}'. Returning np.nan.")
        return np.nan

    # Empirical tau from data
    X, Y = data
    tau_empirical, _ = kendalltau(X, Y)

    try:
        tau_theoretical = copula.kendall_tau(copula.get_parameters())
    except Exception as e:
        print(f"[WARNING] Failed to compute theoretical Kendall's tau: {e}")
        return np.nan

    return abs(tau_empirical - tau_theoretical)

def tail_metrics_huang(copula, data):
    """
    Returns empirical Huang tails (L,U) and model TD (LTDC, UTDC).
    """
    u, v = pseudo_obs(data)
    lamL_emp = huang_lambda(u, v, side="lower")
    lamU_emp = huang_lambda(u, v, side="upper")
    params = copula.get_parameters()
    return {
        "lambdaL_emp_huang": float(lamL_emp),
        "lambdaU_emp_huang": float(lamU_emp),
        "lambdaL_model": float(copula.LTDC(params)),
        "lambdaU_model": float(copula.UTDC(params)),
    }

def _call_get_cdf(copula, u, v, param=None):
    """Compat: certains modèles acceptent get_cdf(u,v,param), d'autres non."""
    try:
        return copula.get_cdf(u, v, param)
    except TypeError:
        return copula.get_cdf(u, v)


def _conditional_cdf_v_given_u_safe(copula, u, v, param=None, *, h=1e-5, eps=1e-12):
    """
    Essaie d'abord copula.conditional_cdf_v_given_u (idéal, analytique via ∂C/∂u).
    Sinon fallback: dérivée finie centrale de C(u,v) wrt u.
    """
    # 1) voie "propre" (si partials implémentées)
    try:
        return copula.conditional_cdf_v_given_u(u, v, param)
    except Exception as e:
        print(f"Error in function _conditional_cdf_v_given_u_safe, gof.py : {e}")
        pass

    # 2) fallback numérique (moins rapide, mais robuste)
    u = np.asarray(u, float)
    v = np.asarray(v, float)

    u0 = np.clip(u - h, eps, 1.0 - eps)
    u1 = np.clip(u + h, eps, 1.0 - eps)

    C1 = _call_get_cdf(copula, u1, v, param)
    C0 = _call_get_cdf(copula, u0, v, param)

    dC_du = (C1 - C0) / (u1 - u0)
    return np.clip(dC_du, eps, 1.0 - eps)


def rosenblatt_transform_2d(copula, data, *, max_n=400, seed=0, q_tail=0.10, tail_frac=0.33,
                            h=1e-5, eps=1e-12):
    """
    Rosenblatt transform (2D):
      z1 = U
      z2 = F_{V|U=u}(V) = ∂C(u,v)/∂u

    data: (X, Y) ou (u, v) mais on applique pseudo_obs par cohérence du projet.
    """
    u, v = pseudo_obs(data)
    u, v = _prepare_uv(u, v, max_n=max_n, seed=seed, q_tail=q_tail, tail_frac=tail_frac, eps=eps)

    theta = copula.get_parameters()
    z1 = u
    z2 = _conditional_cdf_v_given_u_safe(copula, u, v, theta, h=h, eps=eps)

    z1 = np.clip(z1, eps, 1.0 - eps)
    z2 = np.clip(z2, eps, 1.0 - eps)
    return z1, z2


def pit_ks_uniform(z):
    """
    KS test vs Uniform(0,1). Retourne (D, pvalue).
    Note: pvalue pas “exact” car paramètres estimés -> utiliser D surtout pour ranking.
    """
    z = np.asarray(z, float).ravel()
    z = z[np.isfinite(z)]
    if z.size < 10:
        return np.nan, np.nan
    D, p = kstest(z, "uniform")
    return float(D), float(p)


def rosenblatt_pit_metrics(copula, data, *, max_n=400, seed=0, q_tail=0.10, tail_frac=0.33,
                           h=1e-5, eps=1e-12, add_tail_slices=True):
    """
    Renvoie un dict de métriques PIT:
      - PIT_ks_D, PIT_ks_pvalue sur z2
      - optionnel: PIT_loU_*, PIT_hiU_* sur sous-ensembles de u (queue basse/haute)
      - PIT_indep_tau: Kendall tau(z1,z2) ~ 0 si modèle bon
    """
    z1, z2 = rosenblatt_transform_2d(
        copula, data, max_n=max_n, seed=seed, q_tail=q_tail, tail_frac=tail_frac, h=h, eps=eps
    )

    D, p = pit_ks_uniform(z2)

    out = {
        "PIT_ks_D": D,
        "PIT_ks_pvalue": p,
    }

    # Indépendance attendue sous modèle correct (heuristique utile)
    try:
        tau, _ = kendalltau(z1, z2)
        out["PIT_indep_tau"] = float(tau)
    except Exception:
        out["PIT_indep_tau"] = np.nan

    if add_tail_slices:
        lo = z1 <= q_tail
        hi = z1 >= (1.0 - q_tail)

        Dlo, plo = pit_ks_uniform(z2[lo]) if np.any(lo) else (np.nan, np.nan)
        Dhi, phi = pit_ks_uniform(z2[hi]) if np.any(hi) else (np.nan, np.nan)

        out.update({
            "PIT_loU_ks_D": Dlo,
            "PIT_loU_ks_pvalue": plo,
            "PIT_hiU_ks_D": Dhi,
            "PIT_hiU_ks_pvalue": phi,
        })

    # alias pratique pour sélection (plus petit = mieux)
    out["PIT"] = out["PIT_ks_D"]
    return out

