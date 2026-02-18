import numpy as np
from scipy.optimize import minimize
from typing import Sequence, Tuple

from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel
from CopulaFurtif.core.copulas.domain.estimation.utils import (
    auto_initialize_marginal_params,
    flatten_theta,
    adapt_theta,
    log_likelihood_only_copula,
    log_likelihood_joint,
)

# ==============================================================================
# Pseudo-observations (empirical CDF ranks)
# ==============================================================================
def pseudo_obs(data: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute pseudo-observations from raw data using empirical CDF ranks.

    Args:
        data: [X, Y] arrays of equal length

    Returns:
        (u, v) in (0,1)
    """
    if len(data) != 2:
        raise ValueError("Input must be a list or tuple with two elements [X, Y].")
    X, Y = data
    n = len(X)

    def empirical_cdf_ranks(values):
        ranks = np.argsort(np.argsort(values)) + 1
        return ranks / (n + 1)

    u = empirical_cdf_ranks(X)
    v = empirical_cdf_ranks(Y)
    return u, v

# ==============================================================================
# Huang (1992) tail-dependence estimator (upper/lower)
# ==============================================================================

def huang_lambda(u, v, side="upper", k=None):
    """
    Huang (1992) estimator of tail dependence.

    For sample (U_i, V_i), with k ~ sqrt(n):
      Upper tail:  (1/k) * sum 1{ U_i > U_(n-k),  V_i > V_(n-k) }
      Lower tail:  (1/k) * sum 1{ U_i <= U_(k),  V_i <= V_(k) }

    Here U_(j) is the j-th order statistic with j in {1,...,n}.
    """
    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    if u.shape != v.shape:
        raise ValueError("u and v must have the same shape.")
    n = u.size
    if n < 2:
        return np.nan

    if k is None:
        k = int(np.sqrt(n))
    k = int(k)
    k = max(1, min(k, n - 1))

    side = str(side).lower()
    if side in ("upper", "u", "up"):
        # threshold = U_(n-k) -> 0-based index = (n-k)-1
        idx = n - k - 1
        u_thr = np.partition(u, idx)[idx]
        v_thr = np.partition(v, idx)[idx]
        count = np.sum((u > u_thr) & (v > v_thr))
    elif side in ("lower", "l", "down"):
        # threshold = U_(k) -> 0-based index = k-1
        idx = k - 1
        u_thr = np.partition(u, idx)[idx]
        v_thr = np.partition(v, idx)[idx]
        count = np.sum((u <= u_thr) & (v <= v_thr))
    else:
        raise ValueError("side must be 'upper' or 'lower'.")

    return count / k


# ==============================================================================
# Helpers: bounds / init / U,V from marginals
# ==============================================================================
def _finite_bounds(bounds_like):
    """Replace None in (low, high) with large finite values for SciPy."""
    clean = []
    for low, high in bounds_like:
        lo = -1e10 if low  is None else low
        hi =  1e10 if high is None else high
        if hi <= lo:
            hi = lo + 1e-8
        clean.append((lo, hi))
    return clean

def _shrink_open_bounds(bounds, eps_abs: float = 1e-9, eps_rel: float = 1e-12):
    """
    Convert closed bounds [lo, hi] to open-like bounds (lo+δ, hi-δ),
    to avoid hitting strict validators that reject equality.
    """
    import numpy as np

    out = []
    for lo, hi in bounds:
        lo = float(lo)
        hi = float(hi)
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            out.append((lo, hi))
            continue
        width = hi - lo
        delta = max(float(eps_abs), float(eps_rel) * width)
        # Ensure we don't invert bounds
        if lo + delta >= hi - delta:
            delta = 0.49 * width
        out.append((lo + delta, hi - delta))
    return out


def _robust_init_from_uv(copula: CopulaModel, u: np.ndarray, v: np.ndarray, bounds=None) -> np.ndarray:
    """
    Initialize copula parameters from (u,v) via copula.init_from_data(u,v) if available.
    Falls back to current parameters. Clips to bounds if provided.
    """
    theta = np.array(copula.get_parameters(), dtype=float)

    try:
        if hasattr(copula, "init_from_data"):
            guess = copula.init_from_data(u, v)
            if guess is not None:
                theta = np.array(guess, dtype=float)
    except Exception as e:
        print("[INIT WARNING]", getattr(copula, "name", "copula"), "init_from_data failed:", e)

    if bounds is not None:
        clipped = []
        for i, (low, high) in enumerate(bounds):
            lo = -1e10 if low  is None else low
            hi =  1e10 if high is None else high
            val = float(theta[i])
            if val <= lo: val = lo + 1e-6
            if val >= hi: val = hi - 1e-6
            clipped.append(val)
        theta = np.array(clipped, dtype=float)

    return theta


def _uv_from_marginals(X: np.ndarray, Y: np.ndarray, marginals) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (U,V) from provided marginals dicts (already fitted or with guesses).

    IMPORTANT (math): shape parameters must be passed to scipy.stats in the
    exact order defined by dist.shapes, not by arbitrary dict iteration.

    Clips to (eps, 1-eps) for numerical stability.
    """
    m0, m1 = marginals
    dist0, dist1 = m0["distribution"], m1["distribution"]

    def _ordered_shape_params(m, dist):
        shapes = getattr(dist, "shapes", None)
        if shapes:
            names = [s.strip() for s in shapes.split(",")]
            return [m[name] for name in names]
        # fallback: deterministic but only safe if your keys already match the needed order
        keys = [k for k in m.keys() if k not in ("distribution", "loc", "scale")]
        keys = sorted(keys)
        return [m[k] for k in keys]

    s0_vals = _ordered_shape_params(m0, dist0)
    s1_vals = _ordered_shape_params(m1, dist1)

    loc0, scale0 = float(m0.get("loc", 0.0)), float(m0.get("scale", 1.0))
    loc1, scale1 = float(m1.get("loc", 0.0)), float(m1.get("scale", 1.0))

    U = dist0.cdf(X, *s0_vals, loc=loc0, scale=scale0)
    V = dist1.cdf(Y, *s1_vals, loc=loc1, scale=scale1)

    eps = 1e-12
    U = np.clip(U, eps, 1 - eps)
    V = np.clip(V, eps, 1 - eps)
    return U, V



# ==============================================================================
# CMLE (robust init, quick mode, optional Huang metrics)
# ==============================================================================
def _cmle(
    copula: CopulaModel,
    data,
    opti_method: str = 'SLSQP',
    options: dict = None,
    verbose: bool = True,
    use_init: bool = True,
    quick: bool = False,
    return_metrics: bool = False,
):
    """
    Canonical Maximum Likelihood Estimation (CMLE) using pseudo-observations,
    with robust data-driven initialization and an optional quick mode.

    NOTE: CMLE here uses copula.get_pdf(u, v) after copula.set_parameters(params),
          matching your CopulaModel API in this file.
    """
    if options is None:
        options = {}

    # 1) Pseudo-observations
    u, v = pseudo_obs(data)

    # 2) Bounds + "open-like" shrink (validators are strict on equality)
    try:
        base = np.array(copula.get_parameters(), dtype=float)
        bounds = copula.get_bounds() if hasattr(copula, "get_bounds") else [(None, None)] * len(base)
        clean_bounds = _finite_bounds(bounds)
        clean_bounds = _shrink_open_bounds(clean_bounds, eps_abs=1e-8, eps_rel=1e-12)

        lo = np.array([b[0] for b in clean_bounds], dtype=float)
        hi = np.array([b[1] for b in clean_bounds], dtype=float)

        if use_init:
            x0 = _robust_init_from_uv(copula, u, v, bounds)
        else:
            x0 = base.copy()

        x0 = np.asarray(x0, dtype=float).ravel()
        x0 = np.clip(x0, lo, hi)

    except Exception as e:
        print("[CMLE ERROR] Invalid initial parameters or bounds:", e)
        return None

    if quick and "maxiter" not in options:
        options = {**options, "maxiter": 50}

    BIG = 1e50

    # 3) Negative log-likelihood on (u,v)
    def neg_loglik(params_raw):
        try:
            params = np.asarray(params_raw, dtype=float).ravel()
            params = np.clip(params, lo, hi)

            if len(params) != len(copula.get_parameters()):
                return BIG

            # CMLE: set parameters then call get_pdf(u,v) without theta
            copula.set_parameters(params.tolist())
            pdf_vals = np.asarray(copula.get_pdf(u, v), dtype=float)

            if np.any(~np.isfinite(pdf_vals)) or np.any(pdf_vals <= 0):
                return BIG

            return float(-np.sum(np.log(np.maximum(pdf_vals, 1e-300))))
        except Exception as err:
            if verbose:
                print("[CMLE LOG_LIKELIHOOD ERROR]", err)
                print("→ Params received:", params_raw)
            return BIG

    # 4) Optimize
    try:
        result = minimize(neg_loglik, x0, method=opti_method, bounds=clean_bounds, options=options)
    except Exception as e:
        print("[CMLE ERROR] Optimizer crashed:", e)
        return None

    # 5) Output
    if result.success:
        fitted_params = np.clip(np.asarray(result.x, dtype=float).ravel(), lo, hi)
        loglik = float(-result.fun)

        # Safe write-back (avoid strict-boundary ValueError)
        try:
            copula.set_parameters(fitted_params.tolist())
        except ValueError:
            eps = 1e-12
            fitted_params = np.clip(fitted_params, lo + eps, hi - eps)
            copula.set_parameters(fitted_params.tolist())

        copula.set_log_likelihood(loglik)

        if not return_metrics:
            return fitted_params, loglik

        # Optional Huang tails on (u,v)
        try:
            lamU = float(huang_lambda(u, v, side="upper"))
            lamL = float(huang_lambda(u, v, side="lower"))
        except Exception:
            lamU = lamL = None

        extras = {"lambdaU_huang": lamU, "lambdaL_huang": lamL, "n_obs": len(u)}
        return fitted_params, loglik, extras

    else:
        if verbose:
            print(f"[CMLE FAILED] for copula '{getattr(copula, 'name', 'Unnamed Copula')}'")
            print("→ Initial guess:", x0)
            print("→ Bounds:", clean_bounds)
            print("→ Message:", result.message)
        return None


# ==============================================================================
# MLE (robust init on UV, quick mode, optional Huang metrics)
# ==============================================================================
def _fit_mle(
    data,
    copula: CopulaModel,
    marginals,
    opti_method: str = 'SLSQP',
    known_parameters: bool = False,
    options: dict = None,
    verbose: bool = True,
    use_init: bool = True,
    quick: bool = False,
    return_metrics: bool = False,
):
    """
    Fit a bivariate copula by MLE, optionally joint with marginals.
    Adds robust data-driven initialization for copula params, quick mode, and optional Huang tails.
    """
    if options is None:
        options = {}

    X, Y = data

    # Auto-initialize marginals if user provided only distributions
    for i, marg in enumerate(marginals):
        if "distribution" in marg and len(marg.keys()) == 1:
            marginals[i] = auto_initialize_marginal_params(data[i], marg["distribution"].name)

    # Validate marginals (soft vs strict depending on known_parameters)
    for idx, marg in enumerate(marginals):
        dist = marg["distribution"]
        dist_name = dist.name if hasattr(dist, "name") else f"marginal {idx}"

        shape_keys = [k for k in marg if k not in ("distribution", "loc", "scale")]
        shape_guesses = tuple(marg[k] for k in shape_keys)

        x_vals = data[idx]
        loc_guess = marg.get("loc", 0.0)
        scale_guess = marg.get("scale", 1.0)

        if known_parameters:
            if scale_guess is None or scale_guess <= 0:
                raise ValueError(f"Invalid scale parameter for '{dist_name}': {scale_guess}")
            pdf_vals = dist.pdf(x_vals, *shape_guesses, loc=loc_guess, scale=scale_guess)
            if np.any(np.isnan(pdf_vals)) or np.any(np.isinf(pdf_vals)) or np.any(pdf_vals < 0):
                raise ValueError(f"PDF of distribution '{dist_name}' is invalid over the given data.")
        else:
            if scale_guess is not None and scale_guess <= 0 and verbose:
                print(f"[WARNING] Initial scale for distribution '{dist_name}' is non-positive ({scale_guess}).")
            try:
                pdf_vals = dist.pdf(
                    x_vals,
                    *shape_guesses,
                    loc=loc_guess,
                    scale=(1.0 if scale_guess is None else scale_guess),
                )
                if np.any(np.isnan(pdf_vals)) or np.any(np.isinf(pdf_vals)):
                    if verbose:
                        print(f"[WARNING] Initial PDF values for distribution '{dist_name}' contain NaN or Inf.")
            except Exception as e:
                if verbose:
                    print(f"[WARNING] Failed to evaluate initial PDF for distribution '{dist_name}': {e}")

    # Prepare copula starting values (with robust init on U0,V0 if possible)
    theta0 = flatten_theta(copula.get_parameters())

    try:
        U0, V0 = _uv_from_marginals(X, Y, marginals)
    except Exception:
        U0 = V0 = None

    cop_bounds = copula.get_bounds() if hasattr(copula, "get_bounds") else [(None, None)] * len(theta0)

    if use_init and (U0 is not None) and (V0 is not None):
        theta0 = _robust_init_from_uv(copula, U0, V0, cop_bounds).tolist()

    # Dimension info for joint case
    margin_shapes_count = []
    for marg_dict in marginals:
        shape_keys = [k for k in marg_dict if k not in ("distribution", "loc", "scale")]
        margin_shapes_count.append(len(shape_keys))

    BIG = 1e50

    if known_parameters:
        # ---------------------------------------------------------------------
        # Optimize copula only (marginals fixed)
        # ---------------------------------------------------------------------
        x0 = np.asarray(theta0, dtype=float).ravel()
        clean_bounds = _finite_bounds(cop_bounds)
        clean_bounds = _shrink_open_bounds(clean_bounds, eps_abs=1e-8, eps_rel=1e-12)
        lo = np.array([b[0] for b in clean_bounds], dtype=float)
        hi = np.array([b[1] for b in clean_bounds], dtype=float)
        x0 = np.clip(x0, lo, hi)

        def objective(theta_array):
            try:
                theta_array = np.asarray(theta_array, dtype=float).ravel()
                theta_array = np.clip(theta_array, lo, hi)
                val = log_likelihood_only_copula(
                    theta_array=theta_array,
                    copula=copula,
                    X=X,
                    Y=Y,
                    marginals=marginals,
                    adapt_theta_func=adapt_theta,
                )
                return float(val) if np.isfinite(val) else BIG
            except Exception:
                return BIG

        if quick and "maxiter" not in options:
            options = {**options, "maxiter": 50}

        results = minimize(objective, x0, method=opti_method, bounds=clean_bounds, options=options)
        if verbose:
            print("Method:", opti_method, " | success:", results.success, " | message:", results.message)

        if not results.success:
            if verbose:
                print("Optimization failed")
            return None

        final_params = np.clip(np.asarray(results.x, dtype=float).ravel(), lo, hi)
        final_loglike = float(-results.fun)

        try:
            copula.set_parameters(final_params[:len(theta0)].tolist())
        except ValueError:
            eps = 1e-12
            copula.set_parameters(np.clip(final_params[:len(theta0)], lo + eps, hi - eps).tolist())

        copula.set_log_likelihood(final_loglike)

        if not return_metrics:
            return final_params, final_loglike

        # Huang tails
        try:
            U, V = _uv_from_marginals(X, Y, marginals)
            lamU = float(huang_lambda(U, V, side="upper"))
            lamL = float(huang_lambda(U, V, side="lower"))
        except Exception:
            lamU = lamL = None

        return final_params, final_loglike, {"lambdaU_huang": lamU, "lambdaL_huang": lamL, "n_obs": len(X)}

    else:
        # ---------------------------------------------------------------------
        # Joint optimization: copula + marginals
        # ---------------------------------------------------------------------
        # Build marginal init guesses [shapes..., loc, scale]
        margin_init_guesses = []
        for marg in marginals:
            shape_keys = [k for k in marg if k not in ("distribution", "loc", "scale")]
            shape0 = [float(marg[k]) for k in shape_keys]
            loc0 = float(marg.get("loc", 0.0))
            scale0 = float(marg.get("scale", 1.0))
            margin_init_guesses.append(shape0 + [loc0, scale0])

        # Assemble x0: copula theta then marginals
        x0 = list(theta0)
        for g in margin_init_guesses:
            x0.extend(g)
        x0 = np.asarray(x0, dtype=float).ravel()

        # Bounds: copula bounds then (shapes: None,None), loc(None,None), scale(>0)
        bounds = []
        bounds.extend(cop_bounds if cop_bounds is not None else [(None, None)] * len(theta0))
        for shape_count in margin_shapes_count:
            for _ in range(shape_count):
                bounds.append((None, None))
            bounds.append((None, None))   # loc
            bounds.append((1e-6, None))   # scale > 0

        clean_bounds = _finite_bounds(bounds)
        clean_bounds = _shrink_open_bounds(clean_bounds, eps_abs=1e-8, eps_rel=1e-12)
        lo = np.array([b[0] for b in clean_bounds], dtype=float)
        hi = np.array([b[1] for b in clean_bounds], dtype=float)

        if np.any(np.isnan(x0)) or np.any(np.isinf(x0)):
            raise ValueError("Invalid initial guess: contains NaNs or Infs.")

        x0 = np.clip(x0, lo, hi)

        if quick and "maxiter" not in options:
            options = {**options, "maxiter": 50}

        def objective(param_vec):
            try:
                param_vec = np.asarray(param_vec, dtype=float).ravel()
                param_vec = np.clip(param_vec, lo, hi)
                val = log_likelihood_joint(
                    params_vector=param_vec,
                    copula=copula,
                    X=X,
                    Y=Y,
                    marginals=marginals,
                    margin_shapes_count=margin_shapes_count,
                    adapt_theta_func=adapt_theta,
                    theta0_length=len(theta0),
                )
                return float(val) if np.isfinite(val) else BIG
            except Exception:
                return BIG

        results = minimize(objective, x0, method=opti_method, bounds=clean_bounds, options=options)
        if verbose:
            print("Method:", opti_method, " | success:", results.success, " | message:", results.message)

        if not results.success:
            raise RuntimeError("Optimization failed: " + str(results.message))

        final_params = np.clip(np.asarray(results.x, dtype=float).ravel(), lo, hi)
        final_loglike = float(-results.fun)

        # Store only the copula parameters into the object
        try:
            copula.set_parameters(final_params[:len(theta0)].tolist())
        except ValueError:
            eps = 1e-12
            copula.set_parameters(
                np.clip(final_params[:len(theta0)], lo[:len(theta0)] + eps, hi[:len(theta0)] - eps).tolist()
            )

        copula.set_log_likelihood(final_loglike)

        if not return_metrics:
            return final_params, final_loglike

        # Huang tails (simple report)
        try:
            U, V = _uv_from_marginals(X, Y, marginals)
            lamU = float(huang_lambda(U, V, side="upper"))
            lamL = float(huang_lambda(U, V, side="lower"))
        except Exception:
            lamU = lamL = None

        return final_params, final_loglike, {"lambdaU_huang": lamU, "lambdaL_huang": lamL, "n_obs": len(X)}


# ==============================================================================
# IFM (robust init on UV, quick mode, optional Huang metrics)
# ==============================================================================
def _fit_ifm(
    data,
    copula: CopulaModel,
    marginals,
    opti_method: str = 'SLSQP',
    options: dict = None,
    verbose: bool = True,
    use_init: bool = True,
    quick: bool = False,
    return_metrics: bool = False,
):
    """
    IFM with robust copula initialization on (U,V), optional quick mode, and optional Huang tails.

    NOTE: IFM here uses copula.get_pdf(U, V, theta) (your current IFM style), not set_parameters.
    """
    if options is None:
        options = {}

    X, Y = data
    if len(X) != len(Y):
        raise ValueError("X and Y must have the same length.")

    # 1) Fit marginals if needed, else trust provided params
    fitted_marginals = []
    for i, marg in enumerate(marginals):
        dist = marg["distribution"]
        xvals = data[i]

        shape_keys = [k for k in marg if k not in ("distribution", "loc", "scale")]
        loc_ = marg.get("loc", None)
        scale_ = marg.get("scale", None)

        # If only distribution given, auto-init
        if len(marg.keys()) == 1 and "distribution" in marg:
            marg = auto_initialize_marginal_params(xvals, dist.name)
            shape_keys = [k for k in marg if k not in ("distribution", "loc", "scale")]
            loc_ = marg.get("loc", None)
            scale_ = marg.get("scale", None)

        # If loc/scale missing, fit
        if loc_ is None or scale_ is None:
            shape_vals, loc_val, scale_val = dist.fit(xvals)
            shape_vals = shape_vals if isinstance(shape_vals, (list, tuple, np.ndarray)) else (shape_vals,)
            marg_final = {"distribution": dist}
            for j, sv in enumerate(shape_vals):
                marg_final[f"shape_{j}"] = float(sv)
            marg_final["loc"] = float(loc_val)
            marg_final["scale"] = float(scale_val)
            fitted_marginals.append(marg_final)
        else:
            marg_final = {"distribution": dist}
            for k in shape_keys:
                marg_final[k] = float(marg[k])
            marg_final["loc"] = 0.0 if loc_ is None else float(loc_)
            marg_final["scale"] = 1.0 if scale_ is None else float(scale_)
            fitted_marginals.append(marg_final)

    # 2) Build U,V
    U, V = _uv_from_marginals(X, Y, fitted_marginals)

    # 3) Bounds + shrink + x0 clip
    x0 = np.asarray(copula.get_parameters(), dtype=float).ravel()
    bounds = copula.get_bounds() if hasattr(copula, "get_bounds") else [(None, None)] * len(x0)
    clean_bounds = _finite_bounds(bounds)
    clean_bounds = _shrink_open_bounds(clean_bounds, eps_abs=1e-8, eps_rel=1e-12)

    lo = np.array([b[0] for b in clean_bounds], dtype=float)
    hi = np.array([b[1] for b in clean_bounds], dtype=float)

    if use_init:
        x0 = _robust_init_from_uv(copula, U, V, bounds)
    x0 = np.asarray(x0, dtype=float).ravel()
    x0 = np.clip(x0, lo, hi)

    if quick and "maxiter" not in options:
        options = {**options, "maxiter": 50}

    BIG = 1e50

    # 4) Copula NLL (IFM passes theta to get_pdf)
    def neg_log_likelihood(theta):
        try:
            theta = np.asarray(theta, dtype=float).ravel()
            theta = np.clip(theta, lo, hi)

            pdf_vals = np.asarray(copula.get_pdf(U, V, theta), dtype=float)
            if np.any(~np.isfinite(pdf_vals)) or np.any(pdf_vals <= 0):
                return BIG
            return float(-np.sum(np.log(np.maximum(pdf_vals, 1e-300))))
        except Exception:
            return BIG

    result = minimize(neg_log_likelihood, x0, method=opti_method, bounds=clean_bounds, options=options)

    if not result.success:
        if verbose:
            print("[IFM ERROR] Copula optimization failed")
            print("→ message:", result.message)
            print("→ initial guess:", x0)
            print("→ bounds:", clean_bounds)
        return None

    copula_params = np.clip(np.asarray(result.x, dtype=float).ravel(), lo, hi)
    loglik = float(-result.fun)

    try:
        copula.set_parameters(copula_params.tolist())
    except ValueError:
        eps = 1e-12
        copula_params = np.clip(copula_params, lo + eps, hi - eps)
        copula.set_parameters(copula_params.tolist())

    copula.set_log_likelihood(loglik)

    if not return_metrics:
        return copula_params, loglik

    # Optional Huang tails on (U,V)
    try:
        lamU = float(huang_lambda(U, V, side="upper"))
        lamL = float(huang_lambda(U, V, side="lower"))
    except Exception:
        lamU = lamL = None

    return copula_params, loglik, {"lambdaU_huang": lamU, "lambdaL_huang": lamL, "n_obs": len(U)}


def quick_fit(
    data,
    copula,
    mode: str = "cmle",
    marginals=None,
    maxiter: int = 60,
    optimizer: str = "L-BFGS-B",
    return_metrics: bool = True,
):
    """
    Ultra-rapide: init robuste + optimisation tronquée pour une calibration coarse.
    - mode="cmle": pseudo-obs ranks
    - mode="ifm": nécessite marginals (CDF)
    Met à jour le copula (params + loglik).
    """
    if mode not in ("cmle", "ifm"):
        raise ValueError("mode must be 'cmle' or 'ifm'")

    BIG = 1e50

    if mode == "cmle":
        u, v = pseudo_obs(data)

        bounds = copula.get_bounds() if hasattr(copula, "get_bounds") else [(None, None)] * len(copula.get_parameters())
        clean_bounds = _finite_bounds(bounds)
        clean_bounds = _shrink_open_bounds(clean_bounds, eps_abs=1e-8, eps_rel=1e-12)
        lo = np.array([b[0] for b in clean_bounds], dtype=float)
        hi = np.array([b[1] for b in clean_bounds], dtype=float)

        x0 = _robust_init_from_uv(copula, u, v, bounds)
        x0 = np.asarray(x0, dtype=float).ravel()
        x0 = np.clip(x0, lo, hi)

        def nll(theta):
            theta = np.asarray(theta, dtype=float).ravel()
            theta = np.clip(theta, lo, hi)
            try:
                copula.set_parameters(theta.tolist())
            except ValueError:
                return BIG

            pdf = np.asarray(copula.get_pdf(u, v), dtype=float)
            if np.any(~np.isfinite(pdf)) or np.any(pdf <= 0):
                return BIG
            return float(-np.sum(np.log(np.maximum(pdf, 1e-300))))

        res = minimize(nll, x0, method=optimizer, bounds=clean_bounds, options={"maxiter": maxiter})
        Uret, Vret = u, v

    else:
        if marginals is None:
            raise ValueError("marginals must be provided for mode='ifm'")

        X, Y = data
        m0, m1 = marginals
        dist0, dist1 = m0["distribution"], m1["distribution"]
        shape0 = [m0[k] for k in m0 if k not in ("distribution", "loc", "scale")]
        shape1 = [m1[k] for k in m1 if k not in ("distribution", "loc", "scale")]
        U = dist0.cdf(X, *shape0, loc=m0.get("loc", 0.0), scale=m0.get("scale", 1.0))
        V = dist1.cdf(Y, *shape1, loc=m1.get("loc", 0.0), scale=m1.get("scale", 1.0))
        eps = 1e-12
        U = np.clip(U, eps, 1 - eps)
        V = np.clip(V, eps, 1 - eps)

        bounds = copula.get_bounds() if hasattr(copula, "get_bounds") else [(None, None)] * len(copula.get_parameters())
        clean_bounds = _finite_bounds(bounds)
        clean_bounds = _shrink_open_bounds(clean_bounds, eps_abs=1e-8, eps_rel=1e-12)
        lo = np.array([b[0] for b in clean_bounds], dtype=float)
        hi = np.array([b[1] for b in clean_bounds], dtype=float)

        x0 = _robust_init_from_uv(copula, U, V, bounds)
        x0 = np.asarray(x0, dtype=float).ravel()
        x0 = np.clip(x0, lo, hi)

        def nll(theta):
            theta = np.asarray(theta, dtype=float).ravel()
            theta = np.clip(theta, lo, hi)
            try:
                copula.set_parameters(theta.tolist())
            except ValueError:
                return BIG

            pdf = np.asarray(copula.get_pdf(U, V), dtype=float)
            if np.any(~np.isfinite(pdf)) or np.any(pdf <= 0):
                return BIG
            return float(-np.sum(np.log(np.maximum(pdf, 1e-300))))

        res = minimize(nll, x0, method=optimizer, bounds=clean_bounds, options={"maxiter": maxiter})
        Uret, Vret = U, V

    if not res.success and hasattr(res, "message"):
        print("[QUICK FIT WARNING] Optimization not fully converged:", res.message)

    theta = res.x if res.success else x0
    ll = float(-res.fun) if res.success else float("nan")

    cur_len = len(copula.get_parameters())

    theta = np.asarray(theta, dtype=float).ravel()[:cur_len]
    try:
        copula.set_parameters(theta.tolist())
    except ValueError:
        eps = 1e-12
        theta = np.clip(theta, lo[:cur_len] + eps, hi[:cur_len] - eps)
        copula.set_parameters(theta.tolist())

    if hasattr(copula, "set_log_likelihood"):
        copula.set_log_likelihood(float(ll))
    else:
        copula.log_likelihood_ = float(ll)

    if not return_metrics:
        return theta, ll

    try:
        lamU = huang_lambda(Uret, Vret, side="upper")
        lamL = huang_lambda(Uret, Vret, side="lower")
    except Exception:
        lamU = lamL = None

    return {"theta": theta, "loglik": ll, "lambdaU_huang": lamU, "lambdaL_huang": lamL}


def _fit_tau_core(data, copula):
    """
    Minimal 'init-only' fit:
      - build pseudo-observations (u, v) from raw (X, Y) via ranks,
      - delegate to copula.init_from_data(u, v),
      - set copula parameters,
      - return the initial theta.
    No CMLE/MLE/IFM. No optimizer. Just your init_from_data.
    """
    if not isinstance(data, (list, tuple)) or len(data) != 2:
        raise ValueError("fit_tau: 'data' must be a (X, Y) tuple/list.")

    X, Y = data
    X = np.asarray(X)
    Y = np.asarray(Y)

    if X.shape[0] != Y.shape[0]:
        raise ValueError("fit_tau: X and Y must have the same length.")

    # 1) pseudo-observations (rank transform) -> (u, v)
    u, v = pseudo_obs([X, Y])  # this helper already exists in this module

    # 2) mandatory: the copula must expose init_from_data(u, v)
    if not hasattr(copula, "init_from_data"):
        raise AttributeError("fit_tau requires copula.init_from_data(u, v). Please implement it on the copula class.")

    # 3) delegate to the copula's moments-based initializer
    theta = copula.init_from_data(u, v)
    theta = np.atleast_1d(np.array(theta, dtype=float))

    # 4) update the copula instance
    if hasattr(copula, "set_parameters"):
        copula.set_parameters(theta)
    else:
        copula.parameters = theta

    return theta

