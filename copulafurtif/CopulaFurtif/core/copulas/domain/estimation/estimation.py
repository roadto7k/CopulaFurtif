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
    Huang (1992) estimator of tail dependence. k ~ sqrt(n) by default.
    """
    u = np.asarray(u); v = np.asarray(v)
    n = len(u)
    if k is None:
        k = int(np.sqrt(n)) if n > 0 else 1
        k = max(1, min(k, n-1))

    if side == "upper":
        u_thr = np.partition(u, n-k)[-k]
        v_thr = np.partition(v, n-k)[-k]
        count = np.sum((u > u_thr) & (v > v_thr))
    else:
        u_thr = np.partition(u, k)[k]
        v_thr = np.partition(v, k)[k]
        count = np.sum((u < u_thr) & (v < v_thr))

    return count / max(1, k)

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
    Clips to (eps, 1-eps) for numerical stability.
    """
    m0, m1 = marginals
    dist0, dist1 = m0["distribution"], m1["distribution"]

    s0_keys = [k for k in m0 if k not in ("distribution", "loc", "scale")]
    s1_keys = [k for k in m1 if k not in ("distribution", "loc", "scale")]
    s0_vals = [m0[k] for k in s0_keys]
    s1_vals = [m1[k] for k in s1_keys]

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
    try:
        u, v = pseudo_obs(data)
        copula.set_n_obs(len(u))
    except Exception as e:
        print("[CMLE ERROR] Failed to compute pseudo-observations:", e)
        return None

    # 2) Bounds & initialization
    try:
        base = np.array(copula.get_parameters(), dtype=float)
        bounds = copula.get_bounds() if hasattr(copula, "get_bounds") else [(None, None)] * len(base)
        clean_bounds = _finite_bounds(bounds)

        if use_init:
            x0 = _robust_init_from_uv(copula, u, v, bounds)
        else:
            x0 = base.copy()
            for i, (lo, hi) in enumerate(clean_bounds):
                if x0[i] <= lo: x0[i] = lo + 1e-6
                if x0[i] >= hi: x0[i] = hi - 1e-6

    except Exception as e:
        print("[CMLE ERROR] Invalid initial parameters or bounds:", e)
        return None

    if quick and "maxiter" not in options:
        options = {**options, "maxiter": 50}

    # 3) Negative log-likelihood on (u,v)
    def neg_loglik(params_raw):
        try:
            params = np.atleast_1d(np.array(params_raw, dtype=float))
            if len(params) != len(copula.get_parameters()):
                return np.inf
            copula.set_parameters(params.tolist())
            pdf_vals = copula.get_pdf(u, v)  # IMPORTANT: no theta passed here (matches your CMLE)
            if np.any(pdf_vals <= 0) or np.any(np.isnan(pdf_vals)):
                return np.inf
            return -np.sum(np.log(pdf_vals))
        except Exception as err:
            if verbose:
                print("[CMLE LOG_LIKELIHOOD ERROR]", err)
                print("→ Params received:", params_raw)
            return np.inf

    # 4) Optimize
    try:
        result = minimize(neg_loglik, x0, method=opti_method, bounds=clean_bounds, options=options)
    except Exception as e:
        print("[CMLE ERROR] Optimizer crashed:", e)
        return None

    # 5) Output
    if result.success:
        fitted_params = result.x
        loglik = -result.fun
        copula.set_parameters(list(fitted_params))
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
    if copula.type == "mixture":
        raise ValueError("MLE estimation for mixture copulas is not supported. Use CMLE instead.")

    if options is None:
        options = {}

    X, Y = data
    copula.set_n_obs(len(X))

    # Auto-initialize marginals if parameters are missing
    if not known_parameters:
        for i in range(len(marginals)):
            marg = marginals[i]
            if isinstance(marg["distribution"], str):
                marginals[i] = auto_initialize_marginal_params(data[i], marg["distribution"])
            elif len(marg.keys()) == 1 and "distribution" in marg:
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
            if scale_guess <= 0:
                raise ValueError(f"Distribution '{dist_name}' must have scale > 0. Got: {scale_guess}")
            try:
                pdf_vals = dist.pdf(x_vals, *shape_guesses, loc=loc_guess, scale=scale_guess)
                if np.any(np.isnan(pdf_vals)) or np.any(np.isinf(pdf_vals)) or np.any(pdf_vals < 0):
                    raise ValueError(f"PDF of distribution '{dist_name}' is invalid over the given data.")
            except Exception as e:
                raise ValueError(f"Error while evaluating PDF for distribution '{dist_name}': {e}")
        else:
            if scale_guess <= 0:
                print(f"[WARNING] Initial scale for distribution '{dist_name}' is non-positive ({scale_guess}).")
            try:
                pdf_vals = dist.pdf(x_vals, *shape_guesses, loc=loc_guess, scale=scale_guess)
                if np.any(np.isnan(pdf_vals)) or np.any(np.isinf(pdf_vals)):
                    print(f"[WARNING] Initial PDF values for distribution '{dist_name}' contain NaN or Inf.")
            except Exception as e:
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

    if known_parameters:
        # Optimize copula only (marginals fixed) using your existing objective
        def objective(theta_array):
            return log_likelihood_only_copula(
                theta_array=theta_array,
                copula=copula,
                X=X,
                Y=Y,
                marginals=marginals,
                adapt_theta_func=adapt_theta,
            )

        x0 = np.array(theta0, dtype=float)
        clean_bounds = _finite_bounds(cop_bounds)

        if quick and "maxiter" not in options:
            options = {**options, "maxiter": 50}

        results = minimize(objective, x0, method=opti_method, bounds=clean_bounds, options=options)
        print("Method:", opti_method, " | success:", results.success, " | message:", results.message)

        if not results.success:
            print("Optimization failed")
            return None

        final_params = results.x
        final_loglike = -results.fun
        copula.set_parameters(final_params[:len(theta0)])
        copula.set_log_likelihood(final_loglike)

        if not return_metrics:
            return final_params, final_loglike

        # Huang tails on (U,V) for reporting
        try:
            U, V = _uv_from_marginals(X, Y, marginals)
            lamU = float(huang_lambda(U, V, side="upper"))
            lamL = float(huang_lambda(U, V, side="lower"))
        except Exception:
            lamU = lamL = None

        return final_params, final_loglike, {"lambdaU_huang": lamU, "lambdaL_huang": lamL, "n_obs": len(X)}

    else:
        # Joint optimization of copula + marginals
        # Build marginal init guesses [shapes..., loc, scale]
        margin_init_guesses = []
        for marg in marginals:
            shape_keys = [k for k in marg if k not in ("distribution", "loc", "scale")]
            shape0 = [float(marg[k]) for k in shape_keys]
            loc0 = float(marg.get("loc", 0.0))
            scale0 = float(marg.get("scale", 1.0))
            margin_init_guesses.append(shape0 + [loc0, scale0])

        # Assemble x0: start with copula theta (robust-seeded if possible), then marginals
        x0 = theta0[:]
        for g in margin_init_guesses:
            x0.extend(g)
        x0 = np.array(x0, dtype=float)

        # Bounds: copula bounds then (shapes: None,None), loc(None,None), scale(>0)
        bounds = []
        bounds.extend(cop_bounds if cop_bounds is not None else [(None, None)] * len(theta0))
        for shape_count in margin_shapes_count:
            for _ in range(shape_count): bounds.append((None, None))
            bounds.append((None, None))   # loc
            bounds.append((1e-6, None))   # scale > 0
        clean_bounds = _finite_bounds(bounds)

        if np.any(np.isnan(x0)) or np.any(np.isinf(x0)):
            raise ValueError("Invalid initial guess: contains NaNs or Infs.")

        if quick and "maxiter" not in options:
            options = {**options, "maxiter": 80}

        def objective(param_vec):
            return log_likelihood_joint(
                param_vec=param_vec,
                copula=copula,
                X=X,
                Y=Y,
                marginals=marginals,
                margin_shapes_count=margin_shapes_count,
                adapt_theta_func=adapt_theta,
                theta0_length=len(theta0),
            )

        try:
            results = minimize(objective, x0, method=opti_method, bounds=clean_bounds, options=options)
        except Exception as e:
            raise RuntimeError(f"Optimization crashed due to exception: {e}")

        print("Method:", opti_method, " | success:", results.success, " | message:", results.message)

        if not results.success:
            raise RuntimeError("Optimization failed: " + str(results.message))

        final_params = results.x
        final_loglike = -results.fun

        # Store only the copula parameters into the object
        copula.set_parameters(final_params[:len(theta0)])
        copula.set_log_likelihood(final_loglike)

        if not return_metrics:
            return final_params, final_loglike

        # Report Huang tails using current marginals (simple; for exact post-fit marginals, parse final_params)
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

        needs_fit = False
        if hasattr(dist, "shapes") and dist.shapes is not None:
            shape_names = [s.strip() for s in dist.shapes.split(',')]
            if len(shape_names) != len(shape_keys):
                needs_fit = True
        if loc_ is None or scale_ is None:
            needs_fit = True

        if needs_fit:
            fit_res = dist.fit(xvals)
            if hasattr(dist, "shapes") and dist.shapes is not None:
                shape_count = len(dist.shapes.split(','))
                shape_vals = fit_res[:shape_count]
                loc_val = fit_res[shape_count]
                scale_val = fit_res[shape_count + 1]
            else:
                shape_vals = []
                loc_val = fit_res[0]
                scale_val = fit_res[1]

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

    # 2) Build U,V and robust-seed copula params
    U, V = _uv_from_marginals(X, Y, fitted_marginals)

    def neg_log_likelihood(theta):
        pdf_vals = copula.get_pdf(U, V, theta)  # IMPORTANT: IFM passes theta to get_pdf
        if np.any(pdf_vals <= 0) or np.any(np.isnan(pdf_vals)):
            return np.inf
        return -np.sum(np.log(pdf_vals))

    x0 = np.array(copula.get_parameters(), dtype=float)
    bounds = copula.get_bounds() if hasattr(copula, "get_bounds") else [(None, None)] * len(x0)
    clean_bounds = _finite_bounds(bounds)

    if use_init:
        x0 = _robust_init_from_uv(copula, U, V, bounds)

    if quick and "maxiter" not in options:
        options = {**options, "maxiter": 50}

    result = minimize(neg_log_likelihood, x0, method=opti_method, bounds=clean_bounds, options=options)

    if not result.success:
        if verbose:
            print("[IFM ERROR] Copula optimization failed")
            print("→ message:", result.message)
            print("→ initial guess:", x0)
            print("→ bounds:", clean_bounds)
        return None

    copula_params = result.x
    loglik = -result.fun
    copula.set_parameters(list(copula_params))
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

    if mode == "cmle":
        u, v = pseudo_obs(data)
        bounds = copula.get_bounds() if hasattr(copula, "get_bounds") else None
        clean_bounds = _finite_bounds(bounds or [(None, None)] * len(copula.get_parameters()))
        x0 = _robust_init_from_uv(copula, u, v, bounds or [])

        def nll(theta):
            copula.set_parameters(theta)
            pdf = copula.get_pdf(u, v)  # CMLE: get_pdf lit copula.params
            if np.any(pdf <= 0) or np.any(np.isnan(pdf)):
                return np.inf
            return -np.sum(np.log(pdf))

        res = minimize(nll, x0, method=optimizer, bounds=clean_bounds, options={"maxiter": maxiter})

        Uret, Vret = u, v  # pour Huang

    else:
        # IFM quick: use provided marginals to build U,V then optimize copula only
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

        bounds = copula.get_bounds() if hasattr(copula, "get_bounds") else None
        clean_bounds = _finite_bounds(bounds or [(None, None)] * len(copula.get_parameters()))
        x0 = _robust_init_from_uv(copula, U, V, bounds or [])

        def nll(theta):
            copula.set_parameters(theta)
            pdf = copula.get_pdf(U, V)  # IFM: pareil, params déjà set
            if np.any(pdf <= 0) or np.any(np.isnan(pdf)):
                return np.inf
            return -np.sum(np.log(pdf))

        res = minimize(nll, x0, method=optimizer, bounds=clean_bounds, options={"maxiter": maxiter})

        Uret, Vret = U, V  # pour Huang

    if not res.success:
        print("[QUICK FIT WARNING] Optimization not fully converged:", res.message)

    theta = res.x if res.success else x0
    ll = -res.fun if res.success else float("nan")

    # write-back sur la copule
    cur_len = len(copula.get_parameters())
    copula.set_parameters(np.asarray(theta, dtype=float)[:cur_len])
    if hasattr(copula, "set_log_likelihood"):
        copula.set_log_likelihood(float(ll))
    else:
        copula.log_likelihood_ = float(ll)

    if not return_metrics:
        return theta, ll

    # quick diagnostics: Huang tails
    try:
        lamU = huang_lambda(Uret, Vret, side="upper")
        lamL = huang_lambda(Uret, Vret, side="lower")
    except Exception:
        lamU = lamL = None

    return {"theta": theta, "loglik": ll, "lambdaU_huang": lamU, "lambdaL_huang": lamL}

