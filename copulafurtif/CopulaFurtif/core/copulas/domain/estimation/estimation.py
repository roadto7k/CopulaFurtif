import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata
from typing import Sequence, Tuple

from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel
from CopulaFurtif.core.copulas.domain.estimation.utils import (
    auto_initialize_marginal_params,
    flatten_theta,
    adapt_theta,
    log_likelihood_only_copula,
    log_likelihood_joint,
    evaluate_copula_log_pdf,
)

# ==============================================================================
# Pseudo-observations (empirical CDF ranks)
# ==============================================================================
def pseudo_obs(data: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute pseudo-observations from raw data using empirical CDF ranks.

    Uses average ranks so ties are handled more robustly than a double-argsort.

    Args:
        data: [X, Y] arrays of equal length

    Returns:
        (u, v) in (0,1)
    """
    if len(data) != 2:
        raise ValueError("Input must be a list or tuple with two elements [X, Y].")

    X, Y = (np.asarray(arr, dtype=float).ravel() for arr in data)
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same length.")

    n = len(X)

    def empirical_cdf_ranks(values):
        ranks = rankdata(values, method="average")
        return ranks / (n + 1.0)

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
    """
    Normalize parameter bounds for numerical optimization.

    Missing lower or upper bounds are represented by negative or positive
    infinity, respectively. Existing infinite bounds are preserved so that
    semi-infinite parameter domains are not replaced by arbitrary finite caps.

    Parameters
    ----------
    bounds_like : iterable of tuple
        Sequence of (lower, upper) parameter bounds.

    Returns
    -------
    list of tuple
        Normalized floating-point bounds.
    """
    clean = []

    for low, high in bounds_like:
        lo = -np.inf if low is None else float(low)
        hi = np.inf if high is None else float(high)

        if not lo < hi:
            raise ValueError(
                f"Invalid parameter bounds: lower={lo}, upper={hi}"
            )

        clean.append((lo, hi))

    return clean

def _shrink_open_bounds(
    bounds,
    eps_abs: float = 1e-9,
    eps_rel: float = 1e-12,
):
    """
    Shift each finite parameter bound slightly inside its admissible domain.

    Lower and upper bounds are handled independently so that semi-infinite
    domains such as (a, +inf) and (-inf, b) are supported correctly.

    This prevents numerical optimizers from evaluating parameters exactly on
    open boundaries rejected by strict parameter validation. Infinite bounds
    are preserved.

    Parameters
    ----------
    bounds : iterable of tuple
        Sequence of normalized (lower, upper) parameter bounds.

    eps_abs : float, optional
        Minimum absolute inward displacement applied to a finite boundary.

    eps_rel : float, optional
        Relative inward displacement based on the magnitude of each finite
        boundary.

    Returns
    -------
    list of tuple
        Parameter bounds shifted strictly inside each finite boundary.
    """
    out = []

    for lo, hi in bounds:
        lo = float(lo)
        hi = float(hi)

        if not lo < hi:
            raise ValueError(
                f"Invalid parameter bounds: lower={lo}, upper={hi}"
            )

        inner_lo = lo
        inner_hi = hi

        if np.isfinite(lo):
            delta_lo = max(
                float(eps_abs),
                float(eps_rel) * max(1.0, abs(lo)),
            )
            inner_lo = lo + delta_lo

        if np.isfinite(hi):
            delta_hi = max(
                float(eps_abs),
                float(eps_rel) * max(1.0, abs(hi)),
            )
            inner_hi = hi - delta_hi

        if inner_lo >= inner_hi:
            if np.isfinite(lo) and np.isfinite(hi):
                width = hi - lo
                delta = 0.49 * width

                inner_lo = lo + delta
                inner_hi = hi - delta
            else:
                raise ValueError(
                    f"Unable to shrink parameter bounds: "
                    f"lower={lo}, upper={hi}"
                )

        out.append((inner_lo, inner_hi))

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


def _shape_param_names(dist) -> list[str]:
    """Return scipy.stats shape parameter names in canonical order."""
    shapes = getattr(dist, "shapes", None)
    if not shapes:
        return []
    return [s.strip() for s in shapes.split(",")]


def _ordered_shape_params(m, dist):
    """
    Extract marginal shape parameters in scipy.stats canonical order.

    Supports both named shapes (e.g. a, b) and legacy generic keys shape_0,
    shape_1, ... to stay backward compatible with older saved marginal dicts.
    """
    names = _shape_param_names(dist)
    if names and all(name in m for name in names):
        return [m[name] for name in names]

    generic = sorted(
        [k for k in m.keys() if str(k).startswith("shape_")],
        key=lambda key: int(str(key).split("_")[1]) if str(key).split("_")[1].isdigit() else str(key),
    )
    if generic:
        return [m[k] for k in generic]

    keys = [k for k in m.keys() if k not in ("distribution", "loc", "scale")]
    keys = sorted(keys)
    return [m[k] for k in keys]


def _marginal_from_fit_result(dist, fitted_params) -> dict:
    """Build a marginal dict from scipy.stats.fit output using true shape names."""
    fitted_params = tuple(fitted_params)
    shape_names = _shape_param_names(dist)
    shape_count = len(shape_names)

    marginal = {"distribution": dist}
    for name, value in zip(shape_names, fitted_params[:shape_count]):
        marginal[name] = float(value)

    if shape_names:
        loc_val = fitted_params[shape_count]
        scale_val = fitted_params[shape_count + 1]
    else:
        loc_val = fitted_params[0]
        scale_val = fitted_params[1]

    marginal["loc"] = float(loc_val)
    marginal["scale"] = float(scale_val)
    return marginal


def _rebuild_fitted_marginals_from_param_vec(param_vec, marginals, margin_shapes_count, theta0_length):
    """Rebuild fitted marginal dicts from the joint MLE parameter vector."""
    rebuilt = []
    idx = theta0_length
    for marg, shape_count in zip(marginals, margin_shapes_count):
        dist = marg["distribution"]
        shape_names = _shape_param_names(dist)
        shape_vals = np.asarray(param_vec[idx: idx + shape_count], dtype=float)
        idx += shape_count
        loc_val = float(param_vec[idx])
        scale_val = float(param_vec[idx + 1])
        idx += 2

        marginal = {"distribution": dist}
        if shape_names:
            for name, value in zip(shape_names, shape_vals):
                marginal[name] = float(value)
        else:
            for j, value in enumerate(shape_vals):
                marginal[f"shape_{j}"] = float(value)
        marginal["loc"] = loc_val
        marginal["scale"] = scale_val
        rebuilt.append(marginal)
    return rebuilt


def _prepare_ifm_marginals(data, marginals):
    """Fit or normalize marginal dictionaries for IFM / quick IFM."""
    prepared = []
    for i, marg in enumerate(marginals):
        dist = marg["distribution"]
        xvals = np.asarray(data[i], dtype=float)

        if len(marg.keys()) == 1 and "distribution" in marg:
            prepared.append(auto_initialize_marginal_params(xvals, dist.name))
            continue

        loc_ = marg.get("loc", None)
        scale_ = marg.get("scale", None)
        if loc_ is None or scale_ is None:
            prepared.append(_marginal_from_fit_result(dist, dist.fit(xvals)))
            continue

        marginal = {"distribution": dist}
        ordered_shapes = _ordered_shape_params(marg, dist)
        shape_names = _shape_param_names(dist)
        if shape_names:
            for name, value in zip(shape_names, ordered_shapes):
                marginal[name] = float(value)
        else:
            for j, value in enumerate(ordered_shapes):
                marginal[f"shape_{j}"] = float(value)
        marginal["loc"] = float(loc_)
        marginal["scale"] = float(scale_)
        prepared.append(marginal)
    return prepared


def _uv_from_marginals(X: np.ndarray, Y: np.ndarray, marginals) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (U,V) from provided marginals dicts (already fitted or with guesses).

    IMPORTANT (math): shape parameters must be passed to scipy.stats in the
    exact order defined by dist.shapes, not by arbitrary dict iteration.

    Clips to (eps, 1-eps) for numerical stability.
    """
    m0, m1 = marginals
    dist0, dist1 = m0["distribution"], m1["distribution"]

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

def _fallback_optim_methods(primary_method):
    """
    Return the fallback optimizer order for a failed primary optimizer.

    The order is chosen so that the fallback algorithm has a numerical
    behavior different from the primary one.

    Parameters
    ----------
    primary_method : str
        Primary scipy.optimize.minimize method.

    Returns
    -------
    list
        Ordered fallback optimization methods.
    """
    primary = str(primary_method).upper()

    fallback_order = {
        "SLSQP": [
            "L-BFGS-B",
            "Powell",
        ],
        "POWELL": [
            "SLSQP",
            "L-BFGS-B",
        ],
        "L-BFGS-B": [
            "SLSQP",
            "Powell",
        ],
    }

    methods = fallback_order.get(
        primary,
        [
            "L-BFGS-B",
            "SLSQP",
            "Powell",
        ],
    )

    return [
        method
        for method in methods
        if method.upper() != primary
    ]


def _fallback_optimizer_options(options):
    """
    Keep only optimization options that are safe to reuse across the
    supported fallback optimizers.

    Method-specific options from the primary optimizer are intentionally not
    propagated to another optimizer.
    """
    if options is None:
        return {}

    safe_keys = {
        "maxiter",
        "disp",
    }

    return {
        key: value
        for key, value in options.items()
        if key in safe_keys
    }


def _objective_value_is_valid(
    value,
    big_value: float,
) -> bool:
    """
    Check whether an objective value represents a genuine finite evaluation
    rather than the numerical failure sentinel.
    """
    value = float(value)

    return (
        np.isfinite(value)
        and value < 0.5 * float(big_value)
    )


def _minimize_with_fallback(
    objective,
    x0,
    bounds,
    primary_method,
    options=None,
    verbose: bool = True,
    label: str = "FIT",
    big_value: float = 1e50,
    fallback_methods=None,
    degradation_atol: float = 1e-6,
    degradation_rtol: float = 1e-8,
):
    """
    Minimize an objective with a likelihood-degradation guard and optimizer
    fallback.

    The objective value at the initial point x0 is used as a safety baseline.

    An optimizer result is accepted only if:

        1. scipy reports success;
        2. the final objective is finite;
        3. the final objective is not a failure sentinel;
        4. the final objective is not worse than the initial objective beyond
           a small numerical tolerance.

    If the primary optimizer is rejected, alternative optimizers are run from
    the exact same initial point x0. The best acceptable fallback result is
    returned.

    Parameters
    ----------
    objective : callable
        Objective function to minimize.

    x0 : array-like
        Initial parameter vector.

    bounds : sequence of tuple
        Bounds passed to scipy.optimize.minimize.

    primary_method : str
        Primary optimization method.

    options : dict, optional
        Options for the primary optimizer.

    verbose : bool, optional
        Print optimizer diagnostics.

    label : str, optional
        Label used in diagnostic messages.

    big_value : float, optional
        Failure sentinel returned by the objective.

    fallback_methods : sequence of str, optional
        Explicit fallback optimizer order. If omitted, a default order is
        selected from the primary method.

    degradation_atol : float, optional
        Absolute tolerance allowed when comparing the final objective with
        the initial objective.

    degradation_rtol : float, optional
        Relative tolerance allowed when comparing the final objective with
        the initial objective.

    Returns
    -------
    tuple
        result, selected_method, initial_objective.

        If no optimizer produces an acceptable result, result and
        selected_method are None.
    """
    x0 = np.asarray(
        x0,
        dtype=float,
    ).ravel()

    lo = np.asarray(
        [bound[0] for bound in bounds],
        dtype=float,
    )

    hi = np.asarray(
        [bound[1] for bound in bounds],
        dtype=float,
    )

    x0 = np.clip(
        x0,
        lo,
        hi,
    )

    # --------------------------------------------------------------
    # Initial likelihood baseline
    # --------------------------------------------------------------
    initial_objective = float(
        objective(x0)
    )

    if not _objective_value_is_valid(
        initial_objective,
        big_value,
    ):
        if verbose:
            print(
                f"[{label} ERROR] Invalid objective "
                "at the initial point."
            )
            print(
                "→ Initial parameters:",
                x0,
            )
            print(
                "→ Initial objective:",
                initial_objective,
            )

        return None, None, initial_objective

    degradation_tolerance = max(
        float(degradation_atol),
        float(degradation_rtol)
        * max(
            1.0,
            abs(initial_objective),
        ),
    )

    primary_method = str(
        primary_method
    )

    if fallback_methods is None:
        fallback_methods = _fallback_optim_methods(
            primary_method
        )

    methods = [
        primary_method,
    ]

    for method in fallback_methods:
        method = str(method)

        if all(
            method.upper() != existing.upper()
            for existing in methods
        ):
            methods.append(method)

    valid_fallback_results = []

    # --------------------------------------------------------------
    # Optimizer attempts
    # --------------------------------------------------------------
    for attempt_index, method in enumerate(methods):
        is_primary = attempt_index == 0

        # Re-evaluate x0 before every optimizer attempt.
        #
        # This is important because some objectives mutate the copula
        # parameters during evaluation.
        reset_value = float(
            objective(x0)
        )

        if not _objective_value_is_valid(
            reset_value,
            big_value,
        ):
            if verbose:
                print(
                    f"[{label} ERROR] Failed to reset "
                    "the objective at the initial point."
                )

            break

        if is_primary:
            method_options = dict(
                options or {}
            )
        else:
            method_options = (
                _fallback_optimizer_options(
                    options
                )
            )

            if verbose:
                print(
                    f"[{label} FALLBACK] Trying "
                    f"optimizer '{method}' from the "
                    "original initial point."
                )

        try:
            result = minimize(
                objective,
                x0,
                method=method,
                bounds=bounds,
                options=method_options,
            )

        except Exception as exc:
            if verbose:
                print(
                    f"[{label} WARNING] Optimizer "
                    f"'{method}' crashed:",
                    exc,
                )

            continue

        # ----------------------------------------------------------
        # Independently re-evaluate the final point.
        #
        # Do not blindly trust result.fun.
        # ----------------------------------------------------------
        try:
            candidate = np.asarray(
                result.x,
                dtype=float,
            ).ravel()

            if candidate.shape != x0.shape:
                raise ValueError(
                    "Optimizer returned an invalid "
                    "parameter-vector shape."
                )

            candidate = np.clip(
                candidate,
                lo,
                hi,
            )

            final_objective = float(
                objective(candidate)
            )

        except Exception as exc:
            if verbose:
                print(
                    f"[{label} WARNING] Could not "
                    f"validate optimizer '{method}':",
                    exc,
                )

            continue

        result.x = candidate
        result.fun = final_objective

        objective_valid = (
            _objective_value_is_valid(
                final_objective,
                big_value,
            )
        )

        likelihood_preserved = (
            final_objective
            <= initial_objective
            + degradation_tolerance
        )

        acceptable = (
            bool(result.success)
            and objective_valid
            and likelihood_preserved
        )

        if acceptable:
            if is_primary:
                return (
                    result,
                    method,
                    initial_objective,
                )

            valid_fallback_results.append(
                (
                    method,
                    result,
                )
            )

            if verbose:
                print(
                    f"[{label} RECOVERY] Optimizer "
                    f"'{method}' produced a valid fit."
                )
                print(
                    "→ Initial NLL:",
                    initial_objective,
                )
                print(
                    "→ Final NLL:",
                    final_objective,
                )

            continue

        if verbose:
            print(
                f"[{label} WARNING] Optimizer "
                f"'{method}' rejected."
            )
            print(
                "→ success:",
                bool(result.success),
            )
            print(
                "→ initial NLL:",
                initial_objective,
            )
            print(
                "→ final NLL:",
                final_objective,
            )
            print(
                "→ message:",
                getattr(
                    result,
                    "message",
                    "",
                ),
            )

    # --------------------------------------------------------------
    # Select the best successful fallback
    # --------------------------------------------------------------
    if valid_fallback_results:
        selected_method, selected_result = min(
            valid_fallback_results,
            key=lambda item: float(
                item[1].fun
            ),
        )

        # Restore the selected parameter state for objectives mutating
        # the copula object.
        objective(
            selected_result.x
        )

        if verbose:
            print(
                f"[{label} RECOVERY] Selected "
                f"optimizer '{selected_method}'."
            )
            print(
                "→ Selected NLL:",
                float(
                    selected_result.fun
                ),
            )

        return (
            selected_result,
            selected_method,
            initial_objective,
        )

    # Restore x0 rather than leaving the copula at a divergent point.
    objective(x0)

    if verbose:
        print(
            f"[{label} FAILED] No optimizer produced "
            "an acceptable fit."
        )

    return None, None, initial_objective



# ==============================================================================
# CMLE (robust init, quick mode, optional Huang metrics)
# ==============================================================================
def _cmle(
    copula: CopulaModel,
    data,
    opti_method=None,
    options: dict = None,
    verbose: bool = True,
    use_init: bool = True,
    quick: bool = False,
    return_metrics: bool = False,
    inputs_are_uniform: bool = False,
):
    """
    Fit copula parameters by maximizing the copula log-likelihood.

    Two input modes are supported.

    inputs_are_uniform=False
        Raw observations are converted to rank-based pseudo-observations
        before optimization. This corresponds to canonical CML.

    inputs_are_uniform=True
        Inputs are already uniform PIT values and are used directly.
        This corresponds to the copula estimation stage of IFM.

    Parameters
    ----------
    copula : CopulaModel
        Copula model to fit.

    data : sequence
        Pair of input arrays (x, y) or (u, v).

    opti_method : str, optional
        Optimization method.

    options : dict, optional
        Optimizer options.

    verbose : bool, optional
        Enable fitting diagnostics.

    use_init : bool, optional
        Use robust data-driven parameter initialization.

    quick : bool, optional
        Use reduced optimizer iterations.

    return_metrics : bool, optional
        Return optional tail diagnostics.

    inputs_are_uniform : bool, optional
        If True, data are treated as already-uniform PIT values and are
        not rank-transformed.

    Returns
    -------
    tuple or None
        (fitted_params, loglik) or
        (fitted_params, loglik, extras).
    """
    if options is None:
        options = {}

    opti_method = _resolve_optim_method(
        copula,
        opti_method,
    )

    # ------------------------------------------------------------------
    # 1) Prepare copula observations
    # ------------------------------------------------------------------
    if inputs_are_uniform:
        if not isinstance(data, (list, tuple)) or len(data) != 2:
            raise ValueError(
                "inputs_are_uniform=True requires data=(u, v)."
            )

        u = np.asarray(
            data[0],
            dtype=float,
        ).ravel()

        v = np.asarray(
            data[1],
            dtype=float,
        ).ravel()

        if u.shape[0] != v.shape[0]:
            raise ValueError(
                "u and v must have the same length."
            )

        finite_mask = (
            np.isfinite(u)
            & np.isfinite(v)
        )

        u = u[finite_mask]
        v = v[finite_mask]

        if u.size == 0:
            raise ValueError(
                "No finite uniform observations available."
            )

        # Do not silently reinterpret non-uniform data as PIT values.
        tolerance = 1e-12

        outside_unit_square = (
            np.any(u < -tolerance)
            or np.any(u > 1.0 + tolerance)
            or np.any(v < -tolerance)
            or np.any(v > 1.0 + tolerance)
        )

        if outside_unit_square:
            raise ValueError(
                "inputs_are_uniform=True requires values in [0, 1]."
            )

        # Protect copula densities from exact 0/1 evaluations.
        eps_uv = 1e-10

        u = np.clip(
            u,
            eps_uv,
            1.0 - eps_uv,
        )

        v = np.clip(
            v,
            eps_uv,
            1.0 - eps_uv,
        )

    else:
        # Canonical CML:
        # raw observations -> empirical rank pseudo-observations.
        u, v = pseudo_obs(
            data
        )

    # ------------------------------------------------------------------
    # 2) Bounds and robust initialization
    # ------------------------------------------------------------------
    try:
        base = np.array(
            copula.get_parameters(),
            dtype=float,
        )

        bounds = (
            copula.get_bounds()
            if hasattr(copula, "get_bounds")
            else [(None, None)] * len(base)
        )

        clean_bounds = _finite_bounds(
            bounds
        )

        clean_bounds = _shrink_open_bounds(
            clean_bounds,
            eps_abs=1e-8,
            eps_rel=1e-12,
        )

        lo = np.array(
            [bound[0] for bound in clean_bounds],
            dtype=float,
        )

        hi = np.array(
            [bound[1] for bound in clean_bounds],
            dtype=float,
        )

        if use_init:
            x0 = _robust_init_from_uv(
                copula,
                u,
                v,
                bounds,
            )

        else:
            x0 = base.copy()

        x0 = np.asarray(
            x0,
            dtype=float,
        ).ravel()

        x0 = np.clip(
            x0,
            lo,
            hi,
        )

    except Exception as exc:
        print(
            "[CMLE ERROR] Invalid initial parameters or bounds:",
            exc,
        )
        return None

    if quick and "maxiter" not in options:
        options = {
            **options,
            "maxiter": 50,
        }

    BIG = 1e50

    # ------------------------------------------------------------------
    # 3) Negative copula log-likelihood
    # ------------------------------------------------------------------
    def neg_loglik(params_raw):
        try:
            params = np.asarray(
                params_raw,
                dtype=float,
            ).ravel()

            params = np.clip(
                params,
                lo,
                hi,
            )

            if len(params) != len(copula.get_parameters()):
                return BIG

            copula.set_parameters(
                params.tolist()
            )

            logpdf_vals = evaluate_copula_log_pdf(
                copula,
                u,
                v,
                pdf_floor=1e-300,
            )

            if np.any(
                ~np.isfinite(logpdf_vals)
            ):
                return BIG

            return float(
                -np.sum(logpdf_vals)
            )

        except Exception as err:
            if verbose:
                print(
                    "[CMLE LOG_LIKELIHOOD ERROR]",
                    err,
                )
                print(
                    "→ Params received:",
                    params_raw,
                )

            return BIG

    # ------------------------------------------------------------------
    # 4) Optimize with likelihood guard and fallback
    # ------------------------------------------------------------------
    result, selected_method, initial_nll = _minimize_with_fallback(
        objective=neg_loglik,
        x0=x0,
        bounds=clean_bounds,
        primary_method=opti_method,
        options=options,
        verbose=verbose,
        label=(
            f"CMLE/"
            f"{getattr(copula, 'name', 'Copula')}"
        ),
        big_value=BIG,
    )

    if result is None:
        return None

    # ------------------------------------------------------------------
    # 5) Output
    # ------------------------------------------------------------------
    if result.success:
        fitted_params = np.clip(
            np.asarray(
                result.x,
                dtype=float,
            ).ravel(),
            lo,
            hi,
        )

        loglik = float(
            -result.fun
        )

        try:
            copula.set_parameters(
                fitted_params.tolist()
            )

        except ValueError:
            eps = 1e-12

            fitted_params = np.clip(
                fitted_params,
                lo + eps,
                hi - eps,
            )

            copula.set_parameters(
                fitted_params.tolist()
            )

        _write_back_fit_metadata(
            copula=copula,
            log_likelihood=loglik,
            n_obs=len(u),
        )

        if not return_metrics:
            return fitted_params, loglik

        try:
            lamU = float(
                huang_lambda(
                    u,
                    v,
                    side="upper",
                )
            )

            lamL = float(
                huang_lambda(
                    u,
                    v,
                    side="lower",
                )
            )

        except Exception:
            lamU = None
            lamL = None

        extras = {
            "lambdaU_huang": lamU,
            "lambdaL_huang": lamL,
            "n_obs": len(u),
        }

        return (
            fitted_params,
            loglik,
            extras,
        )

    if verbose:
        print(
            f"[CMLE FAILED] for copula "
            f"'{getattr(copula, 'name', 'Unnamed Copula')}'"
        )

        print(
            "→ Initial guess:",
            x0,
        )

        print(
            "→ Bounds:",
            clean_bounds,
        )

        print(
            "→ Message:",
            result.message,
        )

    return None


# ==============================================================================
# MLE (robust init on UV, quick mode, optional Huang metrics)
# ==============================================================================
def _fit_mle(
    data,
    copula: CopulaModel,
    marginals,
    opti_method=None,
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

    opti_method = _resolve_optim_method(
        copula,
        opti_method,
    )

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
        shape_guesses = tuple(_ordered_shape_params(marg, dist))

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

        results, selected_method, initial_nll = _minimize_with_fallback(
            objective=objective,
            x0=x0,
            bounds=clean_bounds,
            primary_method=opti_method,
            options=options,
            verbose=verbose,
            label=f"MLE-COPULA/{getattr(copula, 'name', 'Copula')}",
            big_value=BIG,
        )
        if results is None:
            if verbose:
                print(
                    "Optimization failed with all available optimizers."
                )

            return None

        if verbose:
            print(
                "Method:",
                selected_method,
                " | success:",
                results.success,
                " | message:",
                results.message,
            )

        final_params = np.clip(np.asarray(results.x, dtype=float).ravel(), lo, hi)
        final_loglike = float(-results.fun)

        try:
            copula.set_parameters(final_params[:len(theta0)].tolist())
        except ValueError:
            eps = 1e-12
            copula.set_parameters(np.clip(final_params[:len(theta0)], lo + eps, hi - eps).tolist())

        _write_back_fit_metadata(
            copula=copula,
            log_likelihood=final_loglike,
            n_obs=len(X),
        )

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
                    param_vec=param_vec,
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

        results, selected_method, initial_nll = _minimize_with_fallback(
            objective=objective,
            x0=x0,
            bounds=clean_bounds,
            primary_method=opti_method,
            options=options,
            verbose=verbose,
            label=f"MLE-JOINT/{getattr(copula, 'name', 'Copula')}",
            big_value=BIG,
        )
        if results is None:
            raise RuntimeError(
                "Optimization failed with all available optimizers."
            )

        if verbose:
            print(
                "Method:",
                selected_method,
                " | success:",
                results.success,
                " | message:",
                results.message,
            )

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

        _write_back_fit_metadata(
            copula=copula,
            log_likelihood=final_loglike,
            n_obs=len(X),
        )

        if not return_metrics:
            return final_params, final_loglike

        # Huang tails (simple report) using the optimized marginal parameters
        try:
            fitted_marginals_final = _rebuild_fitted_marginals_from_param_vec(
                final_params, marginals, margin_shapes_count, len(theta0)
            )
            U, V = _uv_from_marginals(X, Y, fitted_marginals_final)
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
    opti_method=None,
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

    opti_method = _resolve_optim_method(
        copula,
        opti_method,
    )

    X, Y = data
    if len(X) != len(Y):
        raise ValueError("X and Y must have the same length.")

    # 1) Fit marginals if needed, else normalize user-provided marginal dictionaries
    fitted_marginals = _prepare_ifm_marginals(data, marginals)

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

            logpdf_vals = evaluate_copula_log_pdf(
                copula,
                U,
                V,
                param=theta,
                pdf_floor=1e-300,
            )

            if np.any(~np.isfinite(logpdf_vals)):
                return BIG

            return float(
                -np.sum(logpdf_vals)
            )
        except Exception:
            return BIG

    result, selected_method, initial_nll = _minimize_with_fallback(
        objective=neg_log_likelihood,
        x0=x0,
        bounds=clean_bounds,
        primary_method=opti_method,
        options=options,
        verbose=verbose,
        label=f"IFM/{getattr(copula, 'name', 'Copula')}",
        big_value=BIG,
    )

    if result is None:
        if verbose:
            print("[IFM ERROR] Copula optimization failed")
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

    _write_back_fit_metadata(
        copula=copula,
        log_likelihood=loglik,
        n_obs=len(U),
    )

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
    optimizer = None,
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

            logpdf = evaluate_copula_log_pdf(
                copula,
                u,
                v,
                pdf_floor=1e-300,
            )

            if np.any(~np.isfinite(logpdf)):
                return BIG

            return float(
                -np.sum(logpdf)
            )

        quick_fallback = _fallback_optim_methods(
            optimizer
        )[:1]

        res, selected_optimizer, initial_nll = _minimize_with_fallback(
            objective=nll,
            x0=x0,
            bounds=clean_bounds,
            primary_method=optimizer,
            options={
                "maxiter": maxiter,
            },
            verbose=True,
            label=f"QUICK-CMLE/{getattr(copula, 'name', 'Copula')}",
            big_value=BIG,
            fallback_methods=quick_fallback,
        )

        Uret, Vret = u, v

    else:
        if marginals is None:
            raise ValueError("marginals must be provided for mode='ifm'")

        X, Y = data
        fitted_marginals = _prepare_ifm_marginals(data, marginals)
        U, V = _uv_from_marginals(X, Y, fitted_marginals)

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

            logpdf = evaluate_copula_log_pdf(
                copula,
                U,
                V,
                pdf_floor=1e-300,
            )

            if np.any(~np.isfinite(logpdf)):
                return BIG

            return float(
                -np.sum(logpdf)
            )

        quick_fallback = _fallback_optim_methods(
            optimizer
        )[:1]

        res, selected_optimizer, initial_nll = _minimize_with_fallback(
            objective=nll,
            x0=x0,
            bounds=clean_bounds,
            primary_method=optimizer,
            options={
                "maxiter": maxiter,
            },
            verbose=True,
            label=f"QUICK-IFM/{getattr(copula, 'name', 'Copula')}",
            big_value=BIG,
            fallback_methods=quick_fallback,
        )

        Uret, Vret = U, V

    if res is None:
        print(
            "[QUICK FIT WARNING] No optimizer "
            "produced an acceptable result."
        )

        theta = np.asarray(
            x0,
            dtype=float,
        ).ravel()

        # Restore the initialization point explicitly.
        copula.set_parameters(
            theta.tolist()
        )

        # The copula must not retain stale fit metadata after a failed fit.
        copula.log_likelihood_ = None
        copula.n_obs = None

        if not return_metrics:
            return theta, float("nan")

        try:
            lamU = huang_lambda(
                Uret,
                Vret,
                side="upper",
            )

            lamL = huang_lambda(
                Uret,
                Vret,
                side="lower",
            )

        except Exception:
            lamU = lamL = None

        return {
            "theta": theta,
            "loglik": float("nan"),
            "lambdaU_huang": lamU,
            "lambdaL_huang": lamL,
        }

    theta = np.asarray(
        res.x,
        dtype=float,
    )

    ll = float(
        -res.fun
    )

    cur_len = len(copula.get_parameters())

    theta = np.asarray(theta, dtype=float).ravel()[:cur_len]
    try:
        copula.set_parameters(theta.tolist())
    except ValueError:
        eps = 1e-12
        theta = np.clip(theta, lo[:cur_len] + eps, hi[:cur_len] - eps)
        copula.set_parameters(theta.tolist())

    _write_back_fit_metadata(
        copula=copula,
        log_likelihood=ll,
        n_obs=len(Uret),
    )

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

def _write_back_fit_metadata(
    copula: CopulaModel,
    log_likelihood: float,
    n_obs: int,
) -> None:
    """
    Store fitted-model metadata required by goodness-of-fit and
    information-criterion calculations.

    A successful likelihood-based fit must persist both:

        log_likelihood_
            Maximized log-likelihood of the fitted model.

        n_obs
            Number of paired observations used in the fit.

    The observation count is required by criteria such as BIC:

        BIC = -2 * log(L_hat) + k * log(n)

    Parameters
    ----------
    copula : CopulaModel
        Fitted copula instance.

    log_likelihood : float
        Maximized model log-likelihood.

    n_obs : int
        Number of paired observations used for estimation.

    Returns
    -------
    None
    """
    log_likelihood = float(log_likelihood)
    n_obs = int(n_obs)

    if n_obs <= 0:
        raise ValueError(
            f"n_obs must be strictly positive, got {n_obs}."
        )

    if hasattr(copula, "set_log_likelihood"):
        copula.set_log_likelihood(log_likelihood)
    else:
        copula.log_likelihood_ = log_likelihood

    if hasattr(copula, "set_n_obs"):
        copula.set_n_obs(n_obs)
    else:
        copula.n_obs = n_obs



def _resolve_optim_method(
    copula: CopulaModel,
    opti_method=None,
) -> str:
    """
    Resolve the optimization method used for copula fitting.

    Priority is:

        1. Explicit method provided by the caller.
        2. Copula-specific default_optim_method.
        3. SLSQP as a global fallback.

    Parameters
    ----------
    copula : CopulaModel
        Copula being fitted.

    opti_method : str, optional
        Explicit optimization method requested by the caller.

    Returns
    -------
    str
        Optimization method passed to scipy.optimize.minimize.
    """
    if opti_method is not None:
        return str(opti_method)

    default_method = getattr(
        copula,
        "default_optim_method",
        None,
    )

    if default_method is not None:
        return str(default_method)

    return "SLSQP"

