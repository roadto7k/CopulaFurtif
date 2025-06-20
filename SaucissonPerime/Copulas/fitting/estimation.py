import numpy as np
from scipy.optimize import minimize

from SaucissonPerime.Copulas.fitting.utils import auto_initialize_marginal_params, flatten_theta, adapt_theta, \
    log_likelihood_only_copula, log_likelihood_joint


# ==============================================================================
# SCIPY OPTIMIZERS REFERENCE (for `scipy.optimize.minimize`)
# ==============================================================================

# === Derivative-based optimizers ===
# ⚙Use when you have smooth, differentiable functions (like log-likelihoods).

# 'L-BFGS-B' : Quasi-Newton method. Handles box constraints (bounds). Default for copulas with bounds.
#              - Does NOT support general inequality or equality constraints (just bounds).
#              - Fast and robust when gradients are smooth.

# 'SLSQP'    : Sequential Least Squares Programming. Handles bounds AND linear/nonlinear constraints.
#              - Sometimes fails near boundary (e.g., when a parameter like nu is close to its lower bound).
#              - Less stable for copulas with highly constrained parameters.

# 'trust-constr' : Trust region method. Supports bounds + equality + inequality constraints.
#                  - Very general but can be slow. Use if you need full constraint support.

# 'CG'       : Conjugate Gradient. No constraints support. Not ideal if you have bounds.
# 'BFGS'     : Quasi-Newton. No bounds support. Mostly academic unless constraints are added manually.

# === Derivative-free optimizers ===
# ⚙Use when function is noisy, discontinuous, or has unreliable gradients.

# 'Nelder-Mead' : Simplex method. No bounds or constraints. Only for prototyping.
#                 - Very robust, but slow and doesn't scale well in high dimensions.

# 'Powell'      : Direction set method. Handles bounds (via penalties internally), no gradients needed.
#                 - Works well for non-smooth functions.

# 'COBYLA'      : Constrained Optimization BY Linear Approximation. Inequality constraints supported.
#                 - No bounds support — bounds must be rephrased as constraints.

# 'TNC'         : Truncated Newton Conjugate-Gradient. Supports bounds only.
#                 - Similar to L-BFGS-B but less used.

# 'dogleg'      : Trust-region method for small problems. Only works with unconstrained problems.

# ==============================================================================
#   Recommended for copula estimation with bounds:
#     → 'L-BFGS-B'  : (robust, handles bounds perfectly)
#     → 'SLSQP'     : (if you need nonlinear constraints — test stability!)
#     → 'Powell'    : (if gradients are unstable / custom model)
# ==============================================================================






def pseudo_obs(data):
    """
    Compute pseudo-observations from raw data using empirical CDF ranks.

    Parameters
    ----------
    data : list or tuple of two arrays
        [X, Y] data samples

    Returns
    -------
    u, v : np.ndarray, np.ndarray
        Pseudo-observations from empirical CDFs, both in (0,1).
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


def cmle(copula, data, opti_method='SLSQP', options=None, verbose=True):
    """
    Canonical Maximum Likelihood Estimation (CMLE) using pseudo-observations.

    Parameters
    ----------
    copula : object with .get_pdf(), .parameters, .bounds_param
    data : list of [X, Y]
    opti_method : str
        Optimization algorithm (default: 'SLSQP')
    options : dict or None
        Extra options passed to scipy.optimize.minimize
    verbose : bool
        If True, prints debug info when optimization fails

    Returns
    -------
    tuple (fitted_params, log_likelihood) or None
    """
    if options is None:
        options = {}

    # === 1. Preprocess input ===
    try:
        u, v = pseudo_obs(data)
        copula.n_obs = len(u)
    except Exception as e:
        print("[CMLE ERROR] Failed to compute pseudo-observations:", e)
        return None

    # === 2. Sanitize initial guess and bounds ===
    try:
        x0 = np.array(copula.parameters, dtype=float)
        bounds = copula.bounds_param if hasattr(copula, "bounds_param") else [(None, None)] * len(x0)

        # Fix values if out-of-bounds
        for i, (low, high) in enumerate(bounds):
            if low is not None and x0[i] < low:
                x0[i] = low + 1e-4
            if high is not None and x0[i] > high:
                x0[i] = high - 1e-4

        # Replace None bounds with large finite values for SLSQP compatibility
        clean_bounds = []
        for low, high in bounds:
            low = low if low is not None else -1e10
            high = high if high is not None else 1e10
            clean_bounds.append((low, high))

    except Exception as e:
        print("[CMLE ERROR] Invalid initial parameters or bounds:", e)
        return None

    # === 3. Define log-likelihood ===
    def log_likelihood(params_raw):
        try:
            # Force into list of floats
            if isinstance(params_raw, (float, int, np.float64)):
                params = [params_raw]
            else:
                params = list(params_raw)

            if len(params) != len(copula.parameters):
                raise ValueError(f"Parameter length mismatch: expected {len(copula.parameters)}, got {len(params)}")

            pdf_vals = copula.get_pdf(u, v, params)
            if np.any(pdf_vals <= 0) or np.any(np.isnan(pdf_vals)):
                return np.inf

            return -np.sum(np.log(pdf_vals))

        except Exception as e:
            if verbose:
                print("[CMLE LOG_LIKELIHOOD ERROR]", e)
                print("→ Params received:", params_raw)
            return np.inf

    # === 4. Run optimizer ===
    try:
        result = minimize(log_likelihood, x0, method=opti_method, bounds=clean_bounds, options=options)
    except Exception as e:
        print("[CMLE ERROR] Optimizer crashed:", e)
        return None

    # === 5. Handle result ===
    if result.success:
        fitted_params = result.x
        loglik = -result.fun
        copula.parameters = list(fitted_params)
        copula.log_likelihood_ = loglik
        return fitted_params, loglik
    else:
        if verbose:
            print(f"[CMLE FAILED] for copula '{getattr(copula, 'name', 'Unnamed Copula')}'")
            print("→ Initial guess:", x0)
            print("→ Bounds:", clean_bounds)
            print("→ Message:", result.message)
        return None


def fit_mle(data, copula, marginals, opti_method='SLSQP', known_parameters=False):
    """
    Fit a bivariate copula by Maximum Likelihood Estimation (MLE),
    allowing joint estimation of marginal distribution parameters
    (including shape parameters).

    Parameters
    ----------
    data : list of arrays
        [X, Y] observations
    copula : object
        Copula instance with attributes:
            - parameters (tuple or array): initial guess(es) for copula params
            - bounds_param (list of tuples, optional): bounds for copula parameters
            - type (str): e.g. 'mixture' or something else
            - get_pdf(u, v, theta): function returning copula PDF value at (u,v)
    marginals : list of dict
        Each dict must contain:
            - "distribution": a scipy.stats distribution
            - zero or more shape parameters (e.g. 'a', 'b' for Beta),
            - optional 'loc' (float),
            - optional 'scale' (float),
          The shape/loc/scale parameters found in the dict are:
            - Interpreted as fixed if known_parameters=True,
            - Interpreted as initial guesses if known_parameters=False.
    opti_method : str
        Optimization method passed to scipy.optimize.minimize (default: 'SLSQP')
    known_parameters : bool
        If True, we treat all marginal parameters as fixed/known, and only optimize copula parameters.
        If False, we optimize copula parameters AND the marginal parameters (shape, loc, scale).

    Returns
    -------
    tuple
        (optimized_parameters, max_log_likelihood)
        If success, 'optimized_parameters' is a flattened array:
          [copula_params, [marginal_1_shape(s), loc1, scale1], [marginal_2_shape(s), loc2, scale2]]
        If the optimizer fails, returns None.
    """

    if copula.type == "mixture":
        raise ValueError("MLE estimation for mixture copulas is not supported. Use CMLE instead.")

    X, Y = data
    copula.n_obs = len(data[0])

    # -------------------------------------------------------------------------
    # Auto-initialize marginals if parameters are missing
    # -------------------------------------------------------------------------
    if not known_parameters:
        for i in range(len(marginals)):
            marg = marginals[i]
            # If user only passed a string name or a dict with just the distribution
            if isinstance(marg["distribution"], str):
                marginals[i] = auto_initialize_marginal_params(data[i], marg["distribution"])
            elif len(marg.keys()) == 1 and "distribution" in marg:
                # dict given with only 'distribution' key, no params
                marginals[i] = auto_initialize_marginal_params(data[i], marg["distribution"].name)

    # -------------------------------------------------------------------------
    # 1) Validate marginal distributions and parse their shape parameters
    # -------------------------------------------------------------------------
    for idx, marg in enumerate(marginals):
        dist = marg["distribution"]
        dist_name = dist.name if hasattr(dist, "name") else f"marginal {idx}"

        # Extract shape parameter names (everything except distribution, loc, scale)
        shape_keys = [k for k in marg if k not in ("distribution", "loc", "scale")]
        shape_guesses = tuple(marg[k] for k in shape_keys)

        x_vals = data[idx]
        loc_guess = marg.get("loc", 0.0)
        scale_guess = marg.get("scale", 1.0)

        if known_parameters:
            # Parameters are declared as known → strict validation required
            if scale_guess <= 0:
                raise ValueError(f"Distribution '{dist_name}' must have scale > 0. Got: {scale_guess}")

            try:
                pdf_vals = dist.pdf(x_vals, *shape_guesses, loc=loc_guess, scale=scale_guess)
                if np.any(np.isnan(pdf_vals)) or np.any(np.isinf(pdf_vals)) or np.any(pdf_vals < 0):
                    raise ValueError(f"PDF of distribution '{dist_name}' is invalid over the given data.")
            except Exception as e:
                raise ValueError(f"Error while evaluating PDF for distribution '{dist_name}': {e}")

        else:
            # ⚠ Parameters are only initial guesses → soft checks for numerical validity
            if scale_guess <= 0:
                print(f"[WARNING] Initial scale for distribution '{dist_name}' is non-positive ({scale_guess}).")

            try:
                pdf_vals = dist.pdf(x_vals, *shape_guesses, loc=loc_guess, scale=scale_guess)
                if np.any(np.isnan(pdf_vals)) or np.any(np.isinf(pdf_vals)):
                    print(f"[WARNING] Initial PDF values for distribution '{dist_name}' contain NaN or Inf. "
                          f"Optimization may fail.")
            except Exception as e:
                print(f"[WARNING] Failed to evaluate initial PDF for distribution '{dist_name}': {e}")

    # -------------------------------------------------------------------------
    # 2) Organize the copula parameter(s) initial guesses
    # -------------------------------------------------------------------------

    # Flatten copula parameter starts into a list
    theta0 = flatten_theta(copula.parameters)

    # -------------------------------------------------------------------------
    # 3) Build the optimization function
    #    - If known_parameters=True, we only optimize copula params.
    #    - If known_parameters=False, we optimize copula params AND marginals.
    # -------------------------------------------------------------------------
    # We'll gather all "free" marginal params in a single vector for shape+loc+scale
    margin_shapes_count = []
    for marg_dict in marginals:
        # the shape parameters are keys except distribution, loc, scale
        shape_keys = [k for k in marg_dict if k not in ("distribution", "loc", "scale")]
        margin_shapes_count.append(len(shape_keys))

    if known_parameters:
        # ---------------------------------------------------------------------
        # All marginals are fixed => we optimize only copula parameters
        # ---------------------------------------------------------------------

        x0 = np.array(theta0, dtype=float)
        bounds = copula.bounds_param if hasattr(copula, "bounds_param") else None

        # define an objective function that calls the external log-likelihood
        def objective(theta_array):
            return log_likelihood_only_copula(
                theta_array=theta_array,
                copula=copula,
                X=X,
                Y=Y,
                marginals=marginals,
                adapt_theta_func=adapt_theta
            )

        results = minimize(objective, x0,
                           method=opti_method, bounds=bounds)
        print("Method:", opti_method, " | success:", results.success,
              " | message:", results.message)

        if results.success:
            final_params = results.x
            final_loglike = -results.fun

            # STORE TO COPULA OBJECT
            copula.parameters = final_params[:len(theta0)]  # only copula params
            copula.log_likelihood_ = final_loglike

            return final_params, final_loglike
        else:
            print("Optimization failed")
            return None

    else:
        # ---------------------------------------------------------------------
        # known_parameters=False => estimate shapes + loc + scale for both marginals
        # ---------------------------------------------------------------------

        # We'll gather initial guesses for shape parameters from marginals.
        # If the user specified them (e.g. 'a': 2.0 for Beta), we use that as an init guess.
        # Then we add loc, scale to the vector (init guess = [0,1] if not in the dict).
        # We do this for each margin in order.
        margin_init_guesses = []
        for i, marg in enumerate(marginals):
            shape_keys = [k for k in marg if k not in ("distribution", "loc", "scale")]
            # shape param initial guesses
            shape0 = [float(marg[k]) for k in shape_keys]
            loc0 = float(marg.get("loc", 0.0))
            scale0 = float(marg.get("scale", 1.0))
            margin_init_guesses.append(shape0 + [loc0, scale0])

        # Flatten everything into one big init vector:
        # [copula_params, (shapeParams_m1, loc_m1, scale_m1), (shapeParams_m2, loc_m2, scale_m2)]
        x0 = theta0[:]
        for g in margin_init_guesses:
            x0.extend(g)
        x0 = np.array(x0, dtype=float)

        # Build default bounds
        # (the user might define copula.bounds_param for copula params,
        #  but shape params rarely have universal bounds, except maybe positivity for Beta, etc.)
        bounds = []
        if hasattr(copula, "bounds_param") and (copula.bounds_param is not None):
            bounds.extend(copula.bounds_param)
        else:
            # If no bounds are specified for copula params, just do None
            for _ in theta0:
                bounds.append((None, None))

        # For each margin, for each shape param, we do (None, None) by default.
        # Then for loc => (None, None), for scale => (1e-6, None).
        for i, (marg, shape_count) in enumerate(zip(marginals, margin_shapes_count)):
            # shape parameters => no explicit bound
            for _ in range(shape_count):
                bounds.append((None, None))
            # loc => no bound
            bounds.append((None, None))
            # scale => must be positive
            bounds.append((1e-6, None))

        def objective(param_vec):
            return log_likelihood_joint(
                param_vec=param_vec,
                copula=copula,
                X=X,
                Y=Y,
                marginals=marginals,
                margin_shapes_count=margin_shapes_count,
                adapt_theta_func=adapt_theta,
                theta0_length=len(theta0)
            )

        # Pre-check: catch NaNs or invalid scales in x0
        if np.any(np.isnan(x0)) or np.any(np.isinf(x0)):
            print("[ERROR] Initial guess x0 contains NaNs or Infs. Check marginal parameters.")
            raise ValueError("Invalid initial guess: contains NaNs or Infs.")

        # Optional: warn if any initial scale is <= 0
        scale_indices = [-1, -3]  # assuming layout ends with [loc2, scale2]
        for idx in scale_indices:
            if x0[idx] <= 0:
                print(f"[WARNING] Initial scale estimate x0[{idx}] = {x0[idx]:.4f} is non-positive.")

        # Try the optimization
        try:
            results = minimize(objective, x0, method=opti_method, bounds=bounds)
        except Exception as e:
            print(f"[ERROR] Optimization raised an exception: {e}")
            raise RuntimeError(f"Optimization crashed due to exception: {e}")

        print("Method:", opti_method, " | success:", results.success, " | message:", results.message)

        if results.success:
            final_params = results.x
            final_loglike = -results.fun

            # STORE TO COPULA OBJECT
            copula.parameters = final_params[:len(theta0)]  # copula params only
            copula.log_likelihood_ = final_loglike

            return final_params, final_loglike

        else:
            print("[ERROR] Optimization failed. Log-likelihood could not be maximized.")
            raise RuntimeError("Optimization failed: " + str(results.message))



def fit_ifm(data, copula, marginals, opti_method='SLSQP', options=None, verbose=True):
    """
    Inference Functions for Margins (IFM) for a bivariate copula.

    This method fits the marginal distributions independently, then transforms
    the data to (U, V) via each fitted margin's CDF, and finally fits the copula
    parameters by maximum likelihood on (U, V).

    It differs from a full joint MLE (where marginals and copula are all fitted
    at once) and is often more stable when margins are unknown or many parameters
    are involved.

    Parameters
    ----------
    data : list of arrays
        [X, Y] observations (each array of shape (n,))
    copula : object
        Copula instance with attributes:
            - parameters (list or array): initial guess(es) for copula params
            - bounds_param (list of tuples): bounds for each copula parameter
            - get_pdf(u, v, theta): method returning copula PDF at (u, v)
            - log_likelihood_ (float): gets filled with final log-likelihood
    marginals : list of dict
        Two dictionaries describing each margin. Each dict should specify:
          - "distribution": a scipy.stats distribution object
          - optionally shape parameters (e.g. 'a', 'b' for Beta)
          - optionally 'loc', 'scale'

        *If shape/loc/scale are not provided, IFM will fit them from the data using dist.fit()*
        *If they are provided, we treat them as fixed (known) and do not re-fit.*

    opti_method : str
        Optimization method for copula fitting, passed to scipy.optimize.minimize
        (default: 'SLSQP'). Other good choices: 'Powell', 'L-BFGS-B'.
    options : dict or None
        Extra options passed to scipy.optimize.minimize (e.g. {'maxiter': 1000}).
    verbose : bool
        If True, prints debug info on failure.

    Returns
    -------
    tuple (copula_params, log_likelihood) or None
        copula_params: final fitted parameters for the copula
        log_likelihood: maximum log-likelihood on (U, V)

    Example
    -------
    # >>> # Suppose we have data = [X, Y], each of length n=1000
    # >>> # marginals = [
    # >>> #     {'distribution': beta},      # no shape provided => fit a,b from data
    # >>> #     {'distribution': lognorm}    # no shape => fit s from data
    # >>> # ]
    # >>> res_ifm = fit_ifm(data, myStudentCopula, marginals, opti_method='Powell')
    # >>> if res_ifm is not None:
    # >>>     params, loglik = res_ifm
    # >>>     print("IFM => Copula params:", params, " log-lik:", loglik)
    """
    if options is None:
        options = {}

    # 1) Fit marginals individually if missing shape/loc/scale
    X, Y = data

    if len(X) != len(Y):
        raise ValueError("X and Y must have the same length.")

    fitted_marginals = []
    for i, marg in enumerate(marginals):
        dist = marg["distribution"]
        xvals = data[i]

        # Gather known shape/loc/scale
        shape_keys = [k for k in marg if k not in ("distribution", "loc", "scale")]
        known_shape_params = [marg[k] for k in shape_keys]
        loc_ = marg.get("loc", None)
        scale_ = marg.get("scale", None)

        # If any of them are missing => we do a dist.fit
        # If user gave shape and loc/scale => we assume it's fully known
        # If user only partially gave shape => it's a bit ambiguous
        # => We'll do a quick check: if "shape_keys" doesn't match the distribution's needed shapes, we still do fit
        needs_fit = False

        # Check if we have a full specification for shape params
        # For example, Beta needs 2 shape params (a,b). If we only have 1, or none, we do fit.
        # We can't easily parse it from scipy, so we'll just guess from dist.shapes if it exists
        if hasattr(dist, "shapes") and dist.shapes is not None:
            # e.g. for Beta: shapes='a, b' => 2 shape params needed
            shape_names = [s.strip() for s in dist.shapes.split(',')]
            if len(shape_names) != len(shape_keys):
                needs_fit = True

        # Check loc/scale
        if loc_ is None or scale_ is None:
            needs_fit = True

        # If we need to fit, do it now
        if needs_fit:
            # dist.fit returns (shapes..., loc, scale)
            fit_res = dist.fit(xvals)
            # We'll parse them out. The number of shape params depends on dist.shapes
            if hasattr(dist, "shapes") and dist.shapes is not None:
                shape_count = len(dist.shapes.split(','))
                shape_vals = fit_res[:shape_count]
                loc_val = fit_res[shape_count]
                scale_val = fit_res[shape_count + 1]
            else:
                # if shapes is None => no shape param => typical for e.g. norm
                shape_vals = []
                loc_val = fit_res[0]
                scale_val = fit_res[1]

            # Overwrite the dict with the newly fitted params
            # If user had partially provided shape, we overwrite everything
            marg_final = {
                "distribution": dist
            }
            # put shape i in the dict with some name. We can't be sure of the name so we'll do shape_0, shape_1, ...
            for j, sv in enumerate(shape_vals):
                marg_final[f"shape_{j}"] = sv
            marg_final["loc"] = loc_val
            marg_final["scale"] = scale_val

            fitted_marginals.append(marg_final)
        else:
            # We trust the user-provided shape, loc, scale as known
            # Just store them in a cleaned-up dict
            marg_final = {"distribution": dist}
            # shapes
            for k in shape_keys:
                marg_final[k] = marg[k]
            marg_final["loc"] = loc_ if loc_ is not None else 0.0
            marg_final["scale"] = scale_ if scale_ is not None else 1.0
            fitted_marginals.append(marg_final)

    # 2) Build the pseudo-observations (U, V) from fitted marginals
    #    For each margin i => CDF_i(X_i,...)
    #    Then MLE only on the copula with these uniform (U, V).
    try:
        m0, m1 = fitted_marginals
        dist0, dist1 = m0["distribution"], m1["distribution"]

        # Gather final shape/loc/scale for margin 0
        shape0_keys = [k for k in m0 if k not in ("distribution", "loc", "scale")]
        shape0_vals = [m0[k] for k in shape0_keys]
        loc0 = m0["loc"]
        scale0 = m0["scale"]

        # Gather final shape/loc/scale for margin 1
        shape1_keys = [k for k in m1 if k not in ("distribution", "loc", "scale")]
        shape1_vals = [m1[k] for k in shape1_keys]
        loc1 = m1["loc"]
        scale1 = m1["scale"]

        U = dist0.cdf(X, *shape0_vals, loc=loc0, scale=scale0)
        V = dist1.cdf(Y, *shape1_vals, loc=loc1, scale=scale1)

        eps = 1e-12
        U = np.clip(U, eps, 1 - eps)
        V = np.clip(V, eps, 1 - eps)

        # 3) Fit the copula by MLE on (U, V)
        def neg_log_likelihood(theta):
            pdf_vals = copula.get_pdf(U, V, theta)
            # handle invalid pdf
            if np.any(pdf_vals <= 0) or np.any(np.isnan(pdf_vals)):
                return np.inf
            return -np.sum(np.log(pdf_vals))

        x0 = np.array(copula.parameters, dtype=float)
        bounds = getattr(copula, "bounds_param", None) or [(None, None)] * len(x0)

        # Replace None with large finite for numeric stability
        clean_bounds = []
        for b in bounds:
            low, high = b
            if low is None: low = -1e10
            if high is None: high = 1e10
            clean_bounds.append((low, high))

        result = minimize(neg_log_likelihood, x0, method=opti_method, bounds=clean_bounds, options=options)

        if result.success:
            copula_params = result.x
            loglik = -result.fun
            copula.parameters = list(copula_params)
            copula.log_likelihood_ = loglik

            return copula_params, loglik
        else:
            if verbose:
                print("[IFM ERROR] Copula optimization failed")
                print("→ message:", result.message)
                print("→ initial guess:", x0)
                print("→ bounds:", clean_bounds)
            return None

    except Exception as e:
        if verbose:
            print("[IFM ERROR] Issue building pseudo-observations or optimizing copula:", e)
        return None









