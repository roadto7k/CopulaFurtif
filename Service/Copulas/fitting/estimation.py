import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.stats import rv_continuous


def auto_initialize_marginal_params(data, dist_name):
    """
    Automatically fit a distribution from scipy.stats to the given 1D data,
    and return a dictionary of its best-fit parameters.

    Parameters
    ----------
    data : array-like
        1D array of observations.
    dist_name : str
        Name of the scipy.stats distribution (e.g., "beta", "lognorm", "gamma", ...)

    Returns
    -------
    param_dict : dict
        A dictionary with keys:
            - "distribution": the scipy.stats distribution object
            - one key per shape parameter (if any), with their fitted values
            - "loc": the fitted location parameter
            - "scale": the fitted scale parameter

    Raises
    ------
    ValueError
        If the distribution name is invalid or if fitting fails.

    Example
    -------
    >>> auto_initialize_marginal_params(X, "beta")
    {'distribution': <scipy.stats._continuous_distns.beta_gen>,
     'a': 2.03, 'b': 4.97, 'loc': 0.01, 'scale': 0.98}
    """
    # Get all available continuous distributions from scipy.stats
    available_distributions = {
        name: getattr(stats, name) for name in dir(stats)
        if isinstance(getattr(stats, name), rv_continuous)
    }

    # Check if the provided name is valid
    if dist_name not in available_distributions:
        valid = sorted(available_distributions.keys())
        raise ValueError(
            f"Invalid distribution name '{dist_name}'. Must be one of:\n{valid}"
        )

    dist = available_distributions[dist_name]

    # Fit the distribution to the data
    try:
        fitted_params = dist.fit(data)
    except Exception as e:
        raise ValueError(f"Failed to fit distribution '{dist_name}': {e}")

    # Extract shape parameter names
    shape_names = []
    if hasattr(dist, 'shapes') and dist.shapes is not None:
        shape_names = [s.strip() for s in dist.shapes.split(',')]

    # Build the parameter dictionary
    param_dict = {"distribution": dist}
    for i, name in enumerate(shape_names):
        param_dict[name] = fitted_params[i]
    param_dict["loc"] = fitted_params[-2]
    param_dict["scale"] = fitted_params[-1]

    return param_dict


def pseudo_obs(data):
    """
    Compute pseudo-observations from raw data using empirical CDF ranks.

    Parameters
    ----------
    data : list or tuple of two arrays
        [X, Y] data samples

    Returns
    -------
    np.ndarray
        Array of shape (2, n) containing scaled ranks (pseudo-observations)
    """
    if len(data) != 2:
        raise ValueError("Input must be a list or tuple with two elements [X, Y].")

    X, Y = data
    n = len(X)

    def empirical_cdf_ranks(values):
        ranks = np.argsort(np.argsort(values)) + 1  # ranks in 1..n
        return ranks / (n + 1)  # scale to (0,1)

    u = empirical_cdf_ranks(X)
    v = empirical_cdf_ranks(Y)

    return np.vstack((u, v))

def cmle(copula, data, opti_method='SLSQP', options={}):
    """
    Canonical Maximum Likelihood Estimation (CMLE) using pseudo-observations.
    Fits the copula on transformed uniform margins.

    Parameters
    ----------
    copula : Copula object
        The copula instance (must have .get_pdf and .parameters).
    data : list of np.ndarray
        [X, Y] input data.
    opti_method : str
        Optimization method (default: 'SLSQP').
    options : dict
        Options for scipy.optimize.minimize.

    Returns
    -------
    tuple
        (estimated_params, log_likelihood)
    """

    psd_obs = pseudo_obs(data)
    copula.n_obs = len(data[0])

    def log_likelihood(parameters):
        if len(copula.bounds_param) == 1:
            params = [parameters]
        else:
            param1, param2 = parameters
            params = [param1, param2]
        logl = -np.sum(np.log(copula.get_pdf(psd_obs[0], psd_obs[1], params)))
        return logl

    # Select optimizer
    if copula.bounds_param[0] == (None, None):
        method = 'Nelder-Mead'
        results = minimize(log_likelihood, copula.parameters, method=method, options=options)
    else:
        method = opti_method
        results = minimize(log_likelihood, copula.parameters, method=method,
                           bounds=copula.bounds_param, options=options)

    # Handle results
    if results.success:
        # Store fitted values inside the copula object
        fitted_params = results.x if isinstance(results.x, np.ndarray) else np.array([results.x])
        copula.parameters = fitted_params
        copula.log_likelihood_ = -results.fun

        return fitted_params, -results.fun
    else:
        print(f"[ERROR] Optimization failed for {copula.name or copula.family}")
        print("Details:", results.message)
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
            # ✅ Parameters are declared as known → strict validation required
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
    def adapt_theta(theta_array):
        """Convert a flattened slice of parameters (theta) back to the
        copula's required format if needed."""
        if isinstance(copula.parameters, tuple):
            return tuple(theta_array[:len(copula.parameters)])
        elif isinstance(copula.parameters, np.ndarray) and copula.parameters.shape == ():
            return [theta_array[0]]
        else:
            return list(theta_array[:len(copula.parameters)])

    # Flatten copula parameter starts into a list
    if isinstance(copula.parameters, tuple):
        theta0 = [float(x) for x in copula.parameters]
    elif isinstance(copula.parameters, np.ndarray) and copula.parameters.shape == ():
        theta0 = [copula.parameters.item()]
    else:
        theta0 = list(copula.parameters)

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
        def log_likelihood_only_copula(theta_array):
            theta = adapt_theta(theta_array)

            # For each margin, read the user-supplied (fixed) shape, loc, scale
            shape_keys_0 = [k for k in marginals[0] if k not in ("distribution", "loc", "scale")]
            shape_keys_1 = [k for k in marginals[1] if k not in ("distribution", "loc", "scale")]
            shape_params_0 = tuple(marginals[0][key] for key in shape_keys_0)
            shape_params_1 = tuple(marginals[1][key] for key in shape_keys_1)

            loc1, scale1 = marginals[0].get('loc', 0), marginals[0].get('scale', 1)
            loc2, scale2 = marginals[1].get('loc', 0), marginals[1].get('scale', 1)

            # Evaluate cdf and pdf for each margin
            u = marginals[0]['distribution'].cdf(X, *shape_params_0, loc=loc1, scale=scale1)
            v = marginals[1]['distribution'].cdf(Y, *shape_params_1, loc=loc2, scale=scale2)
            pdf1 = marginals[0]['distribution'].pdf(X, *shape_params_0, loc=loc1, scale=scale1)
            pdf2 = marginals[1]['distribution'].pdf(Y, *shape_params_1, loc=loc2, scale=scale2)

            # Evaluate copula PDF
            cop_pdf = copula.get_pdf(u, v, theta)
            # sum of log(pdf_cop * pdf_margin1 * pdf_margin2)
            return -np.sum(np.log(cop_pdf) + np.log(pdf1) + np.log(pdf2))

        x0 = np.array(theta0, dtype=float)
        bounds = copula.bounds_param if hasattr(copula, "bounds_param") else None

        results = minimize(log_likelihood_only_copula, x0,
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

        def log_likelihood_joint(param_vec):
            # param_vec layout:
            # 1) copula params
            # 2) margin1 shape param(s) + loc + scale
            # 3) margin2 shape param(s) + loc + scale
            p = len(theta0)

            # Reconstruct copula parameters
            theta = adapt_theta(param_vec)

            # keep track of the index from which margin-1's params start
            idx_current = p
            # margin 1
            shape_count_1 = margin_shapes_count[0]
            shape_params_1 = param_vec[idx_current : idx_current + shape_count_1]
            idx_current += shape_count_1
            loc1 = param_vec[idx_current]
            scale1 = param_vec[idx_current + 1]
            idx_current += 2

            # margin 2
            shape_count_2 = margin_shapes_count[1]
            shape_params_2 = param_vec[idx_current : idx_current + shape_count_2]
            idx_current += shape_count_2
            loc2 = param_vec[idx_current]
            scale2 = param_vec[idx_current + 1]
            idx_current += 2

            dist0 = marginals[0]['distribution']
            dist1 = marginals[1]['distribution']

            # Evaluate CDF/PDF for margin 1
            u = dist0.cdf(X, *shape_params_1, loc=loc1, scale=scale1)
            pdf1 = dist0.pdf(X, *shape_params_1, loc=loc1, scale=scale1)

            # Evaluate CDF/PDF for margin 2
            v = dist1.cdf(Y, *shape_params_2, loc=loc2, scale=scale2)
            pdf2 = dist1.pdf(Y, *shape_params_2, loc=loc2, scale=scale2)

            # Copula PDF
            cop_pdf = copula.get_pdf(u, v, theta)

            # Negative log-likelihood
            return -np.sum(np.log(cop_pdf) + np.log(pdf1) + np.log(pdf2))

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
            results = minimize(log_likelihood_joint, x0, method=opti_method, bounds=bounds)
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








