import numpy as np
import scipy.stats as stats
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

def adapt_theta(theta_array, copula):
    """Convert a flattened slice of parameters (theta) back to the
    copula's required format if needed."""
    if isinstance(copula.parameters, tuple):
        return tuple(theta_array[:len(copula.parameters)])
    elif isinstance(copula.parameters, np.ndarray) and copula.parameters.shape == ():
        return [theta_array[0]]
    else:
        return list(theta_array[:len(copula.parameters)])

def flatten_theta(param):
    if isinstance(param, tuple):
        return [float(x) for x in param]
    elif isinstance(param, np.ndarray) and param.shape == ():
        return [param.item()]
    else:
        return list(param)

# utils.py
import numpy as np

def log_likelihood_only_copula(theta_array, copula, X, Y, marginals, adapt_theta_func):
    """
    Compute the negative log-likelihood for the copula alone
    (assuming the marginal parameters are fixed).
    """
    # 1) Reconstruct copula parameters
    theta = adapt_theta_func(theta_array, copula)

    # 2) Fetch fixed marginal parameters
    shape_keys_0 = [k for k in marginals[0] if k not in ("distribution", "loc", "scale")]
    shape_keys_1 = [k for k in marginals[1] if k not in ("distribution", "loc", "scale")]

    shape_params_0 = tuple(marginals[0][key] for key in shape_keys_0)
    shape_params_1 = tuple(marginals[1][key] for key in shape_keys_1)

    loc1, scale1 = marginals[0].get('loc', 0.0), marginals[0].get('scale', 1.0)
    loc2, scale2 = marginals[1].get('loc', 0.0), marginals[1].get('scale', 1.0)

    # 3) Compute cdf/pdf for each margin
    dist0 = marginals[0]['distribution']
    dist1 = marginals[1]['distribution']

    u = dist0.cdf(X, *shape_params_0, loc=loc1, scale=scale1)
    v = dist1.cdf(Y, *shape_params_1, loc=loc2, scale=scale2)

    pdf1 = dist0.pdf(X, *shape_params_0, loc=loc1, scale=scale1)
    pdf2 = dist1.pdf(Y, *shape_params_1, loc=loc2, scale=scale2)

    # 4) Copula pdf
    cop_pdf = copula.get_pdf(u, v, theta)

    # 5) Negative log-likelihood
    return -np.sum(np.log(cop_pdf) + np.log(pdf1) + np.log(pdf2))


def log_likelihood_joint(param_vec,
                         copula,
                         X, Y,
                         marginals,
                         margin_shapes_count,
                         adapt_theta_func,
                         theta0_length):
    """
    Compute the negative log-likelihood for the copula + marginals,
    i.e. jointly estimating copula params + shape/loc/scale for both margins.
    """
    # param_vec layout:
    #   [ copula parameters (theta0_length of them),
    #     margin1 (shape(s), loc, scale),
    #     margin2 (shape(s), loc, scale) ]

    # 1) Reconstruct copula parameters from param_vec
    theta = adapt_theta_func(param_vec, copula)

    # 2) Unpack margin 1 parameters
    idx_current = theta0_length
    shape_count_1 = margin_shapes_count[0]
    shape_params_1 = param_vec[idx_current : idx_current + shape_count_1]
    idx_current += shape_count_1
    loc1 = param_vec[idx_current]
    scale1 = param_vec[idx_current + 1]
    idx_current += 2

    # 3) Unpack margin 2 parameters
    shape_count_2 = margin_shapes_count[1]
    shape_params_2 = param_vec[idx_current : idx_current + shape_count_2]
    idx_current += shape_count_2
    loc2 = param_vec[idx_current]
    scale2 = param_vec[idx_current + 1]
    idx_current += 2

    dist0 = marginals[0]['distribution']
    dist1 = marginals[1]['distribution']

    # 4) Evaluate CDF/PDF for each margin
    u = dist0.cdf(X, *shape_params_1, loc=loc1, scale=scale1)
    pdf1 = dist0.pdf(X, *shape_params_1, loc=loc1, scale=scale1)

    v = dist1.cdf(Y, *shape_params_2, loc=loc2, scale=scale2)
    pdf2 = dist1.pdf(Y, *shape_params_2, loc=loc2, scale=scale2)

    # 5) Copula PDF
    cop_pdf = copula.get_pdf(u, v, theta)

    # 6) Negative log-likelihood
    return -np.sum(np.log(cop_pdf) + np.log(pdf1) + np.log(pdf2))
