import numpy as np
import scipy.stats as stats
from scipy.stats import rv_continuous
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel

def auto_initialize_marginal_params(data, dist_name):
    """
    Automatically fit a scipy.stats distribution to 1D data and return best-fit parameters.

    Args:
        data (array-like): 1D array of observations.
        dist_name (str): Name of the scipy.stats distribution (e.g., 'beta', 'gamma').

    Returns:
        dict: Dictionary containing:
            - "distribution": scipy.stats distribution object,
            - one entry per shape parameter with fitted values,
            - "loc": fitted location parameter,
            - "scale": fitted scale parameter.

    Raises:
        ValueError: If the distribution name is invalid or fitting fails.

    Examples:
        > auto_initialize_marginal_params(X, "beta")
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

def adapt_theta(theta_array, copula : CopulaModel):
    """
    Convert a flattened array of parameters back to the copula's required format.

    Args:
        theta_array (array-like): Flattened parameter values.
        copula (object): Copula instance whose .parameters attribute defines the required format.

    Returns:
        tuple or list: Parameters formatted to match copula.parameters.
    """

    if isinstance(copula.get_parameters(), tuple):
        return tuple(theta_array[:len(copula.get_parameters())])
    elif isinstance(copula.get_parameters(), np.ndarray) and copula.get_parameters().shape == ():
        return [theta_array[0]]
    else:
        return list(theta_array[:len(copula.get_parameters())])

def flatten_theta(param):
    """
     copula parameter container into a list of floats.

    Args:
        param (tuple, numpy.ndarray, or list): Copula.parameters in tuple, scalar array, or list form.

    Returns:
        list[float]: Flattened list of parameter values.
    """
    if isinstance(param, tuple):
        return [float(x) for x in param]
    elif isinstance(param, np.ndarray) and param.shape == ():
        return [param.item()]
    else:
        return list(param)


def log_likelihood_only_copula(theta_array, copula: CopulaModel, X, Y, marginals, adapt_theta_func):
    """
    Negative log-likelihood of the COPULA ONLY, assuming FIXED marginals.

    Model:
      u_i = F_X(x_i)  (fixed)
      v_i = F_Y(y_i)  (fixed)
      log L_c(theta) = sum_i log c_theta(u_i, v_i)

    Returns:
      - sum log c_theta(u_i, v_i)
    """
    theta = adapt_theta_func(theta_array, copula)

    dist0 = marginals[0]["distribution"]
    dist1 = marginals[1]["distribution"]

    def _ordered_shape_params(m, dist):
        shapes = getattr(dist, "shapes", None)
        if shapes:
            names = [s.strip() for s in shapes.split(",")]
            return tuple(m[name] for name in names)
        keys = [k for k in m.keys() if k not in ("distribution", "loc", "scale")]
        keys = sorted(keys)
        return tuple(m[k] for k in keys)

    shape_params_0 = _ordered_shape_params(marginals[0], dist0)
    shape_params_1 = _ordered_shape_params(marginals[1], dist1)

    loc1, scale1 = marginals[0].get("loc", 0.0), marginals[0].get("scale", 1.0)
    loc2, scale2 = marginals[1].get("loc", 0.0), marginals[1].get("scale", 1.0)

    u = dist0.cdf(X, *shape_params_0, loc=loc1, scale=scale1)
    v = dist1.cdf(Y, *shape_params_1, loc=loc2, scale=scale2)

    eps_u = 1e-12
    u = np.clip(u, eps_u, 1 - eps_u)
    v = np.clip(v, eps_u, 1 - eps_u)

    cop_pdf = copula.get_pdf(u, v, theta)
    eps = 1e-12
    cop_pdf = np.clip(cop_pdf, eps, None)

    return -np.sum(np.log(cop_pdf))


def log_likelihood_joint(param_vec,
                         copula : CopulaModel,
                         X, Y,
                         marginals,
                         margin_shapes_count,
                         adapt_theta_func,
                         theta0_length):
    """
    Compute the negative joint log-likelihood for copula and marginal distributions.

    Args:
        param_vec (array-like): Flattened vector [copula parameters, margin1 params, margin2 params].
        copula (object): Copula instance with method get_pdf(u, v, theta).
        X (array-like): Observations for the first margin.
        Y (array-like): Observations for the second margin.
        marginals (Sequence[dict]): Marginal distribution specifications.
        margin_shapes_count (Sequence[int]): Number of shape parameters for each margin.
        adapt_theta_func (callable): Function to convert param_vec to copula parameter format.
        theta0_length (int): Number of copula parameters at the start of param_vec.

    Returns:
        float: Negative joint log-likelihood combining copula and marginal PDFs.
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
    eps = 1e-8
    cop_pdf = np.clip(cop_pdf, eps, None)
    pdf1 = np.clip(pdf1, eps, None)
    pdf2 = np.clip(pdf2, eps, None)
    # 6) Negative log-likelihood
    return -np.sum(np.log(cop_pdf) + np.log(pdf1) + np.log(pdf2))


