import numpy as np
from scipy.stats import kendalltau


def compute_aic(copula):
    """
    Compute the Akaike Information Criterion (AIC) for a fitted copula.

    Args:
        copula (object): Fitted copula with attributes:
            - log_likelihood_ (float): Log-likelihood of the model.
            - bounds_param (Sequence): Parameter bounds.

    Returns:
        float: AIC score.
    """

    k = len(copula.bounds_param)
    return 2 * k - 2 * copula.log_likelihood_


def compute_bic(copula):
    """
    Compute the Bayesian Information Criterion (BIC) for a fitted copula.

    Args:
        copula (object): Fitted copula with attributes:
            - log_likelihood_ (float): Log-likelihood of the model.
            - bounds_param (Sequence): Parameter bounds.
            - n_obs (int): Number of observations.

    Returns:
        float: BIC score.

    Raises:
        ValueError: If `n_obs` is missing or None on the copula object.
    """

    if not hasattr(copula, "n_obs") or copula.n_obs is None:
        raise ValueError("n_obs (number of observations) is missing in copula object.")

    k = len(copula.bounds_param)
    return k * np.log(copula.n_obs) - 2 * copula.log_likelihood_


def compute_iad_score(copula, data):
    """
    Compute the Integrated Anderson-Darling (IAD) statistic between the empirical copula and a parametric copula model.

    Args:
        copula (object): Copula model with methods:
            - get_cdf(u, v, params) -> array-like: CDF values on a grid.
            - parameters (Sequence): Fitted parameter values.
        data (Sequence[array-like, array-like]): Two-element sequence [u, v] of pseudo-observations.

    Returns:
        float: IAD goodness-of-fit score (lower is better).

    Raises:
        ValueError: If `u` and `v` have different lengths.
    """

    # data must be a list [u, v] where u and v are 1D arrays
    u, v = data
    n = len(u)
    if len(v) != n:
        raise ValueError("Mismatch: len(u) != len(v)")

    # Use the fitted parameters of the copula
    params = copula.parameters

    # --- Construct the empirical copula ---
    # Sort the pseudo-observations to define the grid points.
    u_sorted = np.sort(u)  # shape: (n,)
    v_sorted = np.sort(v)  # shape: (n,)

    # Build comparison matrices via broadcasting:
    # below_u[i, j] = True if u[i] <= u_sorted[j]
    below_u = u[:, None] <= u_sorted[None, :]  # shape: (n, n)
    below_v = v[:, None] <= v_sorted[None, :]  # shape: (n, n)

    # For each grid (i,j), count the number of points (u[k], v[k])
    # such that u[k] <= u_sorted[j] and v[k] <= v_sorted[j].
    # Using np.dot: (below_u.T @ Below_v)[i,j] = sum_{k=1}^n (below_u[k, i] * Below_v[k, j])
    C_empirical = np.dot(below_u.T, below_v) / n  # shape: (n, n)

    # --- Construct the parametric (model) copula grid ---
    # We construct a regular grid in (1/n, 1-1/n)
    grid = np.linspace(1 / n, 1 - 1 / n, n)
    uu, vv = np.meshgrid(grid, grid, indexing='ij')  # uu and vv de shape (n, n)
    u_flat = uu.ravel()
    v_flat = vv.ravel()

    # Calculate the model: returns an array of shape (n, n)
    C_model = copula.get_cdf(u_flat, v_flat, params).reshape(n, n)

    # --- Calculating the IAD Score ---
    # Avoids divisions by zero using np.clip on the denominator
    eps = 1e-10
    denom = np.clip(C_model * (1 - C_model), eps, None)

    iad_score = np.sum(((C_empirical - C_model) ** 2) / denom)

    return iad_score


def AD_score(copula, data):
    """
    Compute the Anderson-Darling (AD) goodness-of-fit statistic between the empirical copula and a parametric model.

    Args:
        copula (object): Copula model with methods:
            - get_cdf(u, v, params) -> array-like: CDF values on a grid.
            - parameters (Sequence): Fitted parameter values.
        data (Sequence[array-like, array-like]): Two-element sequence [u, v] of pseudo-observations.

    Returns:
        float: AD goodness-of-fit score (lower is better, sensitive to tails).

    Raises:
        ValueError: If `u` and `v` have different lengths.
    """

    # Use fitted parameters if requested

    params = copula.parameters

    # Extract pseudo-observations
    u, v = data
    n = len(u)
    if len(v) != n:
        raise ValueError("Mismatch: len(u) != len(v)")

    # Sort the pseudo-observations to define the grid
    u_sorted = np.sort(u)  # shape: (n,)
    v_sorted = np.sort(v)  # shape: (n,)

    # Build the empirical copula matrix:
    # For each threshold (u_sorted[i], v_sorted[j]), compute the fraction of observations
    # with u_k <= u_sorted[i] and v_k <= v_sorted[j].
    below_u = u[:, None] <= u_sorted[None, :]  # shape: (n, n)
    below_v = v[:, None] <= v_sorted[None, :]  # shape: (n, n)
    C_empirical = np.dot(below_u.T, below_v) / n  # shape: (n, n)

    # Build a regular grid in (1/n, 1-1/n) to evaluate the model CDF.
    grid = np.linspace(1 / n, 1 - 1 / n, n)
    uu, vv = np.meshgrid(grid, grid, indexing='ij')  # shapes: (n, n)
    u_flat = uu.ravel()  # shape: (n*n,)
    v_flat = vv.ravel()  # shape: (n*n,)

    # Compute the theoretical copula CDF over the grid and reshape to (n, n)
    C_model = copula.get_cdf(u_flat, v_flat, params).reshape(n, n)

    # To avoid division by zero in weights, clip C_model to (eps, 1-eps)
    eps = 1e-10
    C_model = np.clip(C_model, eps, 1 - eps)

    # Define weights emphasizing the tails
    weights = C_model * (1 - C_model)

    # Compute the maximum weighted squared deviation (the AD score)
    ad_score = np.max(((C_empirical - C_model) ** 2) / weights)

    return ad_score


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
        tau_theoretical = copula.kendall_tau(copula.parameters)
    except Exception as e:
        print(f"[WARNING] Failed to compute theoretical Kendall's tau: {e}")
        return np.nan

    return abs(tau_empirical - tau_theoretical)