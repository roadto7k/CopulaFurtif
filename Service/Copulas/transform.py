import numpy as np
from scipy.stats import norm, beta, lognorm

def to_uniform(x, dist_name, *args, **kwargs):
    """
    Transforme x en u = F_X(x) via la CDF donnée par dist_name.
    ex: dist_name='norm', args=(mu, sigma)
    """
    if dist_name == 'norm':
        return norm.cdf(x, *args, **kwargs)
    elif dist_name == 'beta':
        return beta.cdf(x, *args, **kwargs)
    elif dist_name == 'lognorm':
        return lognorm.cdf(x, *args, **kwargs)
    else:
        raise ValueError(f"Unknown dist_name {dist_name}")
