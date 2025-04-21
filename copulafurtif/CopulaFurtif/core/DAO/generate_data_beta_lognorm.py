import numpy as np
from scipy.stats import beta, lognorm

def generate_data_beta_lognorm(n=1000, rho=0.7):
    """
    Generate bivariate data with controlled dependence using Gaussian copula and marginals.
    """
    from CopulaFurtif.core.copulas.domain.models.elliptical.gaussian import GaussianCopula
    copula = GaussianCopula()
    copula.parameters = np.array([rho])
    uv = copula.sample(n)
    u, v = uv[:, 0], uv[:, 1]

    x = beta.ppf(u, a=2, b=5)
    y = lognorm.ppf(v, s=0.5)
    return [x,y]