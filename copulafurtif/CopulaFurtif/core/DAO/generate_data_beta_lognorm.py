import numpy as np
from scipy.stats import beta, lognorm, norm# def generate_data_beta_lognorm(n=1000, rho=0.7):
#     """
#     Generate bivariate data with controlled dependence using Gaussian copula and marginals.
#     """
#     from CopulaFurtif.core.copulas.domain.models.elliptical.gaussian import GaussianCopula
#     from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaParameters
#     copula = GaussianCopula()
#     copula._parameters.values = np.array([rho])
#     # copula.parameters = np.array([rho])
#     uv = copula.sample(n)
#     u, v = uv[:, 0], uv[:, 1]

#     x = beta.ppf(u, a=2, b=5)
#     y = lognorm.ppf(v, s=0.5)
#     return [x,y]

def generate_data_beta_lognorm(n=1000, rho=0.7):
    """
    Génère (X, Y) de taille n.
    X ~ Beta(2,5)
    Y ~ LogNorm(s=0.5, scale=exp(1))
    avec dépendance copule gaussienne de paramètre rho.
    """

    # 1) On génère Z ~ N(0,I) ensuite on corrèle via Cholesky
    cov = np.array([[1, rho],
                    [rho, 1]])
    L = np.linalg.cholesky(cov)
    Z = np.random.randn(n, 2)
    corr_Z = Z @ L.T  # vecteurs (Z1*, Z2*) corrélés

    # 2) On transforme Z* -> U,V via cdf normale
    U = norm.cdf(corr_Z[:, 0])
    V = norm.cdf(corr_Z[:, 1])

    # 3) On applique les marges Beta et Lognorm
    #    => On obtient X et Y
    X = beta.ppf(U, a=2, b=5)
    #   (remarque: Beta(2,5) dans [0,1])
    Y = lognorm.ppf(V, s=0.5, scale=np.exp(1))

    return X, Y