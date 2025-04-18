import numpy as np
from scipy.stats import beta, lognorm, t as student_t

from Service.Copulas.archimedean.frank import FrankCopula
from Service.Copulas.fitting.estimation import fit_mle


def generate_data_beta_lognorm_student(n=1000, rho=0.7, nu=4):
    """
    Generates (X, Y) with a t-Student copula and known marginals.
    X ~ Beta(2,5), Y ~ LogNorm(s=0.5, scale=exp(1))
    """
    cov = np.array([[1, rho], [rho, 1]])
    L = np.linalg.cholesky(cov)
    Z = np.random.randn(n, 2)
    chi2 = np.random.chisquare(nu, size=n)
    T = (Z @ L.T) / np.sqrt(chi2 / nu)[:, None]

    U = student_t.cdf(T[:, 0], df=nu)
    V = student_t.cdf(T[:, 1], df=nu)

    X = beta.ppf(U, a=2, b=5)
    Y = lognorm.ppf(V, s=0.5, scale=np.exp(1))
    return X, Y


def main():
    np.random.seed(42)

    # Step 1: Generate data
    X, Y = generate_data_beta_lognorm_student(n=5000, rho=0.7, nu=4)
    data = [X, Y]

    # Step 2: Define known marginals
    marginals_known = [
        {"distribution": beta, "a": 2, "b": 5, "loc": 0, "scale": 1},
        {"distribution": lognorm, "s": 0.5, "loc": 0, "scale": np.exp(1)}
    ]

    # Step 3: Fit the copula (Frank)
    frank = FrankCopula()
    fitted_params, loglik = fit_mle(data, frank, marginals=marginals_known,
                                    known_parameters=True,
                                    opti_method=frank.default_optim_method)
    frank.parameters = np.array(fitted_params)
    frank.n_obs = len(X)
    frank.log_likelihood_ = loglik

    # Step 4: Transform to uniform via known marginals
    u = beta.cdf(X, a=2, b=5)
    v = lognorm.cdf(Y, s=0.5, scale=np.exp(1))

    # Step 5: Plot residual heatmap
    frank.residual_heatmap(u, v, bins=500, cmap="coolwarm", show=True)


if __name__ == "__main__":
    main()
