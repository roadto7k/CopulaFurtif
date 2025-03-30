import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta, lognorm, norm

from Service.Copulas.elliptical.gaussian import GaussianCopula
from Service.Copulas.fitting.estimation import cmle, fit_mle
import matplotlib.pyplot as plt

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

# Exemple d'utilisation:
# X, Y = generate_data_beta_lognorm(n=1000, rho=0.7)


def main():
    np.random.seed(42)
    X, Y = generate_data_beta_lognorm(n=1000, rho=0.7)
    data = [X, Y]

    # === Instantiate the Gaussian Copula ===
    copula = GaussianCopula()

    # === CMLE (Canonical MLE, known margins, no param fit) ===
    rho_hat_cmle, ll_cmle = cmle(copula, data)
    print(f"[CMLE] rho_hat = {rho_hat_cmle[0]:.4f}, loglik = {ll_cmle:.4f}")
    print(f"→ Copula internal params: {copula.parameters}")
    print(f"→ Copula log-likelihood:  {copula.log_likelihood_:.4f}\n")

    # === MLE with fully known margins (Beta + LogNorm with true params) ===
    marginals_known = [
        {"distribution": beta, "a": 2, "b": 5, "loc": 0, "scale": 1},
        {"distribution": lognorm, "s": 0.5, "loc": 0, "scale": np.exp(1)}
    ]
    rho_hat_mle_known, ll_known = fit_mle(data, copula, marginals=marginals_known, known_parameters=True)
    print(f"[MLE with known margins] rho_hat = {rho_hat_mle_known[0]:.4f}, loglik = {ll_known:.4f}")
    print(f"→ Copula internal params: {copula.parameters}")
    print(f"→ Copula log-likelihood:  {copula.log_likelihood_:.4f}\n")

    # === MLE with automatic estimation of marginal parameters ===
    marginals_auto = [
        {"distribution": "beta"},       # auto-init from data
        {"distribution": "lognorm"}     # auto-init from data
    ]
    params_mle, ll_mle = fit_mle(
        data,
        copula,
        marginals=marginals_auto,
        known_parameters=False
    )

    # === Decompose the output
    # [theta, a, b, loc1, scale1, s, loc2, scale2]
    rho_hat = params_mle[0]
    a_hat, b_hat = params_mle[1], params_mle[2]
    loc1, scale1 = params_mle[3], params_mle[4]
    s_hat = params_mle[5]
    loc2, scale2 = params_mle[6], params_mle[7]

    print(f"[Full MLE] rho_hat = {rho_hat:.4f}")
    print(f"--> Beta params:      a = {a_hat:.4f}, b = {b_hat:.4f}, loc = {loc1:.4f}, scale = {scale1:.4f}")
    print(f"--> LogNorm params:   s = {s_hat:.4f}, loc = {loc2:.4f}, scale = {scale2:.4f}")
    print(f"→ Copula internal params: {copula.parameters}")
    print(f"→ Copula log-likelihood:  {copula.log_likelihood_:.4f}")

    # === Visualization ===
    plt.figure()
    plt.scatter(X, Y, alpha=0.3, color='blue')
    plt.title("Simulated data: X ~ Beta(2,5), Y ~ LogNorm(0.5, exp(1))")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


