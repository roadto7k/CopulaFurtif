import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta, lognorm, norm

from Service.Copulas.elliptical.gaussian import GaussianCopula
from Service.Copulas.fitting.estimation import cmle, fit_mle, fit_ifm
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

    # === Step 0: Generate data with known Gaussian copula ===
    true_rho = 0.7
    X, Y = generate_data_beta_lognorm(n=10000, rho=true_rho)
    data = [X, Y]

    print("=== Data generated with Gaussian Copula ===")
    print(f"True rho: {true_rho}")
    print("=" * 50)

    # === Step 1: Instantiate Gaussian Copula ===
    copula = GaussianCopula()

    # === Step 2: CMLE (Canonical MLE using pseudo-observations) ===
    print("→ [Step 1] CMLE (Canonical MLE)")
    res = cmle(copula, data)
    if res is None:
        print("❌ CMLE failed.\n")
    else:
        rho_cmle = res[0][0]
        ll_cmle = res[1]
        print(f"✅ CMLE succeeded")
        print(f"   rho = {rho_cmle:.4f}")
        print(f"   log-likelihood = {ll_cmle:.4f}\n")

    # === Step 3: MLE with known marginals ===
    print("→ [Step 2] MLE with known marginals")
    marginals_known = [
        {"distribution": beta, "a": 2, "b": 5, "loc": 0, "scale": 1},
        {"distribution": lognorm, "s": 0.5, "loc": 0, "scale": np.exp(1)}
    ]
    try:
        mle_known, ll_known = fit_mle(data, copula, marginals=marginals_known, known_parameters=True)
        rho_known = mle_known[0]
        print(f"✅ MLE with known margins succeeded")
        print(f"   rho = {rho_known:.4f}")
        print(f"   log-likelihood = {ll_known:.4f}\n")
    except Exception as e:
        print("❌ MLE with known margins failed:", e, "\n")

    # === Step 4: Full MLE (copula + marginals) ===
    print("→ [Step 3] Full MLE (copula + marginals)")
    marginals_auto = [
        {"distribution": "beta"},
        {"distribution": "lognorm"}
    ]
    try:
        full_params, ll_full = fit_mle(
            data,
            copula,
            marginals=marginals_auto,
            known_parameters=False
        )

        rho_mle = full_params[0]
        a_hat, b_hat = full_params[1], full_params[2]
        loc1, scale1 = full_params[3], full_params[4]
        s_hat = full_params[5]
        loc2, scale2 = full_params[6], full_params[7]

        print(f"✅ Full MLE succeeded")
        print(f"   rho = {rho_mle:.4f}")
        print(f"   Beta:    a = {a_hat:.4f}, b = {b_hat:.4f}, loc = {loc1:.4f}, scale = {scale1:.4f}")
        print(f"   LogNorm: s = {s_hat:.4f}, loc = {loc2:.4f}, scale = {scale2:.4f}")
        print(f"   log-likelihood = {ll_full:.4f}\n")
    except Exception as e:
        print("❌ Full MLE failed:", e, "\n")

    # === Step 5: IFM (Inference Functions for Margins) ===
    print("→ [Step 4] IFM (Inference Functions for Margins)")
    marginals_ifm = [
        {"distribution": beta},
        {"distribution": lognorm}
    ]
    try:
        res_ifm = fit_ifm(data, copula, marginals_ifm, opti_method='Powell')
        if res_ifm is None:
            print("❌ IFM failed.\n")
        else:
            rho_ifm = res_ifm[0][0]
            ll_ifm = res_ifm[1]
            print(f"✅ IFM succeeded")
            print(f"   rho = {rho_ifm:.4f}")
            print(f"   log-likelihood = {ll_ifm:.4f}\n")
    except Exception as e:
        print("❌ IFM crashed:", e, "\n")

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


