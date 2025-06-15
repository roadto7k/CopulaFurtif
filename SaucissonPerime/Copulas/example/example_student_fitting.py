import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, lognorm
from scipy.stats import t as student_t

from SaucissonPerime.Copulas.elliptical.gaussian import GaussianCopula
from SaucissonPerime.Copulas.elliptical.student import StudentCopula
from SaucissonPerime.Copulas.fitting.estimation import cmle, fit_mle, fit_ifm


def generate_data_beta_lognorm_student(n=1000, rho=0.7, nu=4):
    """
    Génère (X, Y) avec dépendance suivant une copule t-Student.
    X ~ Beta(2,5)
    Y ~ LogNorm(s=0.5, scale=exp(1))

    Paramètres
    ----------
    n : int
        Nombre d'échantillons
    rho : float
        Corrélation entre les marges (paramètre de la copule)
    nu : float
        Degrés de liberté de la copule t
    """

    # 1) Génération t-copula
    cov = np.array([[1, rho], [rho, 1]])
    L = np.linalg.cholesky(cov)

    Z = np.random.standard_normal((n, 2))
    chi2 = np.random.chisquare(df=nu, size=n)
    T = (Z @ L.T) / np.sqrt(chi2 / nu)[:, None]

    # 2) Transformation en pseudo-observations uniformes
    U = student_t.cdf(T[:, 0], df=nu)
    V = student_t.cdf(T[:, 1], df=nu)

    # 3) Application des marges
    X = beta.ppf(U, a=2, b=5)
    Y = lognorm.ppf(V, s=0.5, scale=np.exp(1))

    return X, Y

# Exemple d'utilisation:
# X, Y = generate_data_beta_lognorm_student(n=1000, rho=0.7, nu=4)


def main():
    np.random.seed(42)

    # === Simulate data with t-Student copula structure ===
    true_rho, true_nu = 0.7, 7
    X, Y = generate_data_beta_lognorm_student(n=10000, rho=true_rho, nu=true_nu)
    data = [X, Y]

    print("=== Data generated with Student Copula ===")
    print(f"True rho: {true_rho}, true nu: {true_nu}")
    print("=" * 50)

    # === Instantiate the Student Copula ===
    copula = StudentCopula()

    # === 1) CMLE ===
    print("→ [Step 1] CMLE (Canonical MLE)")
    res = cmle(copula, data, opti_method=copula.default_optim_method)
    if res is None:
        print("❌ CMLE failed.\n")
    else:
        rho, nu = res[0]
        ll = res[1]
        print(f"✅ CMLE succeeded")
        print(f"   rho = {rho:.4f}, nu = {nu:.4f}")
        print(f"   log-likelihood = {ll:.4f}\n")

    # === 2) MLE with known marginals ===
    print("→ [Step 2] MLE with known marginals")
    marginals_known = [
        {"distribution": beta, "a": 2, "b": 5, "loc": 0, "scale": 1},
        {"distribution": lognorm, "s": 0.5, "loc": 0, "scale": np.exp(1)}
    ]
    try:
        mle_known, ll_known = fit_mle(data, copula, marginals=marginals_known, known_parameters=True,
                                      opti_method=copula.default_optim_method)
        rho_k, nu_k = mle_known[:2]
        print(f"✅ MLE succeeded")
        print(f"   rho = {rho_k:.4f}, nu = {nu_k:.4f}")
        print(f"   log-likelihood = {ll_known:.4f}\n")
    except Exception as e:
        print("❌ MLE with known margins failed:", e, "\n")

    # === 3) Full MLE ===
    print("→ [Step 3] Full MLE (copula + marginals)")
    marginals_auto = [
        {"distribution": "beta"},
        {"distribution": "lognorm"}
    ]
    try:
        full_params, ll_full = fit_mle(data, copula, marginals=marginals_auto,
                                       known_parameters=False, opti_method=copula.default_optim_method)

        rho_hat, nu_hat = full_params[0], full_params[1]
        a_hat, b_hat = full_params[2], full_params[3]
        loc1, scale1 = full_params[4], full_params[5]
        s_hat = full_params[6]
        loc2, scale2 = full_params[7], full_params[8]

        print(f"✅ Full MLE succeeded")
        print(f"   rho = {rho_hat:.4f}, nu = {nu_hat:.4f}")
        print(f"   Beta params: a = {a_hat:.4f}, b = {b_hat:.4f}, loc = {loc1:.4f}, scale = {scale1:.4f}")
        print(f"   LogNorm params: s = {s_hat:.4f}, loc = {loc2:.4f}, scale = {scale2:.4f}")
        print(f"   log-likelihood = {ll_full:.4f}\n")
    except Exception as e:
        print("❌ Full MLE failed:", e, "\n")

    # === 4) IFM ===
    print("→ [Step 4] IFM (Inference Functions for Margins)")
    marginals_ifm = [
        {"distribution": beta},
        {"distribution": lognorm}
    ]
    try:
        result_ifm = fit_ifm(data, copula, marginals_ifm, opti_method=copula.default_optim_method)
        if result_ifm is not None:
            rho_ifm, nu_ifm = result_ifm[0]
            ll_ifm = result_ifm[1]
            print(f"✅ IFM succeeded")
            print(f"   rho = {rho_ifm:.4f}, nu = {nu_ifm:.4f}")
            print(f"   log-likelihood = {ll_ifm:.4f}\n")
        else:
            print("❌ IFM failed to converge.\n")
    except Exception as e:
        print("❌ IFM threw an error:", e, "\n")

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


