import numpy as np
from scipy.stats import beta, lognorm
# from CopulaFurtif.core.copulas.infrastructure.registry import register_all_copulas
from copulafurtif.CopulaFurtif.core.DAO.generate_data_beta_lognorm import generate_data_beta_lognorm
from CopulaFurtif.copulas import CopulaFactory, CopulaType
from CopulaFurtif.visualization import MatplotlibCopulaVisualizer
from CopulaFurtif.copulas import CopulaFitter
from CopulaFurtif.copulas import CopulaDiagnostics
from CopulaFurtif.copula_utils import pseudo_obs

def main():
    np.random.seed(42)
    # register_all_copulas()

    print("=== Step 0: Generate data with known Gaussian copula ===")
    true_rho = 0.7
    data = generate_data_beta_lognorm(n=5000, rho=true_rho)
    print(f"True rho: {true_rho}\n")

    # === Step 1: Instantiate copula ===
    copula = CopulaFactory.create(CopulaType.GAUSSIAN)
    
    copula.set_parameters([0.9])
    # === Step 2: Fit (CMLE) ===
    print("→ Step 1: CMLE")
    usecase = CopulaFitter()
    res = usecase.fit_cmle(data, copula)
    if res:
        rho_cmle = res[0][0]
        print(f"✅ CMLE succeeded: rho = {rho_cmle:.4f}\n")
    else:
        print("❌ CMLE failed.\n")

    # === Step 3: Fit MLE with known marginals ===
    print("→ Step 2: MLE with known marginals")
    marginals = [
        {"distribution": beta, "a": 2, "b": 5, "loc": 0, "scale": 1},
        {"distribution": lognorm, "s": 0.5, "loc": 0, "scale": np.exp(1)}
    ]
    try:
        mle_params, ll = usecase.fit_mle(data, copula, marginals, known_parameters=True)
        print(f"✅ MLE known succeeded: rho = {mle_params[0]:.4f}, log-lik = {ll:.4f}\n")
    except Exception as e:
        print("❌ MLE with known margins failed:", e, "\n")

    # === Step 4: Diagnostics ===
    print("→ Step 3: Diagnostics")
    diagnostics = CopulaDiagnostics()
    report = diagnostics.evaluate(data, copula)
    for k, v in report.items():
        print(f"{k}: {v}")

    # === Step 5: Residual Heatmap ===
    print("\n→ Step 4: Visualisation")
    u, v = pseudo_obs(data)
    MatplotlibCopulaVisualizer.plot_residual_heatmap(copula, u, v)


if __name__ == "__main__":
    main()
