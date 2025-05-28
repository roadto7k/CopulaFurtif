import matplotlib.pyplot as plt
import numpy as np

class MatplotlibCopulaVisualizer:
    @staticmethod
    def plot_residual_heatmap(copula, u, v, bins=50):
        u, v = np.asarray(u).flatten(), np.asarray(v).flatten()
        U, V = np.meshgrid(np.linspace(0, 1, bins), np.linspace(0, 1, bins), indexing="ij")
        emp = np.array([
            np.mean((u <= U[i, j]) & (v <= V[i, j])) for i in range(bins) for j in range(bins)
        ]).reshape(bins, bins)
        model = copula.get_cdf(U.ravel(), V.ravel(), copula.parameters).reshape(bins, bins)
        residuals = emp - model
        plt.figure(figsize=(6, 5))
        plt.imshow(residuals, origin="lower", extent=[0, 1, 0, 1], cmap="coolwarm")
        plt.colorbar(label="Empirical - Model")
        plt.title(f"Residual Heatmap: {copula.name}")
        plt.xlabel("u")
        plt.ylabel("v")
        plt.tight_layout()
        plt.show()
        return residuals