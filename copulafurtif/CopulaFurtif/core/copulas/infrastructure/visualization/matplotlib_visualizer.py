import matplotlib.pyplot as plt
import numpy as np
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel

class MatplotlibCopulaVisualizer:
    @staticmethod
    def plot_residual_heatmap(copula : CopulaModel, u, v, bins=50):
        u, v = np.asarray(u).flatten(), np.asarray(v).flatten()
        U, V = np.meshgrid(np.linspace(0, 1, bins), np.linspace(0, 1, bins), indexing="ij")
        emp = np.array([
            np.mean((u <= U[i, j]) & (v <= V[i, j])) for i in range(bins) for j in range(bins)
        ]).reshape(bins, bins)
        flat = [copula.get_cdf(ui, vi, copula.get_parameters())
                for ui, vi in zip(U.ravel(), V.ravel())]
        model = np.array(flat).reshape(bins, bins)
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