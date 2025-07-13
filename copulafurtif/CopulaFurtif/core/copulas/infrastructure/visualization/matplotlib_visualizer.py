import matplotlib.pyplot as plt
import numpy as np
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel
from CopulaFurtif.copula_utils import pseudo_obs

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
    
    @staticmethod
    def plot_tail_dependence(data, candidate_list, q_low=0.05, q_high=0.95):
        """
        Creates a two-panel plot of the pseudo-observations showing the lower and upper tail regions.
        Also displays a text box summarizing each candidate's theoretical tail dependence.

        Parameters
        ----------
        data : list or tuple of two arrays
            Raw data samples [X, Y].
        candidate_list : list
            List of candidate copula objects (their parameters should have been updated via MLE).
        q_low : float, optional
            Quantile for the lower tail (default=0.05).
        q_high : float, optional
            Quantile for the upper tail (default=0.95).
        """
        # Get pseudo-observations using the existing pseudo_obs() function.
        u, v = pseudo_obs(data)
        # Compute empirical tail dependence values.
        lower_mask = (u <= q_low) & (v <= q_low)
        upper_mask = (u > q_high) & (v > q_high)
        emp_lambda_L = np.sum(lower_mask) / np.sum(u <= q_low) if np.sum(u <= q_low) > 0 else 0.0
        emp_lambda_U = np.sum(upper_mask) / np.sum(u > q_high) if np.sum(u > q_high) > 0 else 0.0

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Lower tail plot
        axs[0].scatter(u, v, s=10, alpha=0.3, color='grey', label="All points")
        axs[0].scatter(u[lower_mask], v[lower_mask], s=10, color='red', label="Lower tail")
        axs[0].set_title(f"Lower Tail (u,v ≤ {q_low})\nEmpirical LTDC: {emp_lambda_L:.3f}")
        axs[0].set_xlabel("u")
        axs[0].set_ylabel("v")
        axs[0].grid(True)
        axs[0].legend()

        # Upper tail plot
        axs[1].scatter(u, v, s=10, alpha=0.3, color='grey', label="All points")
        axs[1].scatter(u[upper_mask], v[upper_mask], s=10, color='green', label="Upper tail")
        axs[1].set_title(f"Upper Tail (u,v > {q_high})\nEmpirical UTDC: {emp_lambda_U:.3f}")
        axs[1].set_xlabel("u")
        axs[1].set_ylabel("v")
        axs[1].grid(True)
        axs[1].legend()

        # Build a text summary of candidate theoretical tail dependencies.
        text_lines = ["Candidate Theoretical Tail Dependence:"]
        for copula in candidate_list:
            param = copula.get_parameters()
            ltdc = copula.LTDC(param)
            utdc = copula.UTDC(param)
            text_lines.append(f"{copula.get_name()}: LTDC = {ltdc:.3f}, UTDC = {utdc:.3f}")
        text_str = "\n".join(text_lines)

        # Add text box in the figure.
        fig.text(0.5, 0.02, text_str, ha="center", fontsize=10, bbox=dict(facecolor="white", alpha=0.8))
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.show()
