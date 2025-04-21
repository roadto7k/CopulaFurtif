import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def plot_cdf(copula, plot_type='3d', Nsplit=80, levels=None, cmap="coolwarm"):
    grid = np.linspace(0, 1, Nsplit)
    U, V = np.meshgrid(grid, grid, indexing="ij")
    Z = copula.get_cdf(U.ravel(), V.ravel(), copula.parameters).reshape(Nsplit, Nsplit)

    if plot_type == '3d':
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(U, V, Z, cmap=cmap)
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_zlabel("C(u,v)")
        ax.set_title("Copula CDF Surface")
        plt.tight_layout()
    elif plot_type == 'contour':
        plt.contourf(U, V, Z, levels=levels if levels is not None else 10, cmap=cmap)
        plt.title("Copula CDF Contours")
        plt.xlabel("u")
        plt.ylabel("v")
        plt.colorbar()
        plt.tight_layout()
    plt.show()


def plot_pdf(copula, plot_type='3d', Nsplit=80, levels=None, cmap="viridis", log_scale=False):
    grid = np.linspace(0, 1, Nsplit)
    U, V = np.meshgrid(grid, grid, indexing="ij")
    Z = copula.get_pdf(U.ravel(), V.ravel(), copula.parameters).reshape(Nsplit, Nsplit)
    if log_scale:
        Z = np.log(Z + 1e-12)

    if plot_type == '3d':
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(U, V, Z, cmap=cmap)
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_zlabel("c(u,v)")
        ax.set_title("Copula PDF Surface")
        plt.tight_layout()
    elif plot_type == 'contour':
        plt.contourf(U, V, Z, levels=levels if levels is not None else 10, cmap=cmap)
        plt.title("Copula PDF Contours")
        plt.xlabel("u")
        plt.ylabel("v")
        plt.colorbar()
        plt.tight_layout()
    plt.show()


def plot_mpdf(copula, marginals, plot_type='contour', Nsplit=80, levels=10, cmap="terrain"):
    from scipy.stats import norm
    grid = np.linspace(0, 1, Nsplit)
    U, V = np.meshgrid(grid, grid, indexing="ij")
    Z = copula.get_pdf(U.ravel(), V.ravel(), copula.parameters).reshape(Nsplit, Nsplit)

    x_marg = marginals[0]['distribution'].ppf(U, **{k: v for k, v in marginals[0].items() if k != 'distribution'})
    y_marg = marginals[1]['distribution'].ppf(V, **{k: v for k, v in marginals[1].items() if k != 'distribution'})

    if plot_type == 'contour':
        plt.contourf(x_marg, y_marg, Z, levels=levels, cmap=cmap)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("PDF with Marginals")
        plt.colorbar()
        plt.tight_layout()
    plt.show()


def plot_arbitrage_frontiers(copula, alpha_low=0.05, alpha_high=0.95, levels=200, scatter=None, scatter_alpha=0.5):
    grid = np.linspace(0, 1, levels)
    U, V = np.meshgrid(grid, grid, indexing="ij")
    Z = copula.get_pdf(U.ravel(), V.ravel(), copula.parameters).reshape(levels, levels)

    plt.figure(figsize=(6, 5))
    contour = plt.contourf(U, V, Z, levels=15, cmap="plasma")
    plt.colorbar(contour)
    plt.title("Arbitrage Frontiers")
    plt.xlabel("u")
    plt.ylabel("v")

    if scatter:
        u_s, v_s = scatter
        plt.scatter(u_s, v_s, c='black', alpha=scatter_alpha, edgecolors='white')
    plt.tight_layout()
    plt.show()