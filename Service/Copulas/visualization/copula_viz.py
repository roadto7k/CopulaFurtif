import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_bivariate_3d(copula, Nsplit=50, **kwargs):
    """
    Plot the 3D surface of the bivariate PDF of a copula using its internal parameters.

    Parameters
    ----------
    copula : object
        Instance of a copula with method get_pdf(u, v, param) and attributes `name` and `parameters`.
    Nsplit : int, optional
        Number of grid points per axis (default=50).
    **kwargs :
        Extra parameters passed to ax.plot_surface.
    """
    u = np.linspace(0.01, 0.99, Nsplit)
    U, V = np.meshgrid(u, u)
    # Compute pdf on grid using internal parameters
    Z = np.array([copula.get_pdf(ui, vi, copula.parameters) for ui, vi in zip(U.ravel(), V.ravel())]).reshape(U.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(U, V, Z, **kwargs)
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_zlabel('PDF')
    plt.title(f'{copula.name} Copula PDF (3D)')
    plt.show()


def plot_bivariate_contour(copula, Nsplit=50, levels=10, **kwargs):
    """
    Plot the contour lines of the bivariate PDF of a copula using its internal parameters.

    Parameters
    ----------
    copula : object
        Instance of a copula with method get_pdf(u, v, param) and attributes `name` and `parameters`.
    Nsplit : int, optional
        Number of grid points per axis (default=50).
    levels : int or list, optional
        Contour levels (default=10 levels).
    **kwargs :
        Extra parameters passed to plt.contour.
    """
    u = np.linspace(0.01, 0.99, Nsplit)
    U, V = np.meshgrid(u, u)
    Z = np.array([copula.get_pdf(ui, vi, copula.parameters) for ui, vi in zip(U.ravel(), V.ravel())]).reshape(U.shape)

    CS = plt.contour(U, V, Z, levels=levels, **kwargs)
    plt.clabel(CS, inline=1, fontsize=8)
    plt.xlabel('u')
    plt.ylabel('v')
    plt.title(f'{copula.name} Copula PDF (contour)')
    plt.show()


def plot_conditional_contours(copula, Nsplit=50, levels=None, scatter=None, **kwargs):
    """
    Plot both conditional contours: P(U ≤ u | V = v) and P(V ≤ v | U = u) using copula's methods.

    Parameters
    ----------
    copula : object
        Instance of a copula with methods conditional_cdf_u_given_v(u,v) and conditional_cdf_v_given_u(u,v), and attribute `name`.
    Nsplit : int, optional
        Number of grid points per axis (default=50).
    levels : list or None
        Contour levels for the conditional probabilities (default [0.1,...,0.9]).
    scatter : tuple (x_points, y_points) or None
        If provided, overlay these observation points.
    **kwargs :
        Extra parameters passed to plt.contour.
    """
    u = np.linspace(0.01, 0.99, Nsplit)
    v = np.linspace(0.01, 0.99, Nsplit)
    U, V = np.meshgrid(u, v)

    # Compute conditional CDF grids directly using copula methods
    Z_uv = np.array([copula.conditional_cdf_u_given_v(ui, vi) for ui, vi in zip(U.ravel(), V.ravel())]).reshape(U.shape)
    Z_vu = np.array([copula.conditional_cdf_v_given_u(ui, vi) for ui, vi in zip(U.ravel(), V.ravel())]).reshape(U.shape)

    if levels is None:
        levels = np.linspace(0.1, 0.9, 9)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # P(U ≤ u | V = v)
    cs1 = axes[0].contour(U, V, Z_uv, levels=levels, **kwargs)
    axes[0].clabel(cs1, inline=1, fontsize=8)
    axes[0].set_xlabel('u')
    axes[0].set_ylabel('v')
    axes[0].set_title(f'P(U ≤ u | V = v) for {copula.name}')

    # P(V ≤ v | U = u)
    cs2 = axes[1].contour(U, V, Z_vu, levels=levels, **kwargs)
    axes[1].clabel(cs2, inline=1, fontsize=8)
    axes[1].set_xlabel('u')
    axes[1].set_ylabel('v')
    axes[1].set_title(f'P(V ≤ v | U = u) for {copula.name}')

    if scatter is not None:
        xs, ys = scatter
        axes[0].scatter(xs, ys, c='red', edgecolors='k')
        axes[1].scatter(xs, ys, c='red', edgecolors='k')

    plt.tight_layout()
    plt.show()
