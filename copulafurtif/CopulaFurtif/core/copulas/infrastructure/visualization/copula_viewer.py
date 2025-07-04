import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.optimize import brentq
from enum import Enum
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel


class Plot_type(Enum):
    DIM3 = "3d"
    CONTOUR = "contour"
    
def plot_bivariate_3d(X, Y, Z, bounds, title, **kwargs):
    """
    Plot a 3D surface with flexible axis bounds.

    Parameters
    ----------
    X, Y, Z : array_like
        Meshgrid arrays of shape (N, N) representing the surface.
    bounds : tuple of length 2 or 4
        If len=2: (min, max) applies to both axes;
        if len=4: (xmin, xmax, ymin, ymax).
    title : str
        Plot title.
    **kwargs
        Keyword arguments for ax.plot_surface (e.g., cmap='viridis').
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # parse bounds
    if len(bounds) == 2:
        xmin, xmax, ymin, ymax = bounds[0], bounds[1], bounds[0], bounds[1]
    else:
        xmin, xmax, ymin, ymax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(np.linspace(xmin, xmax, 6))
    ax.set_yticks(np.linspace(ymin, ymax, 6))
    ax.plot_surface(X, Y, Z, **kwargs)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Z')
    plt.tight_layout()
    plt.show()
    
    
def plot_cdf(copula : CopulaModel, plot_type : Plot_type = Plot_type.DIM3, Nsplit=80, levels=None, cmap="coolwarm"):
    grid = np.linspace(0, 1, Nsplit)
    U, V = np.meshgrid(grid, grid, indexing="ij")
    Z = copula.get_cdf(U.ravel(), V.ravel(), copula.get_parameters()).reshape(Nsplit, Nsplit)

    if plot_type == Plot_type.DIM3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(U, V, Z, cmap=cmap)
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_zlabel("C(u,v)")
        ax.set_title("Copula CDF Surface")
        plt.tight_layout()
        
    elif plot_type == Plot_type.CONTOUR:
        plt.contourf(U, V, Z, levels=levels if levels is not None else 10, cmap=cmap)
        plt.title("Copula CDF Contours")
        plt.xlabel("u")
        plt.ylabel("v")
        plt.colorbar()
        plt.tight_layout()
    plt.show()

def plot_pdf(copula : CopulaModel, plot_type: Plot_type, Nsplit: int = 50, levels=None, log_scale: bool = False, **kwargs):
    """
    Plot the bivariate PDF of a copula with optional log-scaled contours.

    Parameters
    ----------
    copula : object
        Instance with .get_pdf(u, v, param) and .parameters.
    plot_type : {'3d', 'contour'}
    Nsplit : int
        Grid resolution.
    levels : int or array-like, optional
        For contour: number or list of levels.
    log_scale : bool
        If True, use logarithmic spacing for levels.
    **kwargs
        For '3d': passed to plot_surface;
        for 'contour': passed to contourf/contour (e.g., cmap).
    """
    # grid bounds
    if plot_type == Plot_type.DIM3:
        lo, hi = 1e-2, 1 - 1e-2
    elif plot_type == Plot_type.CONTOUR:
        lo, hi = 1e-3, 1 - 1e-3
    else:
        raise ValueError("plot_type must be '3d' or 'contour'.")

    U, V = np.meshgrid(np.linspace(lo, hi, Nsplit), np.linspace(lo, hi, Nsplit))
    Z = np.array([
        copula.get_pdf(u, v, copula.get_parameters())
        for u, v in zip(U.ravel(), V.ravel())
    ]).reshape(U.shape)
    title = f"{copula.name} Copula PDF ({plot_type})"

    if plot_type == Plot_type.DIM3:
        plot_bivariate_3d(U, V, Z, (lo, hi), title, **kwargs)
    else:
        # auto-levels if not provided or integer
        if levels is None or isinstance(levels, int):
            zmin, zmax = np.percentile(Z, 5), np.percentile(Z, 95)
            if log_scale:
                levels = np.logspace(np.log10(zmin), np.log10(zmax), levels or 10)
            else:
                levels = np.linspace(zmin, zmax, levels or 10)
        fig, ax = plt.subplots()
        cf = ax.contour(U, V, Z, levels=levels, **kwargs)
        cs = ax.contour(U, V, Z, levels=levels, colors='k', linewidths=0.5)
        ax.clabel(cs, inline=True, fontsize=8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(title)
        ax.set_xlabel('u')
        ax.set_ylabel('v')
        plt.tight_layout()
        plt.show()
        
# def plot_pdf(copula, plot_type='3d', Nsplit=80, levels=None, cmap="viridis", log_scale=False):
#     grid = np.linspace(0, 1, Nsplit)
#     U, V = np.meshgrid(grid, grid, indexing="ij")
#     Z = copula.get_pdf(U.ravel(), V.ravel(), copula.parameters).reshape(Nsplit, Nsplit)
#     if log_scale:
#         Z = np.log(Z + 1e-12)

#     if plot_type == '3d':
#         fig = plt.figure(figsize=(8, 6))
#         ax = fig.add_subplot(111, projection='3d')
#         ax.plot_surface(U, V, Z, cmap=cmap)
#         ax.set_xlabel("u")
#         ax.set_ylabel("v")
#         ax.set_zlabel("c(u,v)")
#         ax.set_title("Copula PDF Surface")
#         plt.tight_layout()
#     elif plot_type == 'contour':
#         plt.contourf(U, V, Z, levels=levels if levels is not None else 10, cmap=cmap)
#         plt.title("Copula PDF Contours")
#         plt.xlabel("u")
#         plt.ylabel("v")
#         plt.colorbar()
#         plt.tight_layout()
#     plt.show()

def plot_bivariate_contour(X, Y, Z, bounds, title, levels=10, **kwargs):
    """
    Plot contour lines only (no fill), with flexible axis bounds.

    Parameters
    ----------
    X, Y, Z : array_like
        Meshgrid arrays of shape (N, N).
    bounds : tuple of length 2 or 4
        If len=2: (min, max) applies to both axes;
        if len=4: (xmin, xmax, ymin, ymax).
    title : str
        Plot title.
    levels : int or array-like
        Number of contour levels or specific list of levels.
    **kwargs
        Additional keyword args for plt.contour (e.g., cmap, linestyles).
    """
    fig, ax = plt.subplots()
    # choose color argument: if cmap provided, let user define colors, else use black lines
    contour_kwargs = kwargs.copy()
    if 'cmap' in contour_kwargs and 'colors' not in contour_kwargs:
        cs = ax.contour(X, Y, Z, levels=levels, **contour_kwargs)
    else:
        cs = ax.contour(X, Y, Z, levels=levels, colors='k', linewidths=1.0, **contour_kwargs)
    ax.clabel(cs, inline=True, fontsize=8)
    # set bounds
    if len(bounds) == 2:
        xmin, xmax, ymin, ymax = bounds[0], bounds[1], bounds[0], bounds[1]
    else:
        xmin, xmax, ymin, ymax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.tight_layout()
    plt.show()


def plot_mpdf(copula : CopulaModel, margins, plot_type: Plot_type, Nsplit: int = 50, bounds=None, **kwargs):
    """
    Plot the joint PDF with specified marginal distributions.

    If bounds is None, compute (1%%,99%%) percentiles for each margin.
    """
    # unpack margins
    m1, m2 = margins
    dist1, loc1, scale1 = m1['distribution'], m1['loc'], m1['scale']
    dist2, loc2, scale2 = m2['distribution'], m2['loc'], m2['scale']

    # determine original-space bounds
    if bounds is None:
        x_min, x_max = dist1.ppf([0.01, 0.99], loc=loc1, scale=scale1)
        y_min, y_max = dist2.ppf([0.01, 0.99], loc=loc2, scale=scale2)
    else:
        x_min, x_max, y_min, y_max = bounds

    x = np.linspace(x_min, x_max, Nsplit)
    y = np.linspace(y_min, y_max, Nsplit)
    X, Y = np.meshgrid(x, y)

    # map to copula space and compute joint PDF
    U = dist1.cdf(X, loc=loc1, scale=scale1)
    V = dist2.cdf(Y, loc=loc2, scale=scale2)
    Zc = np.array([
        copula.get_pdf(u, v, copula.get_parameters())
        for u, v in zip(U.ravel(), V.ravel())
    ]).reshape(U.shape)
    f1 = dist1.pdf(X, loc=loc1, scale=scale1)
    f2 = dist2.pdf(Y, loc=loc2, scale=scale2)
    Z = Zc * f1 * f2

    title = f"{copula.name} joint PDF ({plot_type})"
    if plot_type == Plot_type.DIM3:
        plot_bivariate_3d(X, Y, Z, (x_min, x_max, y_min, y_max), title, **kwargs)
    else:
        plot_bivariate_contour(X, Y, Z, (x_min, x_max, y_min, y_max), title, **kwargs)




def plot_arbitrage_frontiers(
    copula : CopulaModel,
    alpha_low: float = 0.05,
    alpha_high: float = 0.95,
    levels: int = 200,
    scatter: tuple = None,
    scatter_alpha: float = 0.3,
):
    """
    Plot conditional quantile frontiers for arbitrage detection.
    """
    # style
    plt.rcParams.update({
        'figure.figsize': (7, 7),
        'axes.facecolor': '#f7f7f7',
        'axes.grid': True,
        'grid.color': '#dddddd'
    })
    u_min, u_max = 1e-6, 1 - 1e-6
    v_grid = np.linspace(u_min, u_max, levels)
    Q_low, Q_high = [], []
    for v in v_grid:
        c_min = copula.conditional_cdf_u_given_v(u_min, v)
        c_max = copula.conditional_cdf_u_given_v(u_max, v)
        u_l = u_min if c_min >= alpha_low else (u_max if c_max <= alpha_low else brentq(lambda u: copula.conditional_cdf_u_given_v(u, v) - alpha_low, u_min, u_max))
        u_h = u_min if c_min >= alpha_high else (u_max if c_max <= alpha_high else brentq(lambda u: copula.conditional_cdf_u_given_v(u, v) - alpha_high, u_min, u_max))
        Q_low.append(u_l); Q_high.append(u_h)

    fig, ax = plt.subplots()
    if scatter is not None:
        u_s, v_s = scatter
        ax.scatter(u_s, v_s, c='gray', alpha=scatter_alpha, s=40, marker='o', label='Data')
        p_uv = np.array([copula.conditional_cdf_u_given_v(u, v) for u, v in zip(u_s, v_s)])
        ax.scatter(u_s[p_uv < alpha_low], v_s[p_uv < alpha_low], c='#2a9d8f', edgecolors='k', s=60, label=f'p < {alpha_low:.2f}')
        ax.scatter(u_s[p_uv > alpha_high], v_s[p_uv > alpha_high], c='#e76f51', edgecolors='k', s=60, label=f'p > {alpha_high:.2f}')
    ax.plot(Q_low, v_grid, ls='-', color='#264653', lw=2.5, label=f'Low frontier ({alpha_low:.2f})')
    ax.plot(Q_high, v_grid, ls='-', color='#e76f51', lw=2.5, label=f'High frontier ({alpha_high:.2f})')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('u'); ax.set_ylabel('v')
    ax.set_title('Arbitrage Frontiers', pad=20)
    ax.legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    plt.show()
