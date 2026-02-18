import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.optimize import brentq
from enum import Enum
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────────────────────────────────────

STYLE = {
    # palette – deep navy background, warm cream text, vivid accent
    "bg":          "#0d1117",
    "panel":       "#161b22",
    "grid":        "#21262d",
    "text":        "#e6edf3",
    "subtext":     "#8b949e",
    "accent":      "#58a6ff",
    "danger":      "#f85149",
    "success":     "#3fb950",
    "warn":        "#d29922",
    # typography
    "font_title":  18,
    "font_label":  12,
    "font_tick":   10,
    "font_legend": 11,
    # layout
    "figsize_3d":  (10, 7),
    "figsize_2d":  (8, 7),
    "dpi":         130,
    "lw_main":     2.2,
    "lw_contour":  0.8,
}

# Scientific sequential / diverging palettes
CMAP_SEQ  = "plasma"       # for PDF surfaces
CMAP_DIV  = "RdYlBu_r"    # for CDF
CMAP_COOL = "coolwarm"


def _apply_base_style(ax, title: str = "", xlabel: str = "", ylabel: str = "",
                      zlabel: str = "", is_3d: bool = False):
    """Apply the dark-science theme to any Axes."""
    ax.set_facecolor(STYLE["panel"])
    for spine in ax.spines.values():
        spine.set_edgecolor(STYLE["grid"])

    ax.tick_params(colors=STYLE["subtext"], labelsize=STYLE["font_tick"],
                   which="both", length=4)
    ax.xaxis.label.set_color(STYLE["text"])
    ax.yaxis.label.set_color(STYLE["text"])
    ax.set_xlabel(xlabel, fontsize=STYLE["font_label"], labelpad=8)
    ax.set_ylabel(ylabel, fontsize=STYLE["font_label"], labelpad=8)

    if title:
        ax.set_title(title, fontsize=STYLE["font_title"], color=STYLE["text"],
                     fontweight="bold", pad=14)

    ax.grid(True, color=STYLE["grid"], linewidth=0.6, linestyle="--")

    if is_3d:
        ax.set_zlabel(zlabel, fontsize=STYLE["font_label"], labelpad=8)
        ax.zaxis.label.set_color(STYLE["text"])
        ax.tick_params(axis="z", colors=STYLE["subtext"], labelsize=STYLE["font_tick"])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor(STYLE["grid"])
        ax.yaxis.pane.set_edgecolor(STYLE["grid"])
        ax.zaxis.pane.set_edgecolor(STYLE["grid"])


def _styled_figure(figsize=None, is_3d: bool = False):
    """Create a pre-styled dark figure (and 3-D axes if requested)."""
    figsize = figsize or (STYLE["figsize_3d"] if is_3d else STYLE["figsize_2d"])
    fig = plt.figure(figsize=figsize, dpi=STYLE["dpi"],
                     facecolor=STYLE["bg"])
    if is_3d:
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(STYLE["panel"])
    else:
        ax = fig.add_subplot(111)
        ax.set_facecolor(STYLE["panel"])
    return fig, ax


def _styled_colorbar(fig, mappable, ax, label: str = ""):
    """Attach a styled colour bar."""
    cb = fig.colorbar(mappable, ax=ax, pad=0.02, fraction=0.046)
    cb.ax.yaxis.set_tick_params(color=STYLE["subtext"])
    cb.outline.set_edgecolor(STYLE["grid"])
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=STYLE["subtext"],
             fontsize=STYLE["font_tick"])
    if label:
        cb.set_label(label, color=STYLE["subtext"], fontsize=STYLE["font_tick"])
    return cb


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class Plot_type(Enum):
    DIM3    = "3d"
    CONTOUR = "contour"


# ─────────────────────────────────────────────────────────────────────────────
# LOW-LEVEL SURFACE / CONTOUR HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def plot_bivariate_3d(X, Y, Z, bounds, title, cmap=CMAP_SEQ, **kwargs):
    """
    Plot a high-quality 3-D surface.

    Parameters
    ----------
    X, Y, Z : array-like of shape (N, N)
    bounds   : (min, max) or (xmin, xmax, ymin, ymax)
    title    : str
    cmap     : str or Colormap
    **kwargs : extra kwargs forwarded to plot_surface
    """
    fig, ax = _styled_figure(is_3d=True)

    if len(bounds) == 2:
        xmin, xmax, ymin, ymax = bounds[0], bounds[1], bounds[0], bounds[1]
    else:
        xmin, xmax, ymin, ymax = bounds

    surf = ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0,
                           antialiased=True, alpha=0.92, **kwargs)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(np.linspace(xmin, xmax, 5))
    ax.set_yticks(np.linspace(ymin, ymax, 5))
    _apply_base_style(ax, title=title, xlabel="x", ylabel="y", zlabel="Z", is_3d=True)

    # Add a thin contour projection on the floor for depth
    zfloor = Z.min() - 0.05 * (Z.max() - Z.min())
    ax.contourf(X, Y, Z, zdir="z", offset=zfloor, cmap=cmap, alpha=0.4, levels=15)
    ax.set_zlim(zfloor, Z.max())

    _styled_colorbar(fig, surf, ax)
    fig.tight_layout()
    plt.show()


def plot_bivariate_contour(X, Y, Z, bounds, title, levels=12, cmap=CMAP_SEQ, **kwargs):
    """
    Plot filled + line contours with a clean dark theme.

    Parameters
    ----------
    X, Y, Z : array-like of shape (N, N)
    bounds   : (min, max) or (xmin, xmax, ymin, ymax)
    title    : str
    levels   : int or array-like
    cmap     : str or Colormap
    **kwargs : forwarded to contour
    """
    fig, ax = _styled_figure()

    if len(bounds) == 2:
        xmin, xmax, ymin, ymax = bounds[0], bounds[1], bounds[0], bounds[1]
    else:
        xmin, xmax, ymin, ymax = bounds

    cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=0.85, **kwargs)
    cs = ax.contour(X, Y, Z,  levels=levels, colors="white",
                    linewidths=STYLE["lw_contour"], alpha=0.45)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f",
              colors=STYLE["subtext"])

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    _apply_base_style(ax, title=title, xlabel="x", ylabel="y")
    _styled_colorbar(fig, cf, ax)
    fig.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CDF
# ─────────────────────────────────────────────────────────────────────────────

def plot_cdf(copula: CopulaModel,
             plot_type: Plot_type = Plot_type.DIM3,
             Nsplit: int = 80,
             levels: int = 15,
             cmap: str = CMAP_DIV):
    """
    Plot the copula CDF either as a 3-D surface or filled contour map.

    Parameters
    ----------
    copula    : CopulaModel instance
    plot_type : Plot_type.DIM3 | Plot_type.CONTOUR
    Nsplit    : grid resolution
    levels    : number of contour / colour levels
    cmap      : matplotlib colormap name
    """
    eps = 1e-3
    grid = np.linspace(eps, 1 - eps, Nsplit)
    U, V = np.meshgrid(grid, grid, indexing="ij")
    Z = copula.get_cdf(U.ravel(), V.ravel(),
                       copula.get_parameters()).reshape(Nsplit, Nsplit)
    title = f"{copula.name}  –  Copula CDF"

    if plot_type == Plot_type.DIM3:
        fig, ax = _styled_figure(is_3d=True)
        surf = ax.plot_surface(U, V, Z, cmap=cmap, linewidth=0,
                               antialiased=True, alpha=0.92)
        # floor projection
        ax.contourf(U, V, Z, zdir="z", offset=-0.05, cmap=cmap,
                    alpha=0.35, levels=levels)
        ax.set_zlim(-0.05, 1.0)
        _apply_base_style(ax, title=title, xlabel="u", ylabel="v",
                          zlabel="C(u, v)", is_3d=True)
        _styled_colorbar(fig, surf, ax, label="C(u, v)")

    else:
        fig, ax = _styled_figure()
        cf = ax.contourf(U, V, Z, levels=levels, cmap=cmap, alpha=0.88)
        cs = ax.contour(U, V, Z,  levels=levels, colors="white",
                        linewidths=STYLE["lw_contour"], alpha=0.4)
        ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f",
                  colors=STYLE["subtext"])
        _apply_base_style(ax, title=title, xlabel="u", ylabel="v")
        _styled_colorbar(fig, cf, ax, label="C(u, v)")

    fig.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# PDF
# ─────────────────────────────────────────────────────────────────────────────

def plot_pdf(copula: CopulaModel,
             plot_type: Plot_type,
             Nsplit: int = 60,
             levels: int = 15,
             log_scale: bool = False,
             cmap: str = CMAP_SEQ,
             **kwargs):
    """
    Plot the bivariate copula PDF with optional log-scale for heavy tails.

    Parameters
    ----------
    copula    : CopulaModel instance
    plot_type : Plot_type.DIM3 | Plot_type.CONTOUR
    Nsplit    : grid resolution
    levels    : contour / colour levels (int or array-like)
    log_scale : if True, colour-map is applied on log(pdf)
    cmap      : matplotlib colormap
    **kwargs  : forwarded to the underlying plot call
    """
    lo = 1e-2 if plot_type == Plot_type.DIM3 else 1e-3
    hi = 1 - lo

    grid = np.linspace(lo, hi, Nsplit)
    U, V = np.meshgrid(grid, grid)
    Z = np.array([copula.get_pdf(u, v)
                  for u, v in zip(U.ravel(), V.ravel())]).reshape(U.shape)

    Z_plot = np.log1p(Z) if log_scale else Z
    scale_tag = "  [log scale]" if log_scale else ""
    title = f"{copula.name}  –  Copula PDF{scale_tag}"

    if plot_type == Plot_type.DIM3:
        plot_bivariate_3d(U, V, Z_plot, (lo, hi), title, cmap=cmap, **kwargs)
    else:
        # auto levels from central quantiles to avoid extreme tail spikes
        if isinstance(levels, int):
            zmin = np.percentile(Z_plot, 2)
            zmax = np.percentile(Z_plot, 98)
            if log_scale:
                levels = np.logspace(np.log10(max(zmin, 1e-6)),
                                     np.log10(zmax), levels)
            else:
                levels = np.linspace(zmin, zmax, levels)

        plot_bivariate_contour(U, V, Z_plot, (lo, hi), title,
                               levels=levels, cmap=cmap, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# MARGINAL PDF
# ─────────────────────────────────────────────────────────────────────────────

def plot_mpdf(copula: CopulaModel,
              margins,
              plot_type: Plot_type,
              Nsplit: int = 60,
              bounds=None,
              cmap: str = CMAP_SEQ,
              **kwargs):
    """
    Plot the joint PDF in the original (X, Y) space using specified marginals.

    Parameters
    ----------
    copula    : CopulaModel instance
    margins   : list of two dicts, each with keys 'distribution', 'loc', 'scale'
    plot_type : Plot_type.DIM3 | Plot_type.CONTOUR
    Nsplit    : grid resolution
    bounds    : (xmin, xmax, ymin, ymax) or None  →  auto from 1 %–99 % quantiles
    cmap      : colormap
    **kwargs  : forwarded to the surface / contour helper
    """
    m1, m2 = margins
    dist1, loc1, scale1 = m1["distribution"], m1["loc"], m1["scale"]
    dist2, loc2, scale2 = m2["distribution"], m2["loc"], m2["scale"]

    if bounds is None:
        x_min, x_max = dist1.ppf([0.01, 0.99], loc=loc1, scale=scale1)
        y_min, y_max = dist2.ppf([0.01, 0.99], loc=loc2, scale=scale2)
    else:
        x_min, x_max, y_min, y_max = bounds

    X, Y = np.meshgrid(np.linspace(x_min, x_max, Nsplit),
                       np.linspace(y_min, y_max, Nsplit))
    U = dist1.cdf(X, loc=loc1, scale=scale1)
    V = dist2.cdf(Y, loc=loc2, scale=scale2)

    Zc = np.array([copula.get_pdf(u, v)
                   for u, v in zip(U.ravel(), V.ravel())]).reshape(U.shape)
    Z  = Zc * dist1.pdf(X, loc=loc1, scale=scale1) \
             * dist2.pdf(Y, loc=loc2, scale=scale2)

    title = f"{copula.name}  –  Joint PDF"
    bounds_full = (x_min, x_max, y_min, y_max)

    if plot_type == Plot_type.DIM3:
        plot_bivariate_3d(X, Y, Z, bounds_full, title, cmap=cmap, **kwargs)
    else:
        plot_bivariate_contour(X, Y, Z, bounds_full, title, cmap=cmap, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# ARBITRAGE FRONTIERS
# ─────────────────────────────────────────────────────────────────────────────

def plot_arbitrage_frontiers(
    copula: CopulaModel,
    alpha_low:    float = 0.05,
    alpha_high:   float = 0.95,
    levels:       int   = 300,
    scatter:      tuple = None,
    scatter_alpha: float = 0.4,
    show_density_bg: bool = True,
    Nbg:          int   = 40,
):
    """
    Plot conditional quantile frontiers for arbitrage / tail-risk detection.

    Parameters
    ----------
    copula          : CopulaModel instance
    alpha_low       : lower conditional quantile threshold
    alpha_high      : upper conditional quantile threshold
    levels          : number of v-grid points used to trace each frontier
    scatter         : (u_array, v_array) of observed pseudo-observations; optional
    scatter_alpha   : opacity of background scatter dots
    show_density_bg : if True, draws a faint PDF heatmap background
    Nbg             : grid resolution for the background density (if shown)
    """
    # ── background PDF heatmap ────────────────────────────────────────────────
    fig, ax = _styled_figure()

    if show_density_bg:
        lo_bg, hi_bg = 1e-2, 1 - 1e-2
        gb = np.linspace(lo_bg, hi_bg, Nbg)
        Ubg, Vbg = np.meshgrid(gb, gb)
        Zbg = np.array([copula.get_pdf(u, v)
                        for u, v in zip(Ubg.ravel(), Vbg.ravel())]
                       ).reshape(Ubg.shape)
        Zbg_log = np.log1p(Zbg)
        ax.contourf(Ubg, Vbg, Zbg_log, levels=20, cmap="Blues",
                    alpha=0.22, zorder=0)

    # ── frontier computation ──────────────────────────────────────────────────
    u_eps  = 1e-6
    u_max  = 1 - u_eps
    v_grid = np.linspace(u_eps, u_max, levels)

    Q_low, Q_high = [], []
    for v in v_grid:
        def _cond(u):
            return copula.conditional_cdf_u_given_v(u, v)
        c_min, c_max = _cond(u_eps), _cond(u_max)

        u_l = (u_eps if c_min >= alpha_low
               else u_max if c_max <= alpha_low
               else brentq(lambda u: _cond(u) - alpha_low, u_eps, u_max,
                           xtol=1e-8))
        u_h = (u_eps if c_min >= alpha_high
               else u_max if c_max <= alpha_high
               else brentq(lambda u: _cond(u) - alpha_high, u_eps, u_max,
                           xtol=1e-8))
        Q_low.append(u_l)
        Q_high.append(u_h)

    Q_low  = np.array(Q_low)
    Q_high = np.array(Q_high)

    # ── scatter overlay ───────────────────────────────────────────────────────
    if scatter is not None:
        u_s, v_s = np.asarray(scatter[0]), np.asarray(scatter[1])
        p_uv = np.array([copula.conditional_cdf_u_given_v(u, v)
                         for u, v in zip(u_s, v_s)])

        mask_low  = p_uv < alpha_low
        mask_high = p_uv > alpha_high
        mask_mid  = ~mask_low & ~mask_high

        ax.scatter(u_s[mask_mid],  v_s[mask_mid],  c=STYLE["subtext"],
                   alpha=scatter_alpha, s=22, lw=0, zorder=2, label="In-band")
        ax.scatter(u_s[mask_low],  v_s[mask_low],  c=STYLE["success"],
                   edgecolors="white", linewidths=0.5, s=55, zorder=4,
                   label=f"p < {alpha_low:.2f}  (buy signal)")
        ax.scatter(u_s[mask_high], v_s[mask_high], c=STYLE["danger"],
                   edgecolors="white", linewidths=0.5, s=55, zorder=4,
                   label=f"p > {alpha_high:.2f}  (sell signal)")

    # ── frontier bands (shaded between lines) ────────────────────────────────
    ax.fill_betweenx(v_grid, 0, Q_low,  color=STYLE["success"],
                     alpha=0.10, zorder=1)
    ax.fill_betweenx(v_grid, Q_high, 1, color=STYLE["danger"],
                     alpha=0.10, zorder=1)

    ax.plot(Q_low,  v_grid, lw=STYLE["lw_main"], color=STYLE["success"],
            zorder=3, label=f"Lower frontier  α={alpha_low:.2f}")
    ax.plot(Q_high, v_grid, lw=STYLE["lw_main"], color=STYLE["danger"],
            zorder=3, label=f"Upper frontier  α={alpha_high:.2f}")

    # ── annotations ──────────────────────────────────────────────────────────
    ax.annotate(f"α = {alpha_low:.2f}", xy=(Q_low[levels // 2], v_grid[levels // 2]),
                xytext=(Q_low[levels // 2] - 0.12, v_grid[levels // 2] + 0.05),
                color=STYLE["success"], fontsize=9,
                arrowprops=dict(arrowstyle="->", color=STYLE["success"], lw=1))
    ax.annotate(f"α = {alpha_high:.2f}", xy=(Q_high[levels // 2], v_grid[levels // 2]),
                xytext=(Q_high[levels // 2] + 0.05, v_grid[levels // 2] - 0.08),
                color=STYLE["danger"], fontsize=9,
                arrowprops=dict(arrowstyle="->", color=STYLE["danger"], lw=1))

    # ── styling ───────────────────────────────────────────────────────────────
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    _apply_base_style(ax,
                      title=f"{copula.name}  –  Arbitrage Frontiers",
                      xlabel="u  (rank-transformed asset A)",
                      ylabel="v  (rank-transformed asset B)")

    legend = ax.legend(loc="lower right",
                       fontsize=STYLE["font_legend"],
                       framealpha=0.25,
                       edgecolor=STYLE["grid"],
                       labelcolor=STYLE["text"],
                       facecolor=STYLE["panel"])

    fig.tight_layout()
    plt.show()