"""Shared style, helpers and data utilities for Matplotlib-based copula diagnostics."""

from __future__ import annotations

from typing import Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt

from CopulaFurtif.core.copulas.domain.estimation.estimation import pseudo_obs


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE  — single source of truth for the whole visualisation package
# ─────────────────────────────────────────────────────────────────────────────

STYLE: dict = {
    # palette
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
    "font_title":  16,
    "font_label":  12,
    "font_tick":   10,
    "font_legend": 10,
    # layout
    "dpi":         130,
    "lw_main":     1.8,
    "lw_thin":     0.8,
}

# Canonical colormaps
CMAP_SEQ  = "plasma"      # sequential – PDF surfaces
CMAP_DIV  = "RdYlBu_r"   # diverging  – residuals / CDF


# ─────────────────────────────────────────────────────────────────────────────
# STYLE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _apply_base_style(
    ax: plt.Axes,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
) -> None:
    """Apply the dark-science theme to a 2-D Axes."""
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
                     fontweight="bold", pad=12)
    ax.grid(True, color=STYLE["grid"], linewidth=0.6, linestyle="--")


def _styled_fig(**kwargs) -> plt.Figure:
    """Create a pre-styled dark Figure."""
    kwargs.setdefault("dpi", STYLE["dpi"])
    return plt.figure(facecolor=STYLE["bg"], **kwargs)


def _styled_subplots(nrows: int = 1, ncols: int = 1, **kwargs) -> Tuple[plt.Figure, any]:
    """Wrapper around plt.subplots with the dark background applied."""
    kwargs.setdefault("dpi", STYLE["dpi"])
    fig, axes = plt.subplots(nrows, ncols, facecolor=STYLE["bg"], **kwargs)
    return fig, axes


def _styled_colorbar(
    fig: plt.Figure,
    mappable,
    ax: plt.Axes,
    label: str = "",
) -> plt.colorbar:
    cb = fig.colorbar(mappable, ax=ax, pad=0.02, fraction=0.046)
    cb.ax.yaxis.set_tick_params(color=STYLE["subtext"])
    cb.outline.set_edgecolor(STYLE["grid"])
    plt.setp(cb.ax.yaxis.get_ticklabels(),
             color=STYLE["subtext"], fontsize=STYLE["font_tick"])
    if label:
        cb.set_label(label, color=STYLE["subtext"], fontsize=STYLE["font_tick"])
    return cb


def _styled_legend(ax: plt.Axes, **kwargs) -> plt.Legend:
    defaults = dict(
        fontsize=STYLE["font_legend"],
        framealpha=0.25,
        edgecolor=STYLE["grid"],
        labelcolor=STYLE["text"],
        facecolor=STYLE["panel"],
    )
    defaults.update(kwargs)
    return ax.legend(**defaults)


def _suptitle(fig: plt.Figure, text: str) -> None:
    fig.suptitle(text, color=STYLE["text"], fontsize=STYLE["font_title"],
                 fontweight="bold")


# ─────────────────────────────────────────────────────────────────────────────
# DATA UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def to_pobs(
    x: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    y: Optional[np.ndarray] = None,
    *,
    assume_uniform: bool = False,
    eps: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert raw data to pseudo-observations (U, V) in (0, 1)."""
    if y is None:
        arr = np.asarray(x)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("Expected x to be (n, 2) if y is None")
        x, y = arr[:, 0], arr[:, 1]
    else:
        x, y = np.asarray(x), np.asarray(y)

    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have the same length")

    if assume_uniform:
        return np.clip(x, eps, 1.0 - eps), np.clip(y, eps, 1.0 - eps)

    in_unit = (
        np.nanmin(x) >= 0.0 and np.nanmax(x) <= 1.0
        and np.nanmin(y) >= 0.0 and np.nanmax(y) <= 1.0
    )
    if in_unit:
        return np.clip(x, eps, 1.0 - eps), np.clip(y, eps, 1.0 - eps)

    u, v = pseudo_obs([x, y])
    return np.clip(u, eps, 1.0 - eps), np.clip(v, eps, 1.0 - eps)


def empirical_joint_cdf(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Empirical joint CDF H_i = mean(U_j ≤ U_i and V_j ≤ V_i).  O(n²)."""
    u = np.asarray(u, float).ravel()
    v = np.asarray(v, float).ravel()
    if u.size != v.size:
        raise ValueError("u and v must have same length")
    comp = (u[None, :] <= u[:, None]) & (v[None, :] <= v[:, None])
    return comp.mean(axis=1)