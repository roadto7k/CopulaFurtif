"""Corner-style plots for copula diagnostics.

Contains:
  • Simulated scatter with marginal KDEs  (corner plot)
  • Rosenblatt PIT diagnostic in the same corner layout
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde

from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel
from CopulaFurtif.core.copulas.domain.estimation.gof import (
    rosenblatt_transform_2d,
    pit_ks_uniform,
)

from ._common import (
    STYLE, CMAP_SEQ,
    _apply_base_style, _styled_fig, _styled_legend, _suptitle,
    to_pobs,
)


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _corner_axes(fig: plt.Figure) -> Tuple[plt.Axes, plt.Axes, plt.Axes]:
    """Return (ax_joint, ax_top, ax_right) in a corner layout."""
    gs = GridSpec(
        2, 2, figure=fig,
        width_ratios=[4, 1.3],
        height_ratios=[1.3, 4],
        wspace=0.04, hspace=0.04,
        left=0.10, right=0.95, top=0.88, bottom=0.10,
    )
    ax_top   = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[1, 1])
    ax_joint = fig.add_subplot(gs[1, 0])

    # Tuck away ticks on the marginal axes
    ax_top.tick_params(axis="x", labelbottom=False, length=3)
    ax_right.tick_params(axis="y", labelleft=False, length=3)
    for ax in (ax_top, ax_right):
        ax.set_facecolor(STYLE["panel"])
        for spine in ax.spines.values():
            spine.set_edgecolor(STYLE["grid"])
        ax.tick_params(colors=STYLE["subtext"], labelsize=STYLE["font_tick"])
        ax.grid(True, color=STYLE["grid"], linewidth=0.5, linestyle="--")

    return ax_joint, ax_top, ax_right


# ─────────────────────────────────────────────────────────────────────────────
# 1. SIMULATED CORNER PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_simulated_corner_with_kdes(
    copula: CopulaModel,
    *,
    n: int = 3000,
    seed: Optional[int] = 0,
    overlay_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    overlay_assume_uniform: bool = True,
    bw_method = "scott",
    bins: int = 40,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Draw samples from a copula and display a bivariate scatter with
    marginal KDEs on each axis.  Optionally overlay observed pseudo-obs.

    Parameters
    ----------
    copula                 : CopulaModel with a .sample() method
    n                      : number of simulated points
    seed                   : RNG seed for reproducibility
    overlay_data           : (u_obs, v_obs) to overlay on the joint panel
    overlay_assume_uniform : if True, overlay arrays are treated as ∈(0,1)
    bw_method              : bandwidth for gaussian_kde
    bins                   : histogram bins for the marginal panels
    title                  : optional suptitle
    """
    rng  = np.random.default_rng(seed)
    samp = np.asarray(copula.sample(int(n), rng=rng), float)
    u_sim = np.clip(samp[:, 0], 1e-10, 1 - 1e-10)
    v_sim = np.clip(samp[:, 1], 1e-10, 1 - 1e-10)

    fig = _styled_fig(figsize=(7.5, 7.5))
    ax_joint, ax_top, ax_right = _corner_axes(fig)

    xs = np.linspace(1e-6, 1 - 1e-6, 400)

    # ── Joint scatter ─────────────────────────────────────────────────────────
    ax_joint.scatter(u_sim, v_sim, s=5, alpha=0.20,
                     color=STYLE["accent"], lw=0, rasterized=True,
                     label=f"Simulated  (n={n:,})", zorder=2)

    if overlay_data is not None:
        ou, ov = to_pobs(overlay_data[0], overlay_data[1],
                         assume_uniform=overlay_assume_uniform)
        ax_joint.scatter(ou, ov, s=9, alpha=0.45,
                         color=STYLE["warn"], edgecolors="none",
                         rasterized=True, label="Observed", zorder=3)

    _apply_base_style(ax_joint, xlabel="u", ylabel="v")
    ax_joint.set_xlim(0, 1); ax_joint.set_ylim(0, 1)
    if overlay_data is not None:
        _styled_legend(ax_joint, loc="upper left")

    # ── Top marginal: u ───────────────────────────────────────────────────────
    ku = gaussian_kde(u_sim, bw_method=bw_method)
    ax_top.hist(u_sim, bins=bins, density=True, color=STYLE["accent"],
                alpha=0.25, edgecolor="none")
    ax_top.plot(xs, ku(xs), color=STYLE["accent"], lw=STYLE["lw_main"])
    ax_top.set_xlim(0, 1)
    ax_top.set_ylabel("density", color=STYLE["subtext"],
                      fontsize=STYLE["font_tick"])

    if overlay_data is not None:
        ko_u = gaussian_kde(ou, bw_method=bw_method)
        ax_top.plot(xs, ko_u(xs), color=STYLE["warn"],
                    lw=STYLE["lw_main"], linestyle="--")

    # ── Right marginal: v ─────────────────────────────────────────────────────
    kv = gaussian_kde(v_sim, bw_method=bw_method)
    ax_right.hist(v_sim, bins=bins, density=True, color=STYLE["accent"],
                  alpha=0.25, edgecolor="none", orientation="horizontal")
    ax_right.plot(kv(xs), xs, color=STYLE["accent"], lw=STYLE["lw_main"])
    ax_right.set_ylim(0, 1)
    ax_right.set_xlabel("density", color=STYLE["subtext"],
                         fontsize=STYLE["font_tick"])

    if overlay_data is not None:
        ko_v = gaussian_kde(ov, bw_method=bw_method)
        ax_right.plot(ko_v(xs), xs, color=STYLE["warn"],
                      lw=STYLE["lw_main"], linestyle="--")

    _suptitle(fig, title or f"Simulated corner plot  –  {copula.name}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. ROSENBLATT PIT DIAGNOSTIC
# ─────────────────────────────────────────────────────────────────────────────

def plot_rosenblatt_pit_corner(
    copula: CopulaModel,
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_n: int = 500,
    seed: Optional[int] = 0,
    bins: int = 30,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Rosenblatt PIT diagnostic: if the copula fits well, the transformed
    variables (Z1, Z2) should be i.i.d. Uniform[0,1]².

    The joint scatter and both marginal histograms are shown in a corner
    layout.  A KS-test result is annotated on the joint panel.

    Parameters
    ----------
    copula : fitted CopulaModel
    x, y   : observed data (raw or already pseudo-obs)
    max_n  : maximum sample size (random subsample if n > max_n)
    seed   : RNG seed
    bins   : histogram bins for the marginal panels
    title  : optional suptitle
    """
    z1, z2 = rosenblatt_transform_2d(copula, (x, y), max_n=max_n, seed=seed)
    z1 = np.clip(z1, 1e-10, 1 - 1e-10)
    z2 = np.clip(z2, 1e-10, 1 - 1e-10)
    n  = z1.size

    D1, p1 = pit_ks_uniform(z1)
    D2, p2 = pit_ks_uniform(z2)

    # Colour-code points by distance from the diagonal (joint uniformity proxy)
    dist = np.sqrt((z1 - 0.5) ** 2 + (z2 - 0.5) ** 2)

    fig = _styled_fig(figsize=(7.5, 7.5))
    ax_joint, ax_top, ax_right = _corner_axes(fig)

    # ── Joint scatter ─────────────────────────────────────────────────────────
    sc = ax_joint.scatter(z1, z2, s=8, c=dist, cmap="plasma",
                          alpha=0.55, lw=0, rasterized=True, zorder=2)
    # Reference diagonal
    ax_joint.plot([0, 1], [0, 1], color=STYLE["subtext"],
                  lw=1.0, linestyle="--", alpha=0.5, zorder=1)

    _apply_base_style(ax_joint, xlabel="Z₁", ylabel="Z₂")
    ax_joint.set_xlim(0, 1); ax_joint.set_ylim(0, 1)

    # KS annotation
    p1_str = f"{p1:.3g}" if p1 >= 1e-3 else f"{p1:.2e}"
    p2_str = f"{p2:.3g}" if p2 >= 1e-3 else f"{p2:.2e}"
    good1  = p1 > 0.05
    good2  = p2 > 0.05
    txt = (
        f"KS(Z₁):  D={D1:.3f},  p={p1_str}  {'✓' if good1 else '✗'}\n"
        f"KS(Z₂):  D={D2:.3f},  p={p2_str}  {'✓' if good2 else '✗'}\n"
        f"n = {n:,}"
    )
    color_ks = STYLE["success"] if (good1 and good2) else STYLE["danger"]
    ax_joint.text(0.03, 0.97, txt, va="top", ha="left",
                  transform=ax_joint.transAxes,
                  color=color_ks, fontsize=9,
                  bbox=dict(facecolor=STYLE["panel"], edgecolor=STYLE["grid"],
                            boxstyle="round,pad=0.35", alpha=0.90))

    # ── Top marginal: Z1 ──────────────────────────────────────────────────────
    ax_top.hist(z1, bins=bins, density=True, color=STYLE["accent"],
                alpha=0.30, edgecolor="none")
    ax_top.axhline(1.0, color=STYLE["warn"], lw=1.2, linestyle="--",
                   label="Uniform(0,1)")
    ax_top.set_xlim(0, 1)
    ax_top.set_ylabel("density", color=STYLE["subtext"],
                      fontsize=STYLE["font_tick"])

    # ── Right marginal: Z2 ────────────────────────────────────────────────────
    ax_right.hist(z2, bins=bins, density=True, color=STYLE["accent"],
                  alpha=0.30, edgecolor="none", orientation="horizontal")
    ax_right.axvline(1.0, color=STYLE["warn"], lw=1.2, linestyle="--")
    ax_right.set_ylim(0, 1)
    ax_right.set_xlabel("density", color=STYLE["subtext"],
                         fontsize=STYLE["font_tick"])

    _suptitle(fig, title or f"Rosenblatt PIT diagnostic  –  {copula.name}")
    return fig