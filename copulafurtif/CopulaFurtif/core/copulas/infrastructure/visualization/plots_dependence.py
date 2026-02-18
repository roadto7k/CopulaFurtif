"""Dependence diagnostics plots.

Contains:
  • Tail concentration curves  L(t) / U(t)
  • Kendall K-plot
  • Chi-plot
  • Pickands dependence function  A(ω)
  • Conditional simulation fan
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgrid
from scipy.optimize import brentq
from scipy.stats import rankdata

from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel

from ._common import (
    STYLE,
    _apply_base_style, _styled_fig, _styled_subplots, _styled_legend, _suptitle,
    to_pobs, empirical_joint_cdf,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. TAIL CONCENTRATION CURVES
# ─────────────────────────────────────────────────────────────────────────────

def plot_tail_concentration_curves(
    copula: CopulaModel,
    x: np.ndarray,
    y: np.ndarray,
    *,
    assume_uniform: bool = False,
    t_min: float = 0.01,
    t_max: float = 0.30,
    n_t: int = 60,
    normalize_by_t: bool = True,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot empirical vs theoretical tail concentration functions L(t) and U(t).
    Richer than a single scalar λ_L/λ_U — the shape of the curve reveals
    whether tail dependence is present and how quickly it decays.

    Parameters
    ----------
    copula         : fitted CopulaModel
    x, y           : data arrays
    assume_uniform : treat x/y as already in (0,1)
    t_min, t_max   : threshold range
    n_t            : number of threshold points
    normalize_by_t : divide by t  →  convergence to λ_L / λ_U as t→0
    title          : optional suptitle
    """
    u, v   = to_pobs(x, y, assume_uniform=assume_uniform)
    t_vals = np.linspace(float(t_min), float(t_max), int(n_t))

    # Empirical
    ll_emp = np.array([np.mean((u <= t) & (v <= t)) for t in t_vals], dtype=float)
    ur_emp = np.array([np.mean((u >= 1 - t) & (v >= 1 - t)) for t in t_vals], dtype=float)

    # Theoretical
    ll_theo = np.array([float(copula.get_cdf(t, t)) for t in t_vals], dtype=float)
    ur_theo = np.array([
        1 - 2 * (1 - t) + float(copula.get_cdf(1 - t, 1 - t))
        for t in t_vals
    ], dtype=float)

    if normalize_by_t:
        safe_t  = np.maximum(t_vals, 1e-12)
        ll_emp  /= safe_t; ll_theo /= safe_t
        ur_emp  /= safe_t; ur_theo /= safe_t
        y_label  = "L(t) / t"
    else:
        y_label  = "concentration"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                   facecolor=STYLE["bg"], dpi=STYLE["dpi"])

    def _panel(ax, emp, theo, corner_label):
        ax.fill_between(t_vals, emp, theo,
                        color=STYLE["warn"], alpha=0.15,
                        label="Emp − Theo gap")
        ax.plot(t_vals, emp,  color=STYLE["accent"],  lw=STYLE["lw_main"],
                label="Empirical")
        ax.plot(t_vals, theo, color=STYLE["danger"],   lw=STYLE["lw_main"],
                linestyle="--", label="Theoretical")
        _apply_base_style(ax, title=corner_label, xlabel="t", ylabel=y_label)
        _styled_legend(ax)

    _panel(ax1, ll_emp, ll_theo, "Lower-left concentration  L(t)")
    _panel(ax2, ur_emp, ur_theo, "Upper-right concentration  U(t)")

    _suptitle(fig, title or f"Tail concentration curves  –  {copula.name}")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. KENDALL K-PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_kendall_k_plot(
    x: np.ndarray,
    y: np.ndarray,
    *,
    assume_uniform: bool = False,
    max_n: int = 600,
    seed: Optional[int] = 0,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Kendall K-plot: H_{i:n} (empirical joint CDF order stats) vs W_{i:n}
    (expected order stats under independence).  Deviation from the diagonal
    reveals the sign and strength of dependence.

    Parameters
    ----------
    x, y           : data arrays
    assume_uniform : treat x/y as already in (0,1)
    max_n          : random sub-sample cap (O(n²) algorithm)
    seed           : RNG seed for sub-sampling
    title          : optional title
    """
    u, v = to_pobs(x, y, assume_uniform=assume_uniform)
    n    = u.size
    if n > max_n:
        rng  = np.random.default_rng(seed)
        idx  = rng.choice(n, size=max_n, replace=False)
        u, v = u[idx], v[idx]
        n    = u.size

    H  = empirical_joint_cdf(u, v)
    Hs = np.sort(H)

    def K0(w: float) -> float:
        return w - w * np.log(max(w, 1e-300))

    ps = (np.arange(1, n + 1) - 0.5) / n
    W  = np.array([brentq(lambda w: K0(w) - p, 1e-12, 1 - 1e-12, maxiter=200)
                   for p in ps])

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.5),
                           facecolor=STYLE["bg"], dpi=STYLE["dpi"])

    # Colour points by signed deviation from independence
    dev = Hs - W
    sc  = ax.scatter(W, Hs, s=18, c=dev, cmap="RdYlBu_r",
                     alpha=0.70, lw=0, rasterized=True, zorder=3)

    # Independence diagonal
    ax.plot([0, 1], [0, 1], color=STYLE["subtext"], lw=1.2,
            linestyle="--", label="Independence", zorder=2)

    # Shaded deviation band
    ax.fill_between(W, W, Hs,
                    where=Hs > W, color=STYLE["accent"], alpha=0.10, zorder=1,
                    label="Positive dependence")
    ax.fill_between(W, W, Hs,
                    where=Hs < W, color=STYLE["danger"],  alpha=0.10, zorder=1,
                    label="Negative dependence")

    _apply_base_style(ax,
                      title=f"Kendall K-plot  (n = {n:,})",
                      xlabel="W  (independence reference)",
                      ylabel="H  (empirical joint CDF)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    _styled_legend(ax, loc="upper left")

    if title:
        ax.set_title(title, fontsize=STYLE["font_title"],
                     color=STYLE["text"], fontweight="bold", pad=12)

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. CHI-PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_chi_plot(
    x: np.ndarray,
    y: np.ndarray,
    *,
    assume_uniform: bool = False,
    mode: str = "NULL",
    max_n: int = 800,
    seed: Optional[int] = 0,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Chi-plot: χ_i vs λ_i for each pair of observations.
    Points near zero indicate independence; systematic deviation reveals the
    sign and strength of dependence.

    Parameters
    ----------
    x, y           : data arrays
    assume_uniform : treat x/y as already in (0,1)
    mode           : 'NULL' (all), 'lower', or 'upper' quadrant filter
    max_n          : random sub-sample cap
    seed           : RNG seed
    title          : optional title
    """
    u, v = to_pobs(x, y, assume_uniform=assume_uniform)
    n    = u.size
    if n > max_n:
        rng  = np.random.default_rng(seed)
        idx  = rng.choice(n, size=max_n, replace=False)
        u, v = u[idx], v[idx]
        n    = u.size

    F1 = rankdata(u, method="average") / n
    F2 = rankdata(v, method="average") / n
    H  = empirical_joint_cdf(u, v)

    den = np.sqrt(np.maximum(F1 * (1 - F1) * F2 * (1 - F2), 1e-16))
    chi = (H - F1 * F2) / den

    Ft1 = F1 - 0.5
    Ft2 = F2 - 0.5
    lam = 4.0 * np.sign(Ft1 * Ft2) * np.maximum(Ft1 ** 2, Ft2 ** 2)

    if mode.lower() == "lower":
        m   = (u <= np.mean(u)) & (v <= np.mean(v))
        lam, chi = lam[m], chi[m]
    elif mode.lower() == "upper":
        m   = (u >= np.mean(u)) & (v >= np.mean(v))
        lam, chi = lam[m], chi[m]

    bound = 1.54 / np.sqrt(max(n, 1))

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 5.2),
                           facecolor=STYLE["bg"], dpi=STYLE["dpi"])

    # Colour by χ value
    sc = ax.scatter(lam, chi, s=12, c=chi, cmap="RdYlBu_r",
                    vmin=-1, vmax=1, alpha=0.55, lw=0,
                    rasterized=True, zorder=3)

    # Independence band
    ax.axhline( bound, color=STYLE["warn"], lw=1.2, linestyle="--",
                label=f"±{bound:.3f}  (independence band)")
    ax.axhline(-bound, color=STYLE["warn"], lw=1.2, linestyle="--")
    ax.fill_between([-1, 1], -bound, bound,
                    color=STYLE["warn"], alpha=0.06, zorder=1)
    ax.axhline(0, color=STYLE["subtext"], lw=0.8, linestyle=":")

    _apply_base_style(ax,
                      title=f"Chi-plot  (mode = {mode},  n = {n:,})",
                      xlabel="λ_i", ylabel="χ_i")
    ax.set_xlim(-1, 1)
    _styled_legend(ax)

    if title:
        ax.set_title(title, fontsize=STYLE["font_title"],
                     color=STYLE["text"], fontweight="bold", pad=12)

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. PICKANDS DEPENDENCE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def plot_pickands_dependence_function(
    copula: CopulaModel,
    x: np.ndarray,
    y: np.ndarray,
    *,
    assume_uniform: bool = False,
    t: float = 1.0,
    n_w: int = 101,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Pickands dependence function A(ω) — mandatory for extreme-value copulas.
    Empirical and theoretical curves are plotted alongside the two canonical
    bounds: independence A(ω)=1 and comonotonicity A(ω)=max(ω,1−ω).

    Parameters
    ----------
    copula         : fitted CopulaModel
    x, y           : data arrays
    assume_uniform : treat x/y as already in (0,1)
    t              : Pickands parameter (default 1.0)
    n_w            : number of ω grid points
    title          : optional title
    """
    u, v = to_pobs(x, y, assume_uniform=assume_uniform)
    w    = np.linspace(1e-3, 1 - 1e-3, int(n_w))
    t    = float(t)

    uu = np.exp(-w * t)
    vv = np.exp(-(1 - w) * t)

    c_theo = np.array([float(copula.get_cdf(a, b))
                        for a, b in zip(uu, vv)], dtype=float)
    A_theo = -np.log(np.maximum(c_theo, 1e-16)) / t

    c_emp = np.array([np.mean((u <= a) & (v <= b))
                      for a, b in zip(uu, vv)], dtype=float)
    A_emp = -np.log(np.maximum(c_emp, 1e-16)) / t

    A_comono = np.maximum(w, 1 - w)
    A_indep  = np.ones_like(w)

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.2),
                           facecolor=STYLE["bg"], dpi=STYLE["dpi"])

    # Fill feasible region between the two bounds
    ax.fill_between(w, A_comono, A_indep,
                    color=STYLE["accent"], alpha=0.06, zorder=1,
                    label="Feasible region")

    ax.plot(w, A_emp,    color=STYLE["accent"],  lw=STYLE["lw_main"],
            label="Empirical A(ω)", zorder=4)
    ax.plot(w, A_theo,   color=STYLE["danger"],   lw=STYLE["lw_main"],
            linestyle="--", label="Theoretical A(ω)", zorder=4)
    ax.plot(w, A_comono, color=STYLE["success"],  lw=1.2,
            linestyle=":",  label="Comonotonicity  max(ω, 1−ω)", zorder=3)
    ax.plot(w, A_indep,  color=STYLE["subtext"],  lw=1.0,
            linestyle=":",  label="Independence  A = 1", zorder=3)

    _apply_base_style(ax,
                      title=f"Pickands dependence function  –  {copula.name}",
                      xlabel="ω", ylabel="A(ω)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.45, 1.05)
    _styled_legend(ax, loc="upper center", ncol=2)

    if title:
        ax.set_title(title, fontsize=STYLE["font_title"],
                     color=STYLE["text"], fontweight="bold", pad=12)

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. CONDITIONAL SIMULATION FAN
# ─────────────────────────────────────────────────────────────────────────────

def plot_conditional_simulation_fan(
    copula: CopulaModel,
    *,
    alphas: Iterable[float] = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95),
    n_grid: int = 200,
    eps: float = 1e-6,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Fan chart of conditional quantile curves  Q(α | v)  for multiple α levels.
    Generalises the two-line arbitrage frontier to a full distributional view.

    Parameters
    ----------
    copula : fitted CopulaModel with .conditional_cdf_v_given_u()
    alphas : iterable of α levels to trace  (sorted low→high internally)
    n_grid : v-axis resolution
    eps    : boundary guard
    title  : optional title
    """
    params = copula.get_parameters()
    alphas = sorted(float(a) for a in alphas)
    n_a    = len(alphas)
    v_grid = np.linspace(eps, 1 - eps, int(n_grid))

    def _solve(v: float, a: float) -> float:
        try:
            f = lambda u: float(
                copula.conditional_cdf_v_given_u(u, v, param=params)
            ) - a
            return float(brentq(f, eps, 1 - eps, maxiter=200))
        except Exception:
            return np.nan

    curves = [
        np.array([_solve(float(v), a) for v in v_grid])
        for a in alphas
    ]

    # Colour map: green (low α) → neutral (0.5) → red (high α)
    cmap   = plt.get_cmap("RdYlGn_r")
    colors = [cmap(i / max(n_a - 1, 1)) for i in range(n_a)]

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6.5),
                           facecolor=STYLE["bg"], dpi=STYLE["dpi"])

    # Shaded bands between consecutive curves
    for i in range(n_a - 1):
        c1, c2 = curves[i], curves[i + 1]
        valid  = np.isfinite(c1) & np.isfinite(c2)
        if valid.any():
            ax.fill_between(v_grid[valid], c1[valid], c2[valid],
                            color=colors[i], alpha=0.12, zorder=1)

    # Curve lines
    for a, curve, col in zip(alphas, curves, colors):
        valid = np.isfinite(curve)
        ax.plot(v_grid[valid], curve[valid], lw=STYLE["lw_main"],
                color=col, label=f"α = {a:g}", zorder=3)

    # Median highlighted
    mid_idx = alphas.index(0.5) if 0.5 in alphas else n_a // 2
    valid   = np.isfinite(curves[mid_idx])
    ax.plot(v_grid[valid], curves[mid_idx][valid],
            lw=2.4, color="white", alpha=0.35, zorder=2)

    _apply_base_style(ax,
                      title=f"Conditional simulation fan  –  {copula.name}",
                      xlabel="v", ylabel="u  (conditional quantile)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    _styled_legend(ax, ncol=2, fontsize=9)

    if title:
        ax.set_title(title, fontsize=STYLE["font_title"],
                     color=STYLE["text"], fontweight="bold", pad=12)

    fig.tight_layout()
    return fig