"""Multi-copula comparison and AIC/BIC profile plots."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgrid

from CopulaFurtif.core.copulas.domain.copula_type import CopulaType
from CopulaFurtif.core.copulas.domain.factories.copula_factory import CopulaFactory
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel
from CopulaFurtif.core.copulas.domain.estimation.tau_calibration import calibrate_to_kendall_tau
from CopulaFurtif.core.copulas.domain.estimation.utils import safe_log
from CopulaFurtif.core.copulas.domain.estimation.estimation import pseudo_obs

from ._common import (
    STYLE, CMAP_SEQ,
    _apply_base_style, _styled_fig, _styled_colorbar, _styled_legend, _suptitle,
)


CopulaSpec = Union[CopulaType, CopulaModel]


def plot_pdf_contour_grid_calibrated_tau(
    copulas: Sequence[CopulaSpec],
    *,
    tau_target: float,
    n_cols: int = 2,
    grid_n: int = 120,
    levels: int = 12,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Grid of PDF contour plots, one panel per copula family, all calibrated to
    the same Kendall's τ.  Makes structural differences between families
    immediately visible.

    Parameters
    ----------
    copulas    : sequence of CopulaType enums or fitted CopulaModel instances
    tau_target : target Kendall's τ used to calibrate each family
    n_cols     : number of columns in the grid
    grid_n     : contour grid resolution
    levels     : number of contour levels
    title      : optional figure suptitle
    """
    tau_target = float(tau_target)
    k = len(copulas)
    if k == 0:
        raise ValueError("copulas must be non-empty")

    n_cols = max(int(n_cols), 1)
    n_rows = int(np.ceil(k / n_cols))

    cell_w, cell_h = 4.6, 4.2
    fig = _styled_fig(figsize=(cell_w * n_cols, cell_h * n_rows))
    gs  = mgrid.GridSpec(
        n_rows, n_cols, figure=fig,
        hspace=0.42, wspace=0.32,
        left=0.06, right=0.97, top=0.90, bottom=0.06,
    )

    u = np.linspace(1e-4, 1 - 1e-4, int(grid_n))
    U, V = np.meshgrid(u, u)

    for idx, spec in enumerate(copulas):
        row, col = divmod(idx, n_cols)
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor(STYLE["panel"])

        # — Instantiate & calibrate —
        if isinstance(spec, CopulaType):
            cop = CopulaFactory.create(spec)
        else:
            cop = spec.__class__()

        fixed  = None
        bounds = cop.get_bounds()
        if len(bounds) > 1:
            p0    = np.asarray(cop.get_parameters(), float).ravel()
            fixed = {i: float(p0[i]) for i in range(1, p0.size)}

        res = calibrate_to_kendall_tau(cop, tau_target, param_index=0,
                                       fixed_params=fixed)
        cop.set_parameters(res.param)

        Z = np.asarray(cop.get_pdf(U, V), float)
        Z[~np.isfinite(Z)] = np.nan

        # Filled + line contours
        zmin = np.nanpercentile(Z, 2)
        zmax = np.nanpercentile(Z, 98)
        lvls = np.linspace(zmin, zmax, levels)

        cf = ax.contourf(U, V, Z, levels=lvls, cmap=CMAP_SEQ, alpha=0.85)
        ax.contour(U, V, Z, levels=lvls, colors="white",
                   linewidths=STYLE["lw_thin"], alpha=0.35)

        _apply_base_style(
            ax,
            title=f"{cop.get_name()}\nτ ≈ {res.tau_achieved:.3f}",
            xlabel="u", ylabel="v",
        )
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        _styled_colorbar(fig, cf, ax)

    # Hide unused cells
    for idx in range(k, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        fig.add_subplot(gs[row, col]).axis("off")

    _suptitle(fig, title or f"PDF contour grid  –  τ = {tau_target:.3f}")
    return fig


def plot_aic_bic_vs_param_profile(
    copula: CopulaModel,
    x: np.ndarray,
    y: np.ndarray,
    *,
    assume_uniform: bool = False,
    param_index: int = 0,
    n_grid: int = 120,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Log-likelihood and AIC/BIC as a function of a single copula parameter.
    Reveals flatness of the likelihood surface and identifiability issues.

    Parameters
    ----------
    copula        : fitted CopulaModel (its current parameters set the baseline)
    x, y          : data arrays (raw or pseudo-observations)
    assume_uniform: if True, treat x/y as already in (0,1)
    param_index   : which parameter to sweep (default: 0, the main θ)
    n_grid        : number of grid points
    title         : optional suptitle
    """
    if assume_uniform:
        u = np.asarray(x, float).ravel()
        v = np.asarray(y, float).ravel()
    else:
        u, v = pseudo_obs([x, y])
    n = u.size

    p0     = np.asarray(copula.get_parameters(), float).ravel()
    bounds = copula.get_bounds()
    if not bounds:
        raise ValueError(f"Copula {copula.get_name()} has no bounds")
    if not (0 <= param_index < len(bounds)):
        raise ValueError("param_index out of range")

    lo, hi = bounds[param_index]
    grid   = np.linspace(lo, hi, int(n_grid))
    names  = copula.get_parameters_names()
    param_label = names[param_index] if names else f"θ[{param_index}]"

    loglik = np.empty_like(grid)
    for i, th in enumerate(grid):
        p = p0.copy(); p[param_index] = float(th)
        pdf = np.asarray(copula.get_pdf(u, v, param=p), float)
        loglik[i] = float(np.sum(safe_log(np.maximum(pdf, 1e-300))))

    k_params = len(bounds)
    aic = 2 * k_params - 2 * loglik
    bic = k_params * np.log(max(n, 1)) - 2 * loglik

    # MLE marker
    best_idx = int(np.argmax(loglik))

    fig = _styled_fig(figsize=(9, 7))
    gs  = mgrid.GridSpec(2, 1, figure=fig, hspace=0.12,
                         left=0.10, right=0.95, top=0.88, bottom=0.10)

    # — Log-likelihood panel —
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(grid, loglik, color=STYLE["accent"], lw=STYLE["lw_main"])
    ax1.axvline(grid[best_idx], color=STYLE["warn"], lw=1.2, linestyle="--",
                label=f"MLE  {param_label} = {grid[best_idx]:.4f}")
    ax1.fill_between(grid, loglik.min(), loglik,
                     color=STYLE["accent"], alpha=0.10)
    _apply_base_style(ax1, title="", xlabel="", ylabel="log-likelihood")
    ax1.tick_params(axis="x", labelbottom=False)
    _styled_legend(ax1)

    # — AIC / BIC panel —
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(grid, aic, color=STYLE["danger"],  lw=STYLE["lw_main"], label="AIC")
    ax2.plot(grid, bic, color=STYLE["success"], lw=STYLE["lw_main"], label="BIC")
    ax2.axvline(grid[best_idx], color=STYLE["warn"], lw=1.2, linestyle="--")
    _apply_base_style(ax2, xlabel=param_label, ylabel="Information criteria")
    _styled_legend(ax2)

    _suptitle(fig, title or f"AIC / BIC profile  –  {copula.get_name()}")
    return fig