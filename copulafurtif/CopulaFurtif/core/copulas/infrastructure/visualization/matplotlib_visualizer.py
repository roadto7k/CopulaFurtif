import matplotlib.gridspec as mgrid
import numpy as np

from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel
from CopulaFurtif.core.copulas.domain.estimation.estimation import pseudo_obs

from ._common import (
    STYLE, CMAP_DIV,
    _apply_base_style, _styled_fig, _styled_colorbar, _styled_legend,
)


class MatplotlibCopulaVisualizer:

    # ── 1. Residual Heatmap ──────────────────────────────────────────────────

    @staticmethod
    def plot_residual_heatmap(copula: CopulaModel, u, v, bins: int = 50):
        u, v = np.asarray(u).flatten(), np.asarray(v).flatten()

        grid = np.linspace(0, 1, bins)
        U, V = np.meshgrid(grid, grid, indexing="ij")

        emp = np.array([
            np.mean((u <= U[i, j]) & (v <= V[i, j]))
            for i in range(bins) for j in range(bins)
        ]).reshape(bins, bins)

        model = np.array([
            copula.get_cdf(ui, vi)
            for ui, vi in zip(U.ravel(), V.ravel())
        ]).reshape(bins, bins)

        residuals = emp - model
        vmax = np.abs(residuals).max()

        fig = _styled_fig(figsize=(12, 5))
        gs  = mgrid.GridSpec(1, 2, figure=fig, wspace=0.35,
                             left=0.07, right=0.95, top=0.88, bottom=0.12)

        ax_heat = fig.add_subplot(gs[0])
        ax_heat.set_facecolor(STYLE["panel"])
        im = ax_heat.imshow(
            residuals.T, origin="lower", extent=[0, 1, 0, 1],
            cmap=CMAP_DIV, vmin=-vmax, vmax=vmax, aspect="auto",
        )
        ax_heat.contour(grid, grid, residuals.T, levels=[0],
                        colors="white", linewidths=1.2, linestyles="--")
        _apply_base_style(ax_heat,
                          title=f"{copula.name}  –  Residual Heatmap  (Emp − Model)",
                          xlabel="u", ylabel="v")
        _styled_colorbar(fig, im, ax_heat, label="Residual")

        rmse = np.sqrt(np.mean(residuals ** 2))
        ax_heat.text(0.03, 0.97, f"RMSE = {rmse:.4f}",
                     transform=ax_heat.transAxes, va="top",
                     color=STYLE["warn"], fontsize=10,
                     bbox=dict(facecolor=STYLE["panel"], edgecolor=STYLE["grid"],
                               boxstyle="round,pad=0.3", alpha=0.85))

        ax_diag = fig.add_subplot(gs[1])
        diag_res = residuals[np.arange(bins), np.arange(bins)]
        ax_diag.fill_between(grid, 0, diag_res,
                             where=diag_res >= 0, color=STYLE["accent"], alpha=0.4)
        ax_diag.fill_between(grid, 0, diag_res,
                             where=diag_res <  0, color=STYLE["danger"], alpha=0.4)
        ax_diag.plot(grid, diag_res, color=STYLE["accent"],
                     lw=STYLE["lw_main"], label="Residual on diagonal")
        ax_diag.axhline(0, color=STYLE["subtext"], lw=1, linestyle="--")
        _apply_base_style(ax_diag, title="Diagonal Profile  (u = v)",
                          xlabel="t", ylabel="Residual")
        _styled_legend(ax_diag)

        # tight_layout only on axes that support it (exclude colorbar axes)
        fig.tight_layout(rect=[0, 0, 1, 1])
        return residuals

    # ── 2. Tail Dependence ───────────────────────────────────────────────────

    @staticmethod
    def plot_tail_dependence(data, candidate_list,
                             q_low: float = 0.05, q_high: float = 0.95):
        u, v = pseudo_obs(data)
        u, v = np.asarray(u).flatten(), np.asarray(v).flatten()

        lower_mask = (u <= q_low)  & (v <= q_low)
        upper_mask = (u >  q_high) & (v >  q_high)
        mid_mask   = ~lower_mask & ~upper_mask

        denom_L = np.sum(u <= q_low)
        denom_U = np.sum(u >  q_high)
        emp_L = np.sum(lower_mask) / denom_L if denom_L > 0 else 0.0
        emp_U = np.sum(upper_mask) / denom_U if denom_U > 0 else 0.0

        fig = _styled_fig(figsize=(14, 6))
        gs  = mgrid.GridSpec(
            1, 3, figure=fig,
            width_ratios=[5, 5, 3.5], wspace=0.38,
            left=0.06, right=0.97, top=0.88, bottom=0.13,
        )

        scatter_mid  = dict(s=12, alpha=0.25, lw=0, color=STYLE["subtext"])
        scatter_tail = dict(s=35, edgecolors="white", linewidths=0.4, zorder=4)

        # — Lower tail —
        ax_lo = fig.add_subplot(gs[0])
        ax_lo.scatter(u[mid_mask],   v[mid_mask],   **scatter_mid, label="Background")
        ax_lo.scatter(u[lower_mask], v[lower_mask], color=STYLE["success"],
                      label=f"Lower tail  (n={lower_mask.sum()})", **scatter_tail)
        ax_lo.axvline(q_low, color=STYLE["success"], lw=1.2, linestyle=":")
        ax_lo.axhline(q_low, color=STYLE["success"], lw=1.2, linestyle=":")
        ax_lo.fill_between([0, q_low], 0, q_low, color=STYLE["success"], alpha=0.07)
        _apply_base_style(ax_lo,
                          title=f"Lower Tail  –  q ≤ {q_low}\nEmpirical λ_L = {emp_L:.4f}",
                          xlabel="u", ylabel="v")
        ax_lo.set_xlim(0, 1); ax_lo.set_ylim(0, 1)
        _styled_legend(ax_lo)

        # — Upper tail —
        ax_hi = fig.add_subplot(gs[1])
        ax_hi.scatter(u[mid_mask],   v[mid_mask],   **scatter_mid, label="Background")
        ax_hi.scatter(u[upper_mask], v[upper_mask], color=STYLE["danger"],
                      label=f"Upper tail  (n={upper_mask.sum()})", **scatter_tail)
        ax_hi.axvline(q_high, color=STYLE["danger"], lw=1.2, linestyle=":")
        ax_hi.axhline(q_high, color=STYLE["danger"], lw=1.2, linestyle=":")
        ax_hi.fill_between([q_high, 1], q_high, 1, color=STYLE["danger"], alpha=0.07)
        _apply_base_style(ax_hi,
                          title=f"Upper Tail  –  q > {q_high}\nEmpirical λ_U = {emp_U:.4f}",
                          xlabel="u", ylabel="v")
        ax_hi.set_xlim(0, 1); ax_hi.set_ylim(0, 1)
        _styled_legend(ax_hi)

        # — Candidate table (drawn with ax.text only, no axhline/axhspan) —
        ax_tbl = fig.add_subplot(gs[2])
        ax_tbl.set_facecolor(STYLE["panel"])
        ax_tbl.set_xlim(0, 1)
        ax_tbl.set_ylim(0, 1)
        ax_tbl.axis("off")

        col_x = [0.05, 0.45, 0.72]
        row_h = 0.08
        y     = 0.93

        # Header row
        for cx, hdr in zip(col_x, ["Copula", "λ_L", "λ_U"]):
            ax_tbl.text(cx, y, hdr, color=STYLE["accent"], fontsize=10,
                        fontweight="bold", transform=ax_tbl.transAxes)

        # Separator line via ax.plot (data coords, no transform conflict)
        ax_tbl.plot([0.02, 0.98], [y - 0.035, y - 0.035],
                    color=STYLE["grid"], lw=1, transform=ax_tbl.transAxes,
                    clip_on=False)

        # Empirical row
        y -= row_h
        for cx, val in zip(col_x, ["Empirical", f"{emp_L:.4f}", f"{emp_U:.4f}"]):
            ax_tbl.text(cx, y, val, color=STYLE["warn"], fontsize=9,
                        transform=ax_tbl.transAxes)

        # Thin separator after empirical
        ax_tbl.plot([0.02, 0.98], [y - 0.025, y - 0.025],
                    color=STYLE["grid"], lw=0.6, transform=ax_tbl.transAxes,
                    clip_on=False)

        # Candidate rows
        for k, cop in enumerate(candidate_list):
            param = cop.get_parameters()
            row   = [cop.get_name(), f"{cop.LTDC(param):.4f}", f"{cop.UTDC(param):.4f}"]
            y    -= row_h
            # Alternating background via Rectangle patch instead of axhspan
            bg = STYLE["bg"] if k % 2 == 0 else STYLE["panel"]
            from matplotlib.patches import FancyBboxPatch
            rect = FancyBboxPatch(
                (0.01, y - row_h * 0.35), 0.98, row_h * 0.9,
                boxstyle="square,pad=0",
                transform=ax_tbl.transAxes, clip_on=False,
                facecolor=bg, edgecolor="none", alpha=0.5, zorder=0,
            )
            ax_tbl.add_patch(rect)
            for cx, val in zip(col_x, row):
                ax_tbl.text(cx, y, val, color=STYLE["text"], fontsize=9,
                            transform=ax_tbl.transAxes, zorder=1)

        ax_tbl.set_title("Tail Dependence\nCoefficients",
                         fontsize=12, color=STYLE["text"], fontweight="bold", pad=10)

        # Use subplots_adjust instead of tight_layout to avoid the axis("off") warning
        fig.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.13, wspace=0.38)