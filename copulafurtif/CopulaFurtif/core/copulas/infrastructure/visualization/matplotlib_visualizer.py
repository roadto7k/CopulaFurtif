import matplotlib.pyplot as plt
import matplotlib.gridspec as mgrid
import numpy as np
from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel
from CopulaFurtif.copula_utils import pseudo_obs


# ─────────────────────────────────────────────────────────────────────────────
# SHARED STYLE  (keep in sync with copula_viewer.py)
# ─────────────────────────────────────────────────────────────────────────────

STYLE = {
    "bg":          "#0d1117",
    "panel":       "#161b22",
    "grid":        "#21262d",
    "text":        "#e6edf3",
    "subtext":     "#8b949e",
    "accent":      "#58a6ff",
    "danger":      "#f85149",
    "success":     "#3fb950",
    "warn":        "#d29922",
    "font_title":  16,
    "font_label":  12,
    "font_tick":   10,
    "font_legend": 10,
    "dpi":         130,
    "lw_main":     1.8,
}

CMAP_DIV = "RdYlBu_r"
CMAP_SEQ = "plasma"


def _apply_base_style(ax, title="", xlabel="", ylabel=""):
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


def _styled_fig(*args, **kwargs):
    fig = plt.figure(*args, facecolor=STYLE["bg"], dpi=STYLE["dpi"], **kwargs)
    return fig


def _styled_colorbar(fig, mappable, ax, label=""):
    cb = fig.colorbar(mappable, ax=ax, pad=0.02, fraction=0.046)
    cb.ax.yaxis.set_tick_params(color=STYLE["subtext"])
    cb.outline.set_edgecolor(STYLE["grid"])
    plt.setp(cb.ax.yaxis.get_ticklabels(),
             color=STYLE["subtext"], fontsize=STYLE["font_tick"])
    if label:
        cb.set_label(label, color=STYLE["subtext"], fontsize=STYLE["font_tick"])
    return cb


def _styled_legend(ax):
    leg = ax.legend(fontsize=STYLE["font_legend"],
                    framealpha=0.25,
                    edgecolor=STYLE["grid"],
                    labelcolor=STYLE["text"],
                    facecolor=STYLE["panel"])
    return leg


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZER
# ─────────────────────────────────────────────────────────────────────────────

class MatplotlibCopulaVisualizer:

    # ── 1. Residual Heatmap ──────────────────────────────────────────────────

    @staticmethod
    def plot_residual_heatmap(copula: CopulaModel, u, v, bins: int = 50):
        """
        Plot a signed residual heatmap (Empirical CDF – Model CDF) over the
        unit square, together with a 1-D residual profile along the diagonal.

        Parameters
        ----------
        copula : CopulaModel
        u, v   : array-like – pseudo-observations in [0, 1]
        bins   : grid resolution
        """
        u, v = np.asarray(u).flatten(), np.asarray(v).flatten()

        # Build grid
        grid = np.linspace(0, 1, bins)
        U, V = np.meshgrid(grid, grid, indexing="ij")

        # Empirical CDF
        emp = np.array([
            np.mean((u <= U[i, j]) & (v <= V[i, j]))
            for i in range(bins) for j in range(bins)
        ]).reshape(bins, bins)

        # Model CDF
        model = np.array([
            copula.get_cdf(ui, vi, copula.get_parameters())
            for ui, vi in zip(U.ravel(), V.ravel())
        ]).reshape(bins, bins)

        residuals = emp - model
        vmax = np.abs(residuals).max()

        # ── Layout: heatmap + diagonal profile ───────────────────────────────
        fig = _styled_fig(figsize=(12, 5))
        gs  = mgrid.GridSpec(1, 2, figure=fig, wspace=0.35,
                             left=0.07, right=0.95, top=0.88, bottom=0.12)

        # — Heatmap panel —
        ax_heat = fig.add_subplot(gs[0])
        ax_heat.set_facecolor(STYLE["panel"])
        im = ax_heat.imshow(
            residuals.T,          # transpose so x=u, y=v
            origin="lower",
            extent=[0, 1, 0, 1],
            cmap=CMAP_DIV,
            vmin=-vmax, vmax=vmax,
            aspect="auto",
        )
        # Overlay thin contour lines at zero
        ax_heat.contour(grid, grid, residuals.T, levels=[0],
                        colors="white", linewidths=1.2, linestyles="--")
        _apply_base_style(ax_heat,
                          title=f"{copula.name}  –  Residual Heatmap  (Emp − Model)",
                          xlabel="u", ylabel="v")
        _styled_colorbar(fig, im, ax_heat, label="Residual")

        # RMSE annotation
        rmse = np.sqrt(np.mean(residuals ** 2))
        ax_heat.text(0.03, 0.97, f"RMSE = {rmse:.4f}",
                     transform=ax_heat.transAxes, va="top",
                     color=STYLE["warn"], fontsize=10,
                     bbox=dict(facecolor=STYLE["panel"], edgecolor=STYLE["grid"],
                               boxstyle="round,pad=0.3", alpha=0.85))

        # — Diagonal profile panel —
        ax_diag = fig.add_subplot(gs[1])
        diag_idx = np.arange(bins)
        diag_res = residuals[diag_idx, diag_idx]
        ax_diag.fill_between(grid, 0, diag_res,
                             where=diag_res >= 0, color=STYLE["accent"], alpha=0.4)
        ax_diag.fill_between(grid, 0, diag_res,
                             where=diag_res < 0,  color=STYLE["danger"], alpha=0.4)
        ax_diag.plot(grid, diag_res, color=STYLE["accent"],
                     lw=STYLE["lw_main"], label="Residual on diagonal")
        ax_diag.axhline(0, color=STYLE["subtext"], lw=1, linestyle="--")
        _apply_base_style(ax_diag,
                          title="Diagonal Profile  (u = v)",
                          xlabel="t", ylabel="Residual")
        _styled_legend(ax_diag)

        plt.show()
        return residuals

    # ── 2. Tail Dependence ───────────────────────────────────────────────────

    @staticmethod
    def plot_tail_dependence(data, candidate_list,
                             q_low: float = 0.05, q_high: float = 0.95):
        """
        Two-panel plot of pseudo-observations highlighting tail regions, with a
        structured candidate summary table below.

        Parameters
        ----------
        data           : [X, Y] – raw data samples
        candidate_list : list of fitted CopulaModel instances
        q_low          : lower tail quantile threshold
        q_high         : upper tail quantile threshold
        """
        u, v = pseudo_obs(data)
        u, v = np.asarray(u).flatten(), np.asarray(v).flatten()

        lower_mask = (u <= q_low)  & (v <= q_low)
        upper_mask = (u >  q_high) & (v >  q_high)
        mid_mask   = ~lower_mask & ~upper_mask

        denom_L = np.sum(u <= q_low)
        denom_U = np.sum(u > q_high)
        emp_L = np.sum(lower_mask) / denom_L if denom_L > 0 else 0.0
        emp_U = np.sum(upper_mask) / denom_U if denom_U > 0 else 0.0

        # ── Figure layout ─────────────────────────────────────────────────────
        fig = _styled_fig(figsize=(14, 6))
        gs  = mgrid.GridSpec(
            1, 3,
            figure=fig,
            width_ratios=[5, 5, 3.5],
            wspace=0.38,
            left=0.06, right=0.97, top=0.88, bottom=0.13,
        )

        scatter_kw_mid  = dict(s=12, alpha=0.25, lw=0, color=STYLE["subtext"])
        scatter_kw_tail = dict(s=35, edgecolors="white", linewidths=0.4, zorder=4)

        # ── Lower tail panel ─────────────────────────────────────────────────
        ax_lo = fig.add_subplot(gs[0])
        ax_lo.scatter(u[mid_mask],   v[mid_mask],   **scatter_kw_mid,
                      label="Background")
        ax_lo.scatter(u[lower_mask], v[lower_mask], color=STYLE["success"],
                      label=f"Lower tail  (n={lower_mask.sum()})",
                      **scatter_kw_tail)

        # Reference box
        ax_lo.axvline(q_low, color=STYLE["success"], lw=1.2, linestyle=":")
        ax_lo.axhline(q_low, color=STYLE["success"], lw=1.2, linestyle=":")
        ax_lo.fill_between([0, q_low], 0, q_low,
                           color=STYLE["success"], alpha=0.07)

        _apply_base_style(
            ax_lo,
            title=f"Lower Tail  –  q ≤ {q_low}\nEmpirical λ_L = {emp_L:.4f}",
            xlabel="u", ylabel="v",
        )
        ax_lo.set_xlim(0, 1); ax_lo.set_ylim(0, 1)
        _styled_legend(ax_lo)

        # ── Upper tail panel ─────────────────────────────────────────────────
        ax_hi = fig.add_subplot(gs[1])
        ax_hi.scatter(u[mid_mask],   v[mid_mask],   **scatter_kw_mid,
                      label="Background")
        ax_hi.scatter(u[upper_mask], v[upper_mask], color=STYLE["danger"],
                      label=f"Upper tail  (n={upper_mask.sum()})",
                      **scatter_kw_tail)

        ax_hi.axvline(q_high, color=STYLE["danger"], lw=1.2, linestyle=":")
        ax_hi.axhline(q_high, color=STYLE["danger"], lw=1.2, linestyle=":")
        ax_hi.fill_between([q_high, 1], q_high, 1,
                           color=STYLE["danger"], alpha=0.07)

        _apply_base_style(
            ax_hi,
            title=f"Upper Tail  –  q > {q_high}\nEmpirical λ_U = {emp_U:.4f}",
            xlabel="u", ylabel="v",
        )
        ax_hi.set_xlim(0, 1); ax_hi.set_ylim(0, 1)
        _styled_legend(ax_hi)

        # ── Candidate summary table ───────────────────────────────────────────
        ax_tbl = fig.add_subplot(gs[2])
        ax_tbl.set_facecolor(STYLE["panel"])
        ax_tbl.axis("off")

        header = ["Copula", "λ_L", "λ_U"]
        rows   = []
        for cop in candidate_list:
            param = cop.get_parameters()
            rows.append([
                cop.get_name(),
                f"{cop.LTDC(param):.4f}",
                f"{cop.UTDC(param):.4f}",
            ])

        # Empirical row at top
        emp_row = ["Empirical", f"{emp_L:.4f}", f"{emp_U:.4f}"]

        col_x   = [0.05, 0.45, 0.72]
        row_h   = 0.08
        y_start = 0.93

        # Header
        for cx, hdr in zip(col_x, header):
            ax_tbl.text(cx, y_start, hdr,
                        color=STYLE["accent"], fontsize=10, fontweight="bold",
                        transform=ax_tbl.transAxes)

        ax_tbl.axhline(y_start - 0.03,
                       xmin=0.02, xmax=0.98,
                       color=STYLE["grid"], lw=1,
                       transform=ax_tbl.transAxes)

        # Empirical row (highlighted)
        y = y_start - row_h
        for cx, val in zip(col_x, emp_row):
            ax_tbl.text(cx, y, val,
                        color=STYLE["warn"], fontsize=9,
                        transform=ax_tbl.transAxes)
        y -= row_h * 0.6
        ax_tbl.axhline(y, xmin=0.02, xmax=0.98,
                       color=STYLE["grid"], lw=0.6,
                       transform=ax_tbl.transAxes)

        # Candidate rows
        for k, row in enumerate(rows):
            y -= row_h
            bg_color = STYLE["bg"] if k % 2 == 0 else STYLE["panel"]
            ax_tbl.axhspan(y - row_h * 0.35, y + row_h * 0.55,
                           xmin=0.01, xmax=0.99,
                           color=bg_color, alpha=0.5,
                           transform=ax_tbl.transAxes)
            for cx, val in zip(col_x, row):
                ax_tbl.text(cx, y, val,
                            color=STYLE["text"], fontsize=9,
                            transform=ax_tbl.transAxes)

        ax_tbl.set_title("Tail Dependence\nCoefficients",
                         fontsize=12, color=STYLE["text"],
                         fontweight="bold", pad=10)

        plt.show()