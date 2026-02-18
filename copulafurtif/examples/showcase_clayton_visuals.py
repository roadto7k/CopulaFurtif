# DataAnalysis/showcase_clayton_visuals.py
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ---------------------------------------------------------------------
# Make repo imports work when running from repo root
# ---------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "copulafurtif"))

# ---------------------------------------------------------------------
# Imports from your project
# ---------------------------------------------------------------------
from CopulaFurtif.core.copulas.domain.models.archimedean.clayton import ClaytonCopula
from CopulaFurtif.core.copulas.domain.estimation.estimation import quick_fit

# Existing visuals already in your codebase
from CopulaFurtif.core.copulas.infrastructure.visualization.matplotlib_visualizer import (
    MatplotlibCopulaVisualizer,
)

# New visuals you added
from CopulaFurtif.core.copulas.infrastructure.visualization.plots_corner import (
    plot_simulated_corner_with_kdes,
    plot_rosenblatt_pit_corner,
)
from CopulaFurtif.core.copulas.infrastructure.visualization.plots_dependence import (
    plot_tail_concentration_curves,
    plot_pickands_dependence_function,
    plot_kendall_k_plot,
    plot_chi_plot,
    plot_conditional_simulation_fan,
)


def main():
    # -----------------------------------------------------------------
    # 0) Config
    # -----------------------------------------------------------------
    seed = 42
    rng = np.random.default_rng(seed)
    n = 5000

    theta_true = 2.0  # Clayton tau = theta/(theta+2) => tau = 0.5
    out_dir = REPO / "DataAnalysis" / "_showcase_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # 1) Generate synthetic data from a TRUE Clayton
    # -----------------------------------------------------------------
    cop_true = ClaytonCopula()
    cop_true.set_parameters([theta_true])

    uv = cop_true.sample(n, rng=rng)  # normalized API
    u = uv[:, 0]
    v = uv[:, 1]

    # Turn copula-uniforms into raw marginals (so pseudo_obs path is exercised)
    # Here: Normal marginals (you can swap to t, gamma, etc.)
    X = norm.ppf(u)
    Y = norm.ppf(v)

    print("=== TRUE COPULA ===")
    print("theta_true:", theta_true)
    print("tau_true  :", cop_true.kendall_tau([theta_true]))

    # -----------------------------------------------------------------
    # 2) Fit a new Clayton on (X,Y) using your estimation stack (CMLE quick)
    # -----------------------------------------------------------------
    cop_fit = ClaytonCopula()
    fit_res = quick_fit((X, Y), cop_fit, mode="cmle", maxiter=80, return_metrics=True)

    theta_hat = float(cop_fit.get_parameters()[0])
    tau_hat = float(cop_fit.kendall_tau([theta_hat]))

    print("\n=== FITTED COPULA (CMLE quick) ===")
    print("theta_hat:", theta_hat)
    print("tau_hat  :", tau_hat)
    print("fit_res  :", fit_res)

    # -----------------------------------------------------------------
    # 3) SHOWCASE VISUALS
    # -----------------------------------------------------------------

    # (A) Simulated corner plot (from fitted copula) + overlay data (pseudo-obs)
    fig = plot_simulated_corner_with_kdes(
        cop_fit,
        n=3000,
        seed=seed,
        overlay_data=(X, Y),          # raw data (will be rank-transformed internally if needed)
        overlay_assume_uniform=False,
        title="(1) Simulated corner + KDE marginals (Clayton fit) + overlay data",
    )
    fig.savefig(out_dir / "01_corner_sim_kde.png", dpi=150)
    plt.show()

    # (B) Rosenblatt PIT diagnostic (uniform^2 if fit is good)
    fig = plot_rosenblatt_pit_corner(
        cop_fit,
        X, Y,
        max_n=600,
        seed=seed,
        title="(2) Rosenblatt PIT diagnostic (Clayton fit)",
    )
    fig.savefig(out_dir / "02_rosenblatt_pit.png", dpi=150)
    plt.show()

    # (C) Residual heatmap (your existing diagnostic)
    #     This expects u,v. We pass pseudo-obs by ranking through your helper:
    #     easiest: reuse the copula_utils pseudo_obs indirectly via MatplotlibCopulaVisualizer usage patterns,
    #     but here we just compute rank-based ourselves to keep it simple.
    def _pseudo_obs_2(x):
        x = np.asarray(x)
        r = np.argsort(np.argsort(x)) + 1
        return r / (len(x) + 1)

    u_data = _pseudo_obs_2(X)
    v_data = _pseudo_obs_2(Y)

    MatplotlibCopulaVisualizer.plot_residual_heatmap(cop_fit, u_data, v_data, bins=60)
    plt.savefig(out_dir / "03_residual_heatmap.png", dpi=150)
    plt.show()

    # (D) Tail dependence scatter (your existing plot_tail_dependence)
    #     Candidate list: just fitted copula in a list
    MatplotlibCopulaVisualizer.plot_tail_dependence((X, Y), [cop_fit], q_low=0.05, q_high=0.95)
    plt.savefig(out_dir / "04_tail_dependence_scatter.png", dpi=150)
    plt.show()

    # (E) Tail concentration curves L(t)/U(t) — empirical vs theoretical
    fig = plot_tail_concentration_curves(
        cop_fit,
        X, Y,
        assume_uniform=False,
        t_min=0.01, t_max=0.30, n_t=70,
        normalize_by_t=True,
        title="(5) Tail concentration curves (empirical vs theoretical)",
    )
    fig.savefig(out_dir / "05_tail_concentration.png", dpi=150)
    plt.show()

    # (F) Pickands dependence function (EV-style diagnostic; still informative for others)
    fig = plot_pickands_dependence_function(
        cop_fit,
        X, Y,
        assume_uniform=False,
        t=1.0,
        n_w=121,
        title="(6) Pickands dependence function A(ω): empirical vs theoretical",
    )
    fig.savefig(out_dir / "06_pickands.png", dpi=150)
    plt.show()

    # (G) Kendall K-plot (non-parametric dependence diagnostic)
    fig = plot_kendall_k_plot(
        X, Y,
        assume_uniform=False,
        max_n=600,
        seed=seed,
        title="(7) Kendall K-plot (empirical joint CDF order stats)",
    )
    fig.savefig(out_dir / "07_kendall_kplot.png", dpi=150)
    plt.show()

    # (H) Chi-plot (independence vs systematic deviation)
    fig = plot_chi_plot(
        X, Y,
        assume_uniform=False,
        mode="NULL",
        max_n=900,
        seed=seed,
        title="(8) Chi-plot (NULL mode)",
    )
    fig.savefig(out_dir / "08_chi_plot.png", dpi=150)
    plt.show()

    # (I) Conditional simulation fan: solve conditional quantiles u(v; alpha)
    #     Requires partial derivatives implemented (Clayton has them ✅)
    fig = plot_conditional_simulation_fan(
        cop_fit,
        alphas=(0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95),
        n_grid=220,
        title="(9) Conditional simulation fan (u|v at multiple α)",
    )
    fig.savefig(out_dir / "09_conditional_fan.png", dpi=150)
    plt.show()

    print(f"\nSaved showcase figures to: {out_dir}")


if __name__ == "__main__":
    main()
