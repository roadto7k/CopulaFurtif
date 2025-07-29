
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from matplotlib import colors

from CopulaFurtif.core.copulas.domain.models.archimedean.BB2 import BB2Copula
from CopulaFurtif.core.copulas.domain.models.archimedean.BB1 import BB1Copula


def demo_bb2(theta=4.0, delta=8.0, n_grid=150, n_colors=100):
    """Visual sanity‑check for BB2: CDF, PDF, ∂C/∂u, ∂C/∂v."""

    # ── 1 . Copula ─────────────────────────────────────────────────────
    cop = BB2Copula()
    cop.set_parameters([theta, delta])

    # ── 2 . Grid in (eps,1‑eps)² ────────────────────────────────────────
    eps = 1e-3
    u = jnp.linspace(eps, 1 - eps, n_grid)
    v = jnp.linspace(eps, 1 - eps, n_grid)
    U, V = jnp.meshgrid(u, v, indexing="ij")

    # Flatten once, evaluate, reshape back
    def _eval(f):
        return jax.vmap(lambda uu, vv: f(uu, vv))(U.ravel(), V.ravel())\
                  .reshape((n_grid, n_grid))

    C  = _eval(cop.get_cdf)
    P  = _eval(cop.get_pdf)
    Du = _eval(cop.partial_derivative_C_wrt_u)
    Dv = _eval(cop.partial_derivative_C_wrt_v)

    # Convert to NumPy for Matplotlib
    U, V, C, P, Du, Dv = map(np.asarray, (U, V, C, P, Du, Dv))

    # ── 3 . Colour handling for the PDF (LogNorm) ───────────────────────
    mask = np.isfinite(P) & (P > 0)
    positive = P[mask]
    if positive.size == 0:
        raise ValueError("PDF is identically zero or non‐finite on this grid.")
    vmin, vmax = positive.min(), positive.max()

    # 2) créer un masked array pour laisser NaN/inf « vides », et masquer aussi <= 0
    P_plot = np.ma.masked_invalid(P)  # masque NaN et ±inf
    P_plot = np.ma.masked_where(P_plot <= 0, P_plot)

    # 3) configurer les niveaux et la normalisation log
    levels_pdf = np.logspace(np.log10(vmin), np.log10(vmax), n_colors + 1)
    cmap_pdf = plt.get_cmap("plasma", n_colors).copy()
    cmap_pdf.set_bad(color="lightgray", alpha=0.6)  # couleur pour les zones masquées
    norm_pdf = colors.LogNorm(vmin=vmin, vmax=vmax)

    # Derivatives live in (0,1); linear scale is enough
    levels_du = np.linspace(0, 1, 101)
    levels_dv = np.linspace(0, 1, 101)

    # ── 4 . Plot (2×2 grid) ─────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_cdf, ax_pdf, ax_du, ax_dv = axes.ravel()

    # CDF
    cf_cdf = ax_cdf.contourf(U, V, C, levels=100, cmap="viridis")
    fig.colorbar(cf_cdf, ax=ax_cdf)  # ← added colorbar for CDF
    ax_cdf.set(
        title=f"BB2 CDF (θ={theta}, δ={delta})",
        xlim=(0, 1), ylim=(0, 1),
        xlabel="u", ylabel="v"
    )

    # PDF (log colour)
    cf_pdf = ax_pdf.contourf(
        U, V, P_plot,
        levels=levels_pdf,
        cmap=cmap_pdf,
        norm=norm_pdf,
        extend="both"
    )
    fig.colorbar(cf_pdf, ax=ax_pdf,
                 ticks=levels_pdf[::max(1, n_colors // 10)],
                 format="%.1e")
    ax_pdf.set(
        title="BB2 PDF (Log scale)",
        xlim=(0, 1), ylim=(0, 1),
        xlabel="u", ylabel="v"
    )

    # ∂C/∂u  (linear colour)
    cf_du = ax_du.contourf(U, V, Du, levels=levels_du, cmap="magma")
    fig.colorbar(cf_du, ax=ax_du)
    ax_du.set(
        title="∂C/∂u",
        xlim=(0, 1), ylim=(0, 1),
        xlabel="u", ylabel="v"
    )

    # ∂C/∂v  (linear colour)
    cf_dv = ax_dv.contourf(U, V, Dv, levels=levels_dv, cmap="magma")
    fig.colorbar(cf_dv, ax=ax_dv)
    ax_dv.set(
        title="∂C/∂v",
        xlim=(0, 1), ylim=(0, 1),
        xlabel="u", ylabel="v"
    )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # run it
    demo_bb2(theta=2.0, delta=2.0, n_grid=200, n_colors=100)
    demo_bb2(theta=4.0, delta=2.0, n_grid=200, n_colors=100)
    demo_bb2(theta=6.0, delta=2.0, n_grid=200, n_colors=100)
    demo_bb2(theta=8.0, delta=2.0, n_grid=200, n_colors=100)
    demo_bb2(theta=2.0, delta=4.0, n_grid=200, n_colors=100)
    demo_bb2(theta=2.0, delta=6.0, n_grid=200, n_colors=100)
    demo_bb2(theta=2.0, delta=8.0, n_grid=200, n_colors=100)
