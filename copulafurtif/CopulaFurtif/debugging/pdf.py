import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
# Enable 64-bit for precision
jax.config.update("jax_enable_x64", True)
from CopulaFurtif.core.copulas.domain.models.archimedean.BB2 import BB2Copula
from CopulaFurtif.core.copulas.domain.models.archimedean.BB1 import BB1Copula

def map_pdf_vs_fdiff(copula,
                     theta, delta,
                     n_grid=120,
                     u_min=1e-3, u_max=0.999,
                     v_min=1e-3, v_max=0.999,
                     h=1e-6,
                     top_k=20,
                     err_thresh=0.1,
                     plot=True,
                     cmap='viridis'):
    """
    Compare analytic PDF vs. numeric mixed finite diff over a grid.

    Parameters
    ----------
    copula   : object with get_cdf(u,v,[theta,delta]) and get_pdf(u,v,[theta,delta])
    theta, delta : floats
    n_grid   : int, number of grid points per axis
    u_min, u_max, v_min, v_max : floats, grid bounds
    h        : float, step for finite difference stencil
    top_k    : int, show top-K worst relative errors
    plot     : bool, draw heatmap of log10(relative error)
    cmap     : str, matplotlib colormap

    Returns
    -------
    results : dict with
        'u_grid', 'v_grid'  : 1D arrays
        'pdf_ana', 'pdf_num': 2D arrays shape (n_grid, n_grid)
        'rel_err'           : 2D array
        'worst'             : pandas DataFrame of top_k worst points
    """

    # Shortcuts to reduce attribute lookups
    get_pdf = copula.get_pdf
    get_cdf = copula.get_cdf

    def mixed_finite_diff(C, u, v):
        return (C(u + h, v + h) - C(u + h, v - h)
                - C(u - h, v + h) + C(u - h, v - h)) / (4.0 * h * h)

    # prepare grid
    us = np.linspace(u_min, u_max, n_grid)
    vs = np.linspace(v_min, v_max, n_grid)
    PDF_ANA = np.empty((n_grid, n_grid))
    PDF_NUM = np.empty_like(PDF_ANA)
    REL_ERR = np.empty_like(PDF_ANA)

    # CDF wrapper
    def C(u_, v_):
        return get_cdf(u_, v_, [theta, delta])

    # compute arrays
    for i, u in enumerate(us):
        for j, v in enumerate(vs):
            ana = get_pdf(u, v, [theta, delta])
            num = mixed_finite_diff(C, u, v)
            PDF_ANA[i, j] = ana
            PDF_NUM[i, j] = num
            REL_ERR[i, j] = abs(ana - num) / max(abs(ana), 1e-16)

    # find worst offenders
    flat_idx = np.dstack(np.indices(REL_ERR.shape)).reshape(-1, 2)
    top_idx = np.argsort(REL_ERR.ravel())[::-1][:top_k]
    worst = pd.DataFrame([{
        "u": us[i], "v": vs[j],
        "pdf_ana": PDF_ANA[i, j],
        "pdf_num": PDF_NUM[i, j],
        "rel_err": REL_ERR[i, j]
    } for i, j in flat_idx[top_idx]])

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1) log10(relative error) heatmap
        im0 = axes[0].imshow(
            np.log10(np.maximum(REL_ERR, 1e-16)),
            origin='lower',
            extent=[v_min, v_max, u_min, u_max],
            cmap=cmap
        )
        fig.colorbar(im0, ax=axes[0], label='log10(rel err)')
        axes[0].set(title=f'log10(rel_err) (θ={theta}, δ={delta})',
                    xlabel='v', ylabel='u')

        # 2) mask where rel_err > err_thresh
        mask = REL_ERR > err_thresh
        axes[1].imshow(~mask, origin='lower',
                       extent=[v_min, v_max, u_min, u_max],
                       cmap='Greys')
        axes[1].imshow(mask, origin='lower',
                       extent=[v_min, v_max, u_min, u_max],
                       cmap='Reds', alpha=0.6)
        axes[1].set(title=f'Rel_err > {err_thresh * 100:.0f}%',
                    xlabel='v', ylabel='u')

        # 3) log10 analytic PDF heatmap
        im2 = axes[2].imshow(
            np.log10(np.maximum(PDF_ANA, 1e-16)),
            origin='lower',
            extent=[v_min, v_max, u_min, u_max],
            cmap=cmap
        )
        fig.colorbar(im2, ax=axes[2], label='log10(pdf_ana)')
        axes[2].set(title=f'log10 analytic PDF (θ={theta}, δ={delta})',
                    xlabel='v', ylabel='u')

        plt.tight_layout()
        plt.show()

    return {
        'u_grid': us,
        'v_grid': vs,
        'pdf_ana': PDF_ANA,
        'pdf_num': PDF_NUM,
        'rel_err': REL_ERR,
        'worst': worst
    }

def map_pdf_vs_hessian_jax(copula,
                           theta, delta,
                           n_grid=120,
                           u_min=1e-3, u_max=0.999,
                           v_min=1e-3, v_max=0.999,
                           err_thresh=0.1,
                           plot=True,
                           cmap='viridis'):
    """
    Compare analytic PDF vs. mixed second derivative of C via JAX Hessian.

    Parameters
    ----------
    copula : object
        Must expose:
            - jax_get_cdf(u, v, theta, delta)  -> jnp scalar
            - jax_get_pdf(u, v, theta, delta)  -> jnp scalar
        or adapt your methods accordingly.
    theta, delta : floats
    n_grid : int
    u_min...v_max : float
    err_thresh : float, rel error threshold
    plot : bool
    cmap : str

    Returns
    -------
    dict with grids, arrays, worst offenders DataFrame.
    """

    # JAX wrappers -------------------------------------------------------
    # We'll assume you added these to your class:
    #     copula.jax_get_cdf(u, v, theta, delta)
    #     copula.jax_get_pdf(u, v, theta, delta)
    #
    # If you still only have numpy versions, rewrite them w/ jax.numpy.
    def C_jax(uv):
        u, v = uv[0], uv[1]
        return _jax_get_cdf(u, v, theta, delta)

    # Hessian of C(u,v)
    C_hess = jax.jit(jax.hessian(C_jax))
    # Mixed second derivative is H[0,1]
    def mixed_second(u, v):
        H = C_hess(jnp.array([u, v]))
        return H[0, 1]

    # Analytic PDF wrapper in JAX
    def pdf_ana(u, v):
        return copula.jax_get_pdf(u, v, theta, delta)

    # Build grid ---------------------------------------------------------
    us = np.linspace(u_min, u_max, n_grid)
    vs = np.linspace(v_min, v_max, n_grid)

    # Vectorize over grid with vmap (faster than python loops)
    UU, VV = np.meshgrid(us, vs, indexing='ij')
    uv_pairs = jnp.stack([jnp.ravel(jnp.array(UU)), jnp.ravel(jnp.array(VV))], axis=1)

    pdf_num_flat = jax.vmap(lambda uv: mixed_second(uv[0], uv[1]))(uv_pairs)
    pdf_ana_flat = jax.vmap(lambda uv: jax_bb2_pdf(uv[0], uv[1], theta, delta))(uv_pairs)

    PDF_NUM = np.array(pdf_num_flat).reshape(n_grid, n_grid)
    PDF_ANA = np.array(pdf_ana_flat).reshape(n_grid, n_grid)

    # Relative error
    REL_ERR = np.abs(PDF_ANA - PDF_NUM) / np.maximum(np.abs(PDF_ANA), 1e-16)

    # Worst offenders
    flat_idx = np.dstack(np.indices(REL_ERR.shape)).reshape(-1, 2)
    top_idx = np.argsort(REL_ERR.ravel())[::-1][:20]
    rows = []
    for idx in top_idx:
        i, j = flat_idx[idx]
        rows.append({
            "u": us[i],
            "v": vs[j],
            "pdf_ana": PDF_ANA[i, j],
            "pdf_num": PDF_NUM[i, j],
            "rel_err": REL_ERR[i, j]
        })
    worst = pd.DataFrame(rows)

    # Plots --------------------------------------------------------------
    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1) log10(relative error)
        im0 = axes[0].imshow(
            np.log10(np.maximum(REL_ERR, 1e-16)),
            origin='lower',
            extent=[v_min, v_max, u_min, u_max],
            cmap=cmap
        )
        fig.colorbar(im0, ax=axes[0], label='log10(rel err)')
        axes[0].set(title=f'log10(rel_err) (θ={theta}, δ={delta})',
                    xlabel='v', ylabel='u')

        # 2) mask where rel_err > err_thresh
        mask = REL_ERR > err_thresh
        axes[1].imshow(~mask, origin='lower',
                       extent=[v_min, v_max, u_min, u_max],
                       cmap='Greys')
        axes[1].imshow(mask, origin='lower',
                       extent=[v_min, v_max, u_min, u_max],
                       cmap='Reds', alpha=0.6)
        axes[1].set(title=f'Rel_err > {err_thresh*100:.0f}%',
                    xlabel='v', ylabel='u')

        # 3) log10 analytic PDF
        im2 = axes[2].imshow(
            np.log10(np.maximum(PDF_ANA, 1e-16)),
            origin='lower',
            extent=[v_min, v_max, u_min, u_max],
            cmap=cmap
        )
        fig.colorbar(im2, ax=axes[2], label='log10(pdf_ana)')
        axes[2].set(title=f'log10 analytic PDF (θ={theta}, δ={delta})',
                    xlabel='v', ylabel='u')

        plt.tight_layout()
        plt.show()

    return {
        'u_grid': us,
        'v_grid': vs,
        'pdf_ana': PDF_ANA,
        'pdf_num': PDF_NUM,
        'rel_err': REL_ERR,
        'worst': worst
    }

if __name__ == "__main__":

    # c = BB2Copula()
    # theta = 9
    # delta = 9
    # c.set_parameters([theta, delta])
    # res = map_pdf_vs_fdiff(c, theta, delta, n_grid=100, h=1e-6, top_k=20, u_min=3e-3, u_max=0.999,
    #                  v_min=3e-3, v_max=0.999,)
    # print(res['worst'])

    # 1) Instantiate and set parameters
    c = BB2Copula()
    theta, delta = 2.0, 8.0
    c.set_parameters([theta, delta])

    # 2) Run the JAX‐Hessian based comparison on a 100×100 grid
    res = map_pdf_vs_hessian_jax(
        copula=c,
        theta=theta,
        delta=delta,
        n_grid=300,
        u_min=3e-3,
        u_max=0.999,
        v_min=3e-3,
        v_max=0.999,
        err_thresh=0.1,
        plot=True
    )

    # 3) Inspect the top offenders
    print("Top relative‐error points:")
    print(res['worst'])
