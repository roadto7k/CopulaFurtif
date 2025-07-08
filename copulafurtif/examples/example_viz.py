import numpy as np
from scipy.stats import norm, expon
# from CopulaFurtif.core.copulas.infrastructure.registry import register_all_copulas
from CopulaFurtif.copulas import CopulaType, CopulaFactory
from CopulaFurtif.visualization import plot_arbitrage_frontiers, plot_pdf, plot_cdf, plot_mpdf, Plot_type
def main():
    # register_all_copulas()
    np.random.seed(42)

    # Exemple temporaire : StudentCopula (non refactorisée pour l’instant)
    # from Service.Copulas.elliptical.student import StudentCopula  # TODO: refactor version
    # cop = StudentCopula()
    cop = CopulaFactory.create(CopulaType.GAUSSIAN)
    # cop.parameters = np.array([0.8])

    print("→ Visualisation de la CDF (3D + contours)")
    # viz.plot_cdf(cop, plot_type='3d', Nsplit=60, cmap='coolwarm')
    # viz.plot_cdf(cop, plot_type='contour', Nsplit=60, levels=np.linspace(0.1, 0.9, 9), cmap='viridis')
    plot_cdf(cop, plot_type=Plot_type.DIM3, Nsplit=100, cmap='coolwarm')
    plot_cdf(cop, plot_type=Plot_type.CONTOUR, Nsplit=100, levels=np.linspace(0.1, 0.9, 9), cmap='viridis')

    print("→ Visualisation de la PDF")
    plot_pdf(cop, plot_type=Plot_type.DIM3, Nsplit=60, cmap='plasma')
    plot_pdf(cop, plot_type=Plot_type.CONTOUR, Nsplit=100, log_scale=True, levels=[0.01, 0.1, 0.4, 0.8, 1.3, 1.6])

    print("→ PDF avec marges spécifiques")
    margins = [
        {'distribution': norm, 'loc': 0, 'scale': 1},
        {'distribution': norm, 'loc': 0, 'scale': 1}
    ]
    plot_mpdf(cop, margins, plot_type=Plot_type.CONTOUR, Nsplit=80, levels=10, cmap='terrain')

    print("→ Arbitrage frontiers avec et sans points")
    plot_arbitrage_frontiers(cop, alpha_low=0.05, alpha_high=0.95, levels=200)
    u_pts = np.array([0.2, 0.5, 0.8])
    v_pts = np.array([0.3, 0.6, 0.4])
    plot_arbitrage_frontiers(cop, alpha_low=0.05, alpha_high=0.95, levels=200,
                                 scatter=(u_pts, v_pts), scatter_alpha=0.4)


if __name__ == "__main__":
    main()
