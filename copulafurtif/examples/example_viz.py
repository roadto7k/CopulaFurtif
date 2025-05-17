import numpy as np
from scipy.stats import norm, expon
# from CopulaFurtif.core.copulas.infrastructure.registry import register_all_copulas
from CopulaFurtif.core.copulas.domain.factories.copula_factory import CopulaFactory
from CopulaFurtif.copulas import CopulaType
from CopulaFurtif.core.copulas.infrastructure.visualization import copula_viz_adapter as viz

def main():
    # register_all_copulas()
    np.random.seed(42)

    # Exemple temporaire : StudentCopula (non refactorisée pour l’instant)
    # from Service.Copulas.elliptical.student import StudentCopula  # TODO: refactor version
    # cop = StudentCopula()
    cop = CopulaFactory.create(CopulaType.AMH)
    # cop.parameters = np.array([0.8])

    print("→ Visualisation de la CDF (3D + contours)")
    # viz.plot_cdf(cop, plot_type='3d', Nsplit=60, cmap='coolwarm')
    # viz.plot_cdf(cop, plot_type='contour', Nsplit=60, levels=np.linspace(0.1, 0.9, 9), cmap='viridis')
    viz.plot_cdf(cop, plot_type='3d', Nsplit=100, cmap='coolwarm')
    viz.plot_cdf(cop, plot_type='contour', Nsplit=100, levels=np.linspace(0.1, 0.9, 9), cmap='viridis')

    print("→ Visualisation de la PDF")
    viz.plot_pdf(cop, plot_type='3d', Nsplit=60, cmap='plasma')
    viz.plot_pdf(cop, plot_type='contour', Nsplit=100, log_scale=True, levels=[0.01, 0.1, 0.4, 0.8, 1.3, 1.6])

    print("→ PDF avec marges spécifiques")
    margins = [
        {'distribution': norm, 'loc': 0, 'scale': 1},
        {'distribution': norm, 'loc': 0, 'scale': 1}
    ]
    viz.plot_mpdf(cop, margins, plot_type='contour', Nsplit=80, levels=10, cmap='terrain')

    print("→ Arbitrage frontiers avec et sans points")
    viz.plot_arbitrage_frontiers(cop, alpha_low=0.05, alpha_high=0.95, levels=200)
    u_pts = np.array([0.2, 0.5, 0.8])
    v_pts = np.array([0.3, 0.6, 0.4])
    viz.plot_arbitrage_frontiers(cop, alpha_low=0.05, alpha_high=0.95, levels=200,
                                 scatter=(u_pts, v_pts), scatter_alpha=0.4)


if __name__ == "__main__":
    main()
