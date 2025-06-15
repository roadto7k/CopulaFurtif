import numpy as np
from scipy.stats import norm, expon

# 1) Instanciation et paramétrage de la copule gaussienne
from SaucissonPerime.Copulas.elliptical.gaussian import GaussianCopula
from SaucissonPerime.Copulas.elliptical.student import StudentCopula
from SaucissonPerime.Copulas.visualization.copula_viz import plot_cdf, plot_pdf, plot_mpdf, plot_arbitrage_frontiers

cop = StudentCopula()
cop.parameters = np.array([0.8, 4.0])

# 2) Affichage de la CDF jointe en 3D et en contours
# plot_cdf(cop, plot_type='3d', Nsplit=60, cmap='coolwarm')
# plot_cdf(cop, plot_type='contour', Nsplit=60, levels=np.linspace(0.1, 0.9, 9), cmap='viridis')

# 3) Affichage de la PDF jointe en 3D et en contours
plot_pdf(cop, plot_type='3d', Nsplit=60, cmap='plasma')
plot_pdf(cop, plot_type='contour', Nsplit=100, log_scale=True, levels = [0.01,0.1,0.4,0.8,1.3,1.6])

# 4) PDF avec marges (normal et exponentielle) en contours
margins = [
    {'distribution': norm, 'loc': 0, 'scale': 1},
    {'distribution': norm, 'loc': 0, 'scale': 1}
]
plot_mpdf(cop, margins, plot_type='contour', Nsplit=80, levels=10, cmap='terrain')

# 5) Tracé du frontier d'arbitrage sans et avec points d'exemple
plot_arbitrage_frontiers(cop, alpha_low=0.05, alpha_high=0.95, levels=200)
u_pts = np.array([0.2, 0.5, 0.8])
v_pts = np.array([0.3, 0.6, 0.4])
plot_arbitrage_frontiers(cop, alpha_low=0.05, alpha_high=0.95, levels=200,
                         scatter=(u_pts, v_pts), scatter_alpha=0.4)


