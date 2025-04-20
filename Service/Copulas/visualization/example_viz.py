import numpy as np

from Service.Copulas.archimedean.clayton import ClaytonCopula
from Service.Copulas.elliptical.gaussian import GaussianCopula
from Service.Copulas.visualization.copula_viz import plot_bivariate_contour, plot_conditional_contours

# exemple
my_copula = GaussianCopula()
my_copula.param = np.array([0.8])
plot_bivariate_contour(my_copula, Nsplit=100, levels=15, linestyles='--')

# pour le conditional plot
plot_conditional_contours(my_copula, Nsplit=50)
