from CopulaFurtif.core.copulas.domain.factories.copula_factory import CopulaFactory
from CopulaFurtif.core.copulas.application.use_cases.fit_copula import FitCopulaUseCase
# 1. Création via Factory
from CopulaFurtif.core.copulas.application.services.diagnostics_service import DiagnosticsService
from CopulaFurtif.core.copulas.infrastructure.visualization.matplotlib_visualizer import MatplotlibCopulaVisualizer
from CopulaFurtif.core.copulas.domain.estimation.estimation import pseudo_obs

import numpy as np
from scipy.stats import beta, lognorm

def generate_data_beta_lognorm(n=1000, rho=0.7):
    """
    Generate bivariate data with controlled dependence using Gaussian copula and marginals.
    """
    from CopulaFurtif.core.copulas.domain.models.elliptical.gaussian import GaussianCopula
    copula = GaussianCopula()
    copula.parameters = np.array([rho])
    uv = copula.sample(n)
    u, v = uv[:, 0], uv[:, 1]

    x = beta.ppf(u, a=2, b=5)
    y = lognorm.ppf(v, s=0.5)
    return [x,y]

copula = CopulaFactory.create("gaussian")

data = generate_data_beta_lognorm(n=1000, rho=0.7)

# 2. Fit
usecase = FitCopulaUseCase()
usecase.fit_cmle(data, copula)

# 3. Diagnostic
diag = DiagnosticsService()
print(diag.evaluate(data, copula))

# 4. Visualisation
visu = MatplotlibCopulaVisualizer()
visu.plot_residual_heatmap(copula, *pseudo_obs(data))
