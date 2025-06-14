import pytest
# from CopulaFurtif.core.copulas.infrastructure.registry import register_all_copulas
from CopulaFurtif.core.copulas.domain.factories.copula_factory import CopulaFactory
from CopulaFurtif.core.copulas.application.services.fit_copula import FitCopulaUseCase
from CopulaFurtif.core.copulas.application.services.diagnostics_service import DiagnosticsService
from CopulaFurtif.core.DAO.generate_data_beta_lognorm import generate_data_beta_lognorm

@pytest.fixture(scope="module")
def gaussian_context():
    # register_all_copulas()
    data = generate_data_beta_lognorm(n=1000, rho=0.5)
    copula = CopulaFactory.create("gaussian")
    return copula, data

def test_fit_and_score(gaussian_context):
    copula, data = gaussian_context
    usecase = FitCopulaUseCase()
    result = usecase.fit_cmle(data, copula)
    assert result is not None
    assert copula.log_likelihood_ is not None

def test_diagnostics_output(gaussian_context):
    copula, data = gaussian_context
    diagnostics = DiagnosticsService()
    report = diagnostics.evaluate(data, copula)
    assert isinstance(report["AIC"], float)
    assert isinstance(report["BIC"], float)
