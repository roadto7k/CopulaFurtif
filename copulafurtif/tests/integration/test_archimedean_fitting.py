import pytest
from infra.registry import register_all_copulas
from domain.factories.copula_factory import CopulaFactory
from application.use_cases.fit_copula import FitCopulaUseCase
from example.data_generator import generate_data_beta_lognorm


@pytest.mark.parametrize("copula_name", ["clayton", "frank", "joe"])
def test_cmle_on_archimedean(copula_name):
    register_all_copulas()
    copula = CopulaFactory.create(copula_name)
    data = generate_data_beta_lognorm(n=1000, rho=0.6)
    result = FitCopulaUseCase().fit_cmle(data, copula)
    assert result is not None
    assert copula.log_likelihood_ is not None
