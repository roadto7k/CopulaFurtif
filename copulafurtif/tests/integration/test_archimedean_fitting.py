import pytest
# from infra.registry import register_all_copulas
from CopulaFurtif.core.copulas.domain.factories.copula_factory import CopulaFactory
from CopulaFurtif.core.copulas.application.services.fit_copula import FitCopulaUseCase
from CopulaFurtif.core.DAO.generate_data_beta_lognorm import generate_data_beta_lognorm


@pytest.mark.parametrize("copula_name", ["clayton", "frank", "joe"])
def test_cmle_on_archimedean(copula_name):
    # register_all_copulas()
    copula = CopulaFactory.create(copula_name)
    data = generate_data_beta_lognorm(n=1000, rho=0.6)
    result = FitCopulaUseCase().fit_cmle(data, copula)
    assert result is not None
    assert copula.log_likelihood_ is not None
