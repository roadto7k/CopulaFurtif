import pytest
# from infra.registry import register_all_copulas
from CopulaFurtif.core.copulas.domain.factories.copula_factory import CopulaFactory, CopulaType
from CopulaFurtif.copulas import CopulaFitter
from CopulaFurtif.core.DAO.generate_data_beta_lognorm import generate_data_beta_lognorm


@pytest.mark.parametrize("copula_name", [CopulaType.CLAYTON, CopulaType.FRANK, CopulaType.JOE])
def test_cmle_on_archimedean(copula_name):
    """
    Test CMLE fitting on Archimedean copulas.

    Args:
        copula_name (str): Name of the Archimedean copula to test (e.g., 'clayton', 'frank', 'joe').

    Raises:
        AssertionError: If the CMLE result is None or copula.log_likelihood_ is not set.
    """

    # register_all_copulas()
    copula = CopulaFactory.create(copula_name)
    data = generate_data_beta_lognorm(n=1000, rho=0.6)
    result = CopulaFitter().fit_cmle(data, copula)
    assert result is not None
    assert copula.log_likelihood_ is not None
