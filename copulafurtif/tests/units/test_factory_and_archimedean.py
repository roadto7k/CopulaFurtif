import pytest
# from CopulaFurtif.core.copulas.infrastructure.registry import register_all_copulas
from CopulaFurtif.core.copulas.domain.factories.copula_factory import CopulaFactory, CopulaType


@pytest.mark.parametrize("copula_name", [
    CopulaType.AMH, CopulaType.CLAYTON, CopulaType.FGM, 
    CopulaType.FRANK, CopulaType.GUMBEL, CopulaType.JOE, 
    CopulaType.PLACKETT, CopulaType.GALAMBOS, CopulaType.TAWN
])
def test_factory_creates_copula(copula_name):
    # register_all_copulas()
    copula = CopulaFactory.create(copula_name)
    # assert copula.name.lower().startswith(copula_name)
    assert copula._parameters is not None
