import pytest
# from CopulaFurtif.core.copulas.infrastructure.registry import register_all_copulas
from CopulaFurtif.core.copulas.domain.factories.copula_factory import CopulaFactory


@pytest.mark.parametrize("copula_name", [
    "amh", "clayton", "fgm", "frank", "gumbel", "joe", "plackett", "galambos", "tawn"
])
def test_factory_creates_copula(copula_name):
    # register_all_copulas()
    copula = CopulaFactory.create(copula_name)
    assert copula.name.lower().startswith(copula_name)
    assert copula.parameters is not None
