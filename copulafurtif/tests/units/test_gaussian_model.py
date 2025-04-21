import numpy as np
# from CopulaFurtif.core.copulas.infra.registry import register_all_copulas
from CopulaFurtif.core.copulas.domain.factories.copula_factory import CopulaFactory

def test_factory_can_create_gaussian():
    # register_all_copulas()
    copula = CopulaFactory.create("gaussian")
    assert copula.type == "gaussian"

def test_sampling_is_in_unit_interval():
    copula = CopulaFactory.create("gaussian")
    samples = copula.sample(100)
    assert samples.shape == (100, 2)
    assert np.all((samples > 0) & (samples < 1))
