import numpy as np
import pytest
from CopulaFurtif.core.copulas.domain.models.archimedean.Tawn import TawnCopula


@pytest.fixture
def copula():
    return TawnCopula()

def test_parameters_set_get(copula):
    copula.parameters = [2.5, 0.7]
    assert np.allclose(copula.parameters, [2.5, 0.7])

def test_parameters_out_of_bounds(copula):
    with pytest.raises(ValueError):
        copula.parameters = [0.5, 0.5]  # theta too low
    with pytest.raises(ValueError):
        copula.parameters = [2.5, 1.5]  # delta too high

def test_cdf_sample_shapes(copula):
    u = np.linspace(0.1, 0.9, 10)
    v = np.linspace(0.1, 0.9, 10)
    cdf_vals = copula.get_cdf(u, v)
    assert len(cdf_vals) == 10

    samples = copula.sample(100)
    assert samples.shape == (100, 2)

def test_kendall_tau(copula):
    tau = copula.kendall_tau()
    theta, delta = copula.parameters
    expected = (theta * (1 - delta + delta)) / (theta + 2)
    assert np.isclose(tau, expected)

def test_tail_dependence(copula):
    assert copula.LTDC() == 0.0
    val = copula.UTDC()
    assert 0 < val < 1

def test_partial_derivatives_and_conditionals(copula):
    u, v = 0.4, 0.8
    du = copula.partial_derivative_C_wrt_u(u, v)
    dv = copula.partial_derivative_C_wrt_v(u, v)
    assert isinstance(du, (int, float, np.ndarray))
    assert isinstance(dv, (int, float, np.ndarray))

    cond1 = copula.conditional_cdf_u_given_v(u, v)
    cond2 = copula.conditional_cdf_v_given_u(u, v)
    assert np.isclose(cond1, dv)
    assert np.isclose(cond2, du)

def test_iad_ad_disabled(copula):
    assert np.isnan(copula.IAD(None))
    assert np.isnan(copula.AD(None))