import numpy as np
import pytest
from CopulaFurtif.core.copulas.domain.models.archimedean.plackett import PlackettCopula


@pytest.fixture
def copula():
    return PlackettCopula()

def test_parameters_set_get(copula):
    copula.parameters = [3.5]
    assert np.allclose(copula.parameters, [3.5])

def test_parameters_out_of_bounds(copula):
    with pytest.raises(ValueError):
        copula.parameters = [0.001]
    with pytest.raises(ValueError):
        copula.parameters = [150.0]

def test_cdf_pdf_sample_shapes(copula):
    u = np.linspace(0.1, 0.9, 10)
    v = np.linspace(0.1, 0.9, 10)
    cdf_vals = copula.get_cdf(u, v)
    pdf_vals = copula.get_pdf(u, v)
    assert len(cdf_vals) == 10
    assert len(pdf_vals) == 10

    samples = copula.sample(100)
    assert samples.shape == (100, 2)

def test_kendall_tau(copula):
    theta = copula.parameters[0]
    tau = copula.kendall_tau()
    expected = (theta - 1) / (theta + 1)
    assert np.isclose(tau, expected)

def test_tail_dependence(copula):
    assert copula.LTDC() == 0.0
    assert copula.UTDC() == 0.0

def test_partial_derivatives_and_conditionals(copula):
    u, v = 0.3, 0.6
    du = copula.partial_derivative_C_wrt_u(u, v)
    dv = copula.partial_derivative_C_wrt_v(u, v)
    assert 0 <= du <= 10
    assert 0 <= dv <= 10

    cond1 = copula.conditional_cdf_u_given_v(u, v)
    cond2 = copula.conditional_cdf_v_given_u(u, v)
    assert np.isclose(cond1, dv)
    assert np.isclose(cond2, du)

def test_iad_ad_disabled(copula):
    assert np.isnan(copula.IAD(None))
    assert np.isnan(copula.AD(None))