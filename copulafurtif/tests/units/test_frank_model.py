import numpy as np
import pytest
from CopulaFurtif.core.copulas.domain.models.archimedean.frank import FrankCopula


@pytest.fixture
def copula():
    return FrankCopula()

def test_parameters_set_get(copula):
    copula.parameters = [5.0]
    assert np.allclose(copula.parameters, [5.0])

def test_parameters_out_of_bounds(copula):
    with pytest.raises(ValueError):
        copula.parameters = [40.0]  # out of bounds upper
    with pytest.raises(ValueError):
        copula.parameters = [-40.0]  # out of bounds lower

def test_cdf_pdf_sample_shapes(copula):
    u = np.linspace(0.01, 0.99, 10)
    v = np.linspace(0.01, 0.99, 10)
    cdf_vals = copula.get_cdf(u, v)
    pdf_vals = copula.get_pdf(u, v)
    assert len(cdf_vals) == len(pdf_vals) == 10

    samples = copula.sample(100)
    assert samples.shape == (100, 2)

def test_kendall_tau(copula):
    copula.parameters = [2.5]
    tau = copula.kendall_tau()
    assert -1 <= tau <= 1

def test_tail_dependence(copula):
    assert copula.LTDC() == 0.0
    assert copula.UTDC() == 0.0

def test_partial_derivatives_and_conditionals(copula):
    u, v = 0.3, 0.7
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