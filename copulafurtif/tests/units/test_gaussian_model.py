import numpy as np
# from CopulaFurtif.core.copulas.infra.registry import register_all_copulas
from CopulaFurtif.core.copulas.domain.factories.copula_factory import CopulaFactory

import numpy as np
import pytest
from CopulaFurtif.core.copulas.domain.models.elliptical.gaussian import GaussianCopula


@pytest.fixture
def copula():
    return GaussianCopula()

def test_parameters_set_get(copula):
    copula.set_parameters([0.7])
    # copula.parameters = [0.7]
    assert np.allclose(copula.get_parameters(), [0.7])

def test_parameters_out_of_bounds(copula):
    with pytest.raises(ValueError):
        copula.set_parameters([1.5])
        # copula.parameters = [1.5]  # out of bounds

def test_cdf_pdf_sample_shapes(copula):
    u = np.linspace(0.01, 0.99, 10)
    v = np.linspace(0.01, 0.99, 10)
    cdf_vals = copula.get_cdf(u, v)
    pdf_vals = copula.get_pdf(u, v)
    assert len(cdf_vals) == len(pdf_vals) == 10

    samples = copula.sample(100)
    assert samples.shape == (100, 2)

def test_kendall_tau(copula):
    copula.set_parameters([0.6])
    # copula.parameters = [0.6]
    tau = copula.kendall_tau()
    expected = (2 / np.pi) * np.arcsin(0.6)
    assert np.isclose(tau, expected)

def test_tail_dependence(copula):
    assert copula.LTDC() == 0.0
    assert copula.UTDC() == 0.0

def test_partial_derivatives_and_conditionals(copula):
    u, v = 0.3, 0.7
    du = copula.partial_derivative_C_wrt_u(u, v)
    dv = copula.partial_derivative_C_wrt_v(u, v)
    assert 0 <= du <= 1
    assert 0 <= dv <= 1

    cond1 = copula.conditional_cdf_u_given_v(u, v)
    cond2 = copula.conditional_cdf_v_given_u(u, v)
    assert np.isclose(cond1, dv)
    assert np.isclose(cond2, du)

def test_iad_ad_disabled(copula):
    assert np.isnan(copula.IAD(None))
    assert np.isnan(copula.AD(None))