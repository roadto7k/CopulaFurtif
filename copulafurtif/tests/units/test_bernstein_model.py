import math
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from CopulaFurtif.core.copulas.domain.models.exotic.bernstein import BernsteinCopula


# ----------------------------
# Helpers / strategies
# ----------------------------
@st.composite
def unit_interval(draw, eps=1e-3):
    return draw(st.floats(
        min_value=eps, max_value=1.0 - eps,
        exclude_min=True, exclude_max=True,
        allow_nan=False, allow_infinity=False,
        allow_subnormal=False,
        width=64,
    ))

def _clip01(x, eps=1e-12):
    return min(max(float(x), eps), 1.0 - eps)

def _clipped_f(f, x, y, eps=1e-12):
    return f(_clip01(x, eps), _clip01(y, eps))

def _finite_diff(f, x, y, h=1e-5, eps=1e-12):
    return (_clipped_f(f, x + h, y, eps) - _clipped_f(f, x - h, y, eps)) / (2.0 * h)

def _mixed_finite_diff(C, u, v, h=1e-5, eps=1e-12):
    return (
        _clipped_f(C, u + h, v + h, eps)
        - _clipped_f(C, u + h, v - h, eps)
        - _clipped_f(C, u - h, v + h, eps)
        + _clipped_f(C, u - h, v - h, eps)
    ) / (4.0 * h * h)

def _uniform_margin_P(m):
    """Independence Bernstein weights: uniform over grid cells => uniform margins."""
    P = np.full((m+1, m+1), 1.0 / ((m+1)*(m+1)))
    return P

@pytest.fixture(scope="module")
def copula_indep():
    m = 8
    P = _uniform_margin_P(m)
    return BernsteinCopula(P, validate=True)


# ----------------------------
# Weight matrix tests
# ----------------------------
def test_weights_shape_sum_and_margins(copula_indep):
    P = copula_indep.P
    m = copula_indep.m
    assert P.shape == (m+1, m+1)
    assert np.all(P >= 0)
    assert math.isclose(float(P.sum()), 1.0, rel_tol=0, abs_tol=1e-12)

    target = 1.0 / (m+1)
    assert np.allclose(P.sum(axis=0), target, atol=1e-12)
    assert np.allclose(P.sum(axis=1), target, atol=1e-12)

def test_set_weights_rejects_bad_sum():
    m = 6
    P = _uniform_margin_P(m)
    P = P * 0.9
    with pytest.raises(ValueError):
        BernsteinCopula(P, validate=True)

def test_no_scalar_parameters(copula_indep):
    params = copula_indep.get_parameters()
    assert isinstance(params, np.ndarray)
    assert params.size == 0
    with pytest.raises(ValueError):
        copula_indep.set_parameters([0.2])


# ----------------------------
# CDF tests
# ----------------------------
@given(u=unit_interval(), v=unit_interval())
def test_cdf_bounds(copula_indep, u, v):
    c = float(copula_indep.get_cdf(u, v))
    assert 0.0 <= c <= 1.0

@given(u=unit_interval(), v=unit_interval())
def test_cdf_monotone_in_u(copula_indep, u, v):
    u2 = min(u + 1e-3, 0.999)
    assert copula_indep.get_cdf(u, v) <= copula_indep.get_cdf(u2, v) + 1e-12

def test_cdf_boundaries(copula_indep):
    grid = np.linspace(0.01, 0.99, 25)
    for u in grid:
        assert math.isclose(copula_indep.get_cdf(u, 0.0), 0.0, abs_tol=1e-10)
        assert math.isclose(copula_indep.get_cdf(0.0, u), 0.0, abs_tol=1e-10)
        assert math.isclose(copula_indep.get_cdf(u, 1.0), u, abs_tol=1e-3)  # approx due to clipping
        assert math.isclose(copula_indep.get_cdf(1.0, u), u, abs_tol=1e-3)


# ----------------------------
# PDF tests
# ----------------------------
@given(u=unit_interval(), v=unit_interval())
def test_pdf_nonnegative(copula_indep, u, v):
    assert copula_indep.get_pdf(u, v) >= 0.0

@given(u=unit_interval(eps=0.05), v=unit_interval(eps=0.05))
@settings(max_examples=80)
def test_pdf_matches_mixed_derivative(copula_indep, u, v):
    C = lambda a, b: float(copula_indep.get_cdf(a, b))
    pdf_num = _mixed_finite_diff(C, u, v, h=1e-4)
    pdf_ana = float(copula_indep.get_pdf(u, v))
    assert math.isfinite(pdf_ana)
    # Bernstein densities can be a bit numerically rough at small m; keep tol moderate.
    assert math.isclose(pdf_ana, pdf_num, rel_tol=2e-2, abs_tol=2e-2)

def test_pdf_integrates_to_one(copula_indep):
    rng = np.random.default_rng(0)
    u = rng.uniform(1e-3, 1-1e-3, size=60000)
    v = rng.uniform(1e-3, 1-1e-3, size=60000)
    pdf = copula_indep.get_pdf(u, v)
    # Integral of density over unit square ~ mean(pdf) since area=1
    est = float(np.mean(pdf))
    assert math.isclose(est, 1.0, rel_tol=2e-2, abs_tol=2e-2)


# ----------------------------
# h-function tests
# ----------------------------
@given(u=unit_interval(), v=unit_interval())
def test_h_functions_are_probabilities(copula_indep, u, v):
    h1 = float(copula_indep.partial_derivative_C_wrt_u(u, v))
    h2 = float(copula_indep.partial_derivative_C_wrt_v(u, v))
    eps = 1e-12
    assert -eps <= h1 <= 1.0 + eps
    assert -eps <= h2 <= 1.0 + eps

@given(u=unit_interval(eps=0.05), v=unit_interval(eps=0.05))
@settings(max_examples=60)
def test_partial_derivative_matches_finite_diff(copula_indep, u, v):
    C = lambda a, b: float(copula_indep.get_cdf(a, b))
    h_num = _finite_diff(C, u, v, h=1e-4)
    h_ana = float(copula_indep.partial_derivative_C_wrt_u(u, v))
    assert math.isclose(h_ana, h_num, rel_tol=2e-2, abs_tol=2e-2)


# ----------------------------
# Independence sanity checks (P uniform)
# ----------------------------
@given(u=unit_interval(eps=0.05), v=unit_interval(eps=0.05))
def test_independence_cdf_equals_product(copula_indep, u, v):
    assert math.isclose(float(copula_indep.get_cdf(u, v)), u * v, rel_tol=2e-2, abs_tol=2e-2)

@given(u=unit_interval(eps=0.05), v=unit_interval(eps=0.05))
def test_independence_pdf_equals_one(copula_indep, u, v):
    assert math.isclose(float(copula_indep.get_pdf(u, v)), 1.0, rel_tol=2e-2, abs_tol=2e-2)

@given(u=unit_interval(eps=0.05), v=unit_interval(eps=0.05))
def test_independence_h_identity(copula_indep, u, v):
    assert math.isclose(float(copula_indep.partial_derivative_C_wrt_u(u, v)), v, rel_tol=3e-2, abs_tol=3e-2)
    assert math.isclose(float(copula_indep.partial_derivative_C_wrt_v(u, v)), u, rel_tol=3e-2, abs_tol=3e-2)


# ----------------------------
# Sampling sanity
# ----------------------------
def test_sample_shapes(copula_indep):
    data = copula_indep.sample(1000, rng=np.random.default_rng(0))
    assert data.shape == (1000, 2)
    assert np.all((data > 0) & (data < 1))

def test_blomqvist_beta_independence(copula_indep):
    # For independence, beta should be ~0
    b = copula_indep.blomqvist_beta()
    assert abs(b) < 2e-2


# ----------------------------
# Vectorization contract
# ----------------------------
def test_vectorised_inputs_are_pairwise_not_grid(copula_indep):
    u = np.array([0.2, 0.8])
    v = np.array([0.3, 0.7])
    cdf_vec = copula_indep.get_cdf(u, v)
    cdf_pair = np.array([copula_indep.get_cdf(float(u[0]), float(v[0])),
                         copula_indep.get_cdf(float(u[1]), float(v[1]))])
    assert cdf_vec.shape == (2,)
    assert np.allclose(cdf_vec, cdf_pair, rtol=1e-10, atol=1e-12)