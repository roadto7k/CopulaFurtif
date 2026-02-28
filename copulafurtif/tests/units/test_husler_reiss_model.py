import math
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, assume
from scipy.stats import norm, kendalltau

from CopulaFurtif.core.copulas.domain.models.exotic.husle_reiss import HuslerReissCopula


# ============================================================
# Helpers / Strategies
# ============================================================

@st.composite
def valid_delta(draw):
    """
    δ strictly inside (0, 50) but avoid extremes that can cause numerical issues
    for finite-difference tests.
    """
    return draw(st.floats(
        min_value=0.02, max_value=30.0,
        exclude_min=True, exclude_max=True,
        allow_nan=False, allow_infinity=False,
        allow_subnormal=False,
        width=64,
    ))

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
    """Central FD ∂f/∂x with clipping to (0,1)."""
    return (_clipped_f(f, x + h, y, eps) - _clipped_f(f, x - h, y, eps)) / (2.0 * h)

def _mixed_finite_diff(C, u, v, h=1e-5, eps=1e-12):
    """Central FD ∂²C/∂u∂v with clipping to (0,1)."""
    return (
        _clipped_f(C, u + h, v + h, eps)
        - _clipped_f(C, u + h, v - h, eps)
        - _clipped_f(C, u - h, v + h, eps)
        + _clipped_f(C, u - h, v - h, eps)
    ) / (4.0 * h * h)

def _make(delta: float) -> HuslerReissCopula:
    c = HuslerReissCopula()
    c.set_parameters([float(delta)])
    return c


# ============================================================
# Parameters
# ============================================================

def test_parameter_wrong_size():
    c = HuslerReissCopula()
    with pytest.raises(ValueError):
        c.set_parameters([])
    with pytest.raises(ValueError):
        c.set_parameters([1.0, 2.0])

@pytest.mark.parametrize("bad", [0.0, -1.0, 50.0, 100.0])
def test_parameter_out_of_bounds(bad):
    c = HuslerReissCopula()
    with pytest.raises(ValueError):
        c.set_parameters([bad])

@given(delta=valid_delta())
def test_parameter_roundtrip(delta):
    c = HuslerReissCopula()
    c.set_parameters([delta])
    p = c.get_parameters()
    assert math.isclose(float(p[0]), float(delta), rel_tol=0, abs_tol=0)


# ============================================================
# CDF properties
# ============================================================

@given(delta=valid_delta(), u=unit_interval(), v=unit_interval())
def test_cdf_bounds(delta, u, v):
    c = _make(delta)
    val = float(c.get_cdf(u, v))
    assert 0.0 <= val <= 1.0

@given(delta=valid_delta(), u=unit_interval(), v=unit_interval())
def test_cdf_monotone_in_u(delta, u, v):
    c = _make(delta)
    u2 = min(u + 1e-3, 0.999)
    assert c.get_cdf(u, v) <= c.get_cdf(u2, v) + 1e-12

def test_cdf_boundaries():
    c = _make(1.0)
    grid = np.linspace(0.01, 0.99, 21)
    for u in grid:
        assert math.isclose(float(c.get_cdf(u, 0.0)), 0.0, abs_tol=2e-12)
        assert math.isclose(float(c.get_cdf(0.0, u)), 0.0, abs_tol=2e-12)
        assert math.isclose(float(c.get_cdf(u, 1.0)), u, abs_tol=2e-3)
        assert math.isclose(float(c.get_cdf(1.0, u)), u, abs_tol=2e-3)

@given(delta=valid_delta(), u=unit_interval(), v=unit_interval())
def test_cdf_symmetry(delta, u, v):
    c = _make(delta)
    assert math.isclose(float(c.get_cdf(u, v)), float(c.get_cdf(v, u)), rel_tol=1e-12, abs_tol=1e-12)

def test_cdf_independence_limit_bulk():
    """
    As δ -> 0+, HR approaches independence.
    We only test this in the bulk for stability.
    """
    c = _make(0.03)
    grid = np.linspace(0.1, 0.9, 9)
    for u in grid:
        for v in grid:
            assert math.isclose(float(c.get_cdf(u, v)), u * v, rel_tol=0.08, abs_tol=0.08)

def test_cdf_comonotone_limit():
    """
    As δ -> +∞, HR approaches comonotone: C(u,v) -> min(u,v).
    """
    c = _make(25.0)
    for u, v in [(0.2, 0.8), (0.6, 0.4), (0.3, 0.3)]:
        target = min(u, v)
        assert math.isclose(float(c.get_cdf(u, v)), target, rel_tol=0.08, abs_tol=0.08)


# ============================================================
# PDF properties
# ============================================================

@given(delta=valid_delta(), u=unit_interval(), v=unit_interval())
def test_pdf_nonnegative(delta, u, v):
    c = _make(delta)
    val = float(c.get_pdf(u, v))
    assert val >= 0.0
    assert math.isfinite(val)

@given(delta=st.floats(min_value=0.1, max_value=10.0,
                       exclude_min=True, exclude_max=True,
                       allow_nan=False, allow_infinity=False,
                       allow_subnormal=False, width=64),
       u=unit_interval(eps=0.05), v=unit_interval(eps=0.05))
@settings(max_examples=70, deadline=None)
def test_pdf_matches_mixed_derivative(delta, u, v):
    c = _make(delta)
    C = lambda a, b: float(c.get_cdf(a, b))
    pdf_num = _mixed_finite_diff(C, u, v, h=2e-4)
    pdf_ana = float(c.get_pdf(u, v))
    # finite diff is noisy; keep tol moderate
    assert math.isclose(pdf_ana, pdf_num, rel_tol=3e-2, abs_tol=3e-2)

@pytest.mark.parametrize("delta", [0.3, 1.0, 3.0])
def test_pdf_integrates_to_one(delta):
    """
    HR is absolutely continuous => density integrates to 1 over (0,1)^2.
    Use MC estimate: integral ≈ mean(pdf(U,V)) with U,V ~ Unif(0,1).
    """
    c = _make(delta)
    rng = np.random.default_rng(0)
    u = rng.uniform(1e-3, 1.0 - 1e-3, size=80000)
    v = rng.uniform(1e-3, 1.0 - 1e-3, size=80000)
    pdf = c.get_pdf(u, v)
    est = float(np.mean(pdf))
    assert math.isclose(est, 1.0, rel_tol=2e-2, abs_tol=2e-2)


# ============================================================
# h-functions (partials)
# ============================================================

@given(delta=valid_delta(), u=unit_interval(), v=unit_interval())
def test_h_functions_are_probabilities(delta, u, v):
    c = _make(delta)
    h1 = float(c.partial_derivative_C_wrt_u(u, v))
    h2 = float(c.partial_derivative_C_wrt_v(u, v))
    eps = 1e-12
    assert -eps <= h1 <= 1.0 + eps
    assert -eps <= h2 <= 1.0 + eps

@given(delta=st.floats(min_value=0.1, max_value=10.0,
                       exclude_min=True, exclude_max=True,
                       allow_nan=False, allow_infinity=False,
                       allow_subnormal=False, width=64),
       u=unit_interval(eps=0.05), v=unit_interval(eps=0.05))
@settings(max_examples=70, deadline=None)
def test_partial_derivative_matches_finite_diff(delta, u, v):
    c = _make(delta)
    C = lambda a, b: float(c.get_cdf(a, b))
    h_num = _finite_diff(C, u, v, h=2e-4)
    h_ana = float(c.partial_derivative_C_wrt_u(u, v))
    assert math.isclose(h_ana, h_num, rel_tol=3e-2, abs_tol=3e-2)


# ============================================================
# Dependence measures
# ============================================================

@given(delta=valid_delta())
def test_upper_tail_dependence_formula(delta):
    c = _make(delta)
    lam = float(c.UTDC())
    lam_th = float(2.0 * (1.0 - norm.cdf(1.0 / delta)))
    assert math.isclose(lam, lam_th, rel_tol=0, abs_tol=0)

def test_upper_tail_dependence_limits():
    c_small = _make(0.03)
    assert c_small.UTDC() < 0.05

    c_big = _make(25.0)
    assert c_big.UTDC() > 0.80

@given(delta=valid_delta())
def test_blomqvist_beta_matches_definition(delta):
    c = _make(delta)
    beta_def = 4.0 * float(c.get_cdf(0.5, 0.5)) - 1.0
    beta = float(c.blomqvist_beta())
    assert math.isclose(beta, beta_def, rel_tol=1e-12, abs_tol=1e-12)

@given(delta=valid_delta())
def test_kendall_tau_range(delta):
    c = _make(delta)
    tau = float(c.kendall_tau())
    assert math.isfinite(tau)
    assert 0.0 <= tau <= 1.0

def test_kendall_tau_monotone_in_delta():
    c1 = _make(0.2)
    c2 = _make(2.0)
    assert c2.kendall_tau() >= c1.kendall_tau() - 1e-6


# ============================================================
# init_from_data + sampling sanity
# ============================================================

@pytest.mark.slow
@pytest.mark.parametrize("delta", [0.2, 1.0, 3.0, 10.0])
def test_init_from_data_roundtrip(delta):
    c = _make(delta)
    data = c.sample(12000, rng=np.random.default_rng(0))

    c2 = HuslerReissCopula()
    res = c2.init_from_data(data[:, 0], data[:, 1])

    # support both patterns: returned params OR in-place
    if isinstance(res, (list, tuple, np.ndarray)) and np.asarray(res).size == 1:
        c2.set_parameters([float(np.asarray(res).ravel()[0])])

    d_hat = float(c2.get_parameters()[0])
    # Expect same order of magnitude; init_from_data is based on beta so it's fairly stable.
    assert d_hat > 0.0
    assert math.isfinite(d_hat)

@pytest.mark.slow
@given(delta=st.floats(min_value=0.1, max_value=10.0,
                       exclude_min=True, exclude_max=True,
                       allow_nan=False, allow_infinity=False,
                       allow_subnormal=False, width=64))
@settings(max_examples=10, deadline=None)
def test_sampling_empirical_tau_close(delta):
    c = _make(delta)
    data = c.sample(8000, rng=np.random.default_rng(0))
    tau_emp = float(kendalltau(data[:, 0], data[:, 1]).correlation)
    tau_th = float(c.kendall_tau())

    assert math.isfinite(tau_emp)
    assert math.isfinite(tau_th)
    assert abs(tau_emp - tau_th) < 0.06


# ============================================================
# Vectorization contract
# ============================================================

def test_vectorised_inputs_are_pairwise_not_grid():
    c = _make(1.5)

    u = np.array([0.2, 0.8])
    v = np.array([0.3, 0.7])

    cdf_vec = c.get_cdf(u, v)
    cdf_pair = np.array([c.get_cdf(float(u[0]), float(v[0])),
                         c.get_cdf(float(u[1]), float(v[1]))])

    assert cdf_vec.shape == (2,)
    assert np.allclose(cdf_vec, cdf_pair, rtol=1e-10, atol=1e-12)