"""Unit-test suite for the bivariate BB1 (Joe-Clayton) Archimedean Copula.

Run with:  pytest -q  (add -m 'not slow' on CI to skip heavy tests)

Dependencies (add to requirements-dev.txt):
    pytest
    hypothesis
    scipy

The tests focus on:
    * Parameter validation (inside/outside the admissible interval).
    * Core invariants: symmetry, bounds of CDF/PDF.
    * PDF integrates to 1 (Monte-Carlo).
    * Analytical vs numerical partial derivatives (spot-check).
    * PDF matches mixed second-order finite difference of the CDF.
    * Tail dependence: λ_L = 2^{-1/(δ·θ)}, λ_U = 2 - 2^{1/δ}.
    * Blomqvist's beta: β = 4·C(0.5, 0.5) - 1.
    * Kendall's tau: Monte-Carlo empirical vs theoretical.
    * init_from_data round-trip.

BB1 copula properties:
    - θ > 0, δ ≥ 1 (strict bounds in this implementation).
    - Both lower and upper tail dependence.
    - Special case δ = 1 → Clayton copula.
    - Special case θ → 0 → Gumbel copula.

Slow / stochastic tests are marked with @pytest.mark.slow so they can be
optionally skipped (-m "not slow").
"""

import math
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume
from scipy.stats import kendalltau

from CopulaFurtif.core.copulas.domain.models.archimedean.BB1 import BB1Copula


# =============================================================================
# Strategies
# =============================================================================

@st.composite
def valid_theta(draw):
    # in-model: theta in (0, +inf). For numerical tests, keep a reasonable range.
    return draw(st.floats(
        min_value=0.05, max_value=6.0,
        exclude_min=True, exclude_max=True,
        allow_nan=False, allow_infinity=False,
        allow_subnormal=False,
        width=64,
    ))

@st.composite
def valid_delta(draw):
    # in-model: delta in (1, +inf) with exclusive bounds
    return draw(st.floats(
        min_value=1.05, max_value=8.0,
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

theta_invalid = st.one_of(
    st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
    st.just(float("nan")),
    st.just(float("inf")),
    st.just(float("-inf")),
)

delta_invalid = st.one_of(
    st.floats(max_value=1.0, allow_nan=False, allow_infinity=False),
    st.just(float("nan")),
    st.just(float("inf")),
    st.just(float("-inf")),
)

# =============================================================================
# Finite differences (clipped) — canonical
# =============================================================================

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


# =============================================================================
# 1) PARAMETERS
# =============================================================================

@given(theta=valid_theta(), delta=valid_delta())
def test_parameter_roundtrip(theta, delta):
    c = BB1Copula()
    c.set_parameters([theta, delta])
    p = c.get_parameters()
    assert math.isclose(float(p[0]), float(theta), rel_tol=0, abs_tol=0)
    assert math.isclose(float(p[1]), float(delta), rel_tol=0, abs_tol=0)

@given(theta=theta_invalid, delta=valid_delta())
def test_parameter_out_of_bounds_extreme_theta(theta, delta):
    c = BB1Copula()
    with pytest.raises(ValueError):
        c.set_parameters([theta, float(delta)])

@given(theta=valid_theta(), delta=delta_invalid)
def test_parameter_out_of_bounds_extreme_delta(theta, delta):
    c = BB1Copula()
    with pytest.raises(ValueError):
        c.set_parameters([float(theta), delta])

@pytest.mark.parametrize("theta,delta", [
    (0.0, 2.0),
    (-1.0, 2.0),
    (1.0, 1.0),     # exclusive bound at 1
    (1.0, 0.5),
])
def test_parameter_at_boundary_rejected(theta, delta):
    c = BB1Copula()
    with pytest.raises(ValueError):
        c.set_parameters([theta, delta])

def test_parameter_wrong_size():
    c = BB1Copula()
    with pytest.raises(ValueError):
        c.set_parameters([])
    with pytest.raises(ValueError):
        c.set_parameters([1.0])
    with pytest.raises(ValueError):
        c.set_parameters([1.0, 2.0, 3.0])


# =============================================================================
# 2) CDF INVARIANTS
# =============================================================================

@given(theta=valid_theta(), delta=valid_delta(), u=unit_interval(), v=unit_interval())
def test_cdf_bounds(theta, delta, u, v):
    c = BB1Copula()
    c.set_parameters([theta, delta])
    val = float(c.get_cdf(u, v))
    assert 0.0 <= val <= 1.0

@given(theta=valid_theta(), delta=valid_delta(), u=unit_interval(), v=unit_interval())
def test_cdf_monotone_in_u(theta, delta, u, v):
    c = BB1Copula()
    c.set_parameters([theta, delta])
    u2 = min(float(u) + 1e-3, 0.999)
    assert c.get_cdf(u, v) <= c.get_cdf(u2, v) + 1e-12

@given(theta=valid_theta(), delta=valid_delta(), u=unit_interval(), v=unit_interval())
def test_cdf_symmetry(theta, delta, u, v):
    c = BB1Copula()
    c.set_parameters([theta, delta])
    assert math.isclose(float(c.get_cdf(u, v)), float(c.get_cdf(v, u)),
                        rel_tol=1e-12, abs_tol=1e-12)


# =============================================================================
# 3) FRÉCHET–HOEFFDING BOUNDARIES (4 tests)
# =============================================================================

@given(theta=valid_theta(), delta=valid_delta(), u=unit_interval())
def test_cdf_boundary_v_zero(theta, delta, u):
    c = BB1Copula(); c.set_parameters([theta, delta])
    assert math.isclose(float(c.get_cdf(u, 0.0)), 0.0, abs_tol=1e-12)

@given(theta=valid_theta(), delta=valid_delta(), v=unit_interval())
def test_cdf_boundary_u_zero(theta, delta, v):
    c = BB1Copula(); c.set_parameters([theta, delta])
    assert math.isclose(float(c.get_cdf(0.0, v)), 0.0, abs_tol=1e-12)

@given(theta=valid_theta(), delta=valid_delta(), u=unit_interval())
def test_cdf_boundary_v_one(theta, delta, u):
    c = BB1Copula(); c.set_parameters([theta, delta])
    assert math.isclose(float(c.get_cdf(u, 1.0)), float(u), abs_tol=2e-3)

@given(theta=valid_theta(), delta=valid_delta(), v=unit_interval())
def test_cdf_boundary_u_one(theta, delta, v):
    c = BB1Copula(); c.set_parameters([theta, delta])
    assert math.isclose(float(c.get_cdf(1.0, v)), float(v), abs_tol=2e-3)


# =============================================================================
# 4) PDF INVARIANTS
# =============================================================================

@given(theta=valid_theta(), delta=valid_delta(), u=unit_interval(), v=unit_interval())
def test_pdf_nonnegative(theta, delta, u, v):
    c = BB1Copula(); c.set_parameters([theta, delta])
    val = float(c.get_pdf(u, v))
    assert val >= 0.0
    assert math.isfinite(val)

@given(theta=valid_theta(), delta=valid_delta(), u=unit_interval(eps=0.01), v=unit_interval(eps=0.01))
def test_pdf_symmetry(theta, delta, u, v):
    c = BB1Copula(); c.set_parameters([theta, delta])
    assert math.isclose(float(c.get_pdf(u, v)), float(c.get_pdf(v, u)),
                        rel_tol=1e-10, abs_tol=1e-10)

@given(theta=valid_theta(), delta=valid_delta(),
       u=unit_interval(eps=0.05), v=unit_interval(eps=0.05))
@settings(max_examples=80, deadline=None)
def test_pdf_matches_mixed_derivative(theta, delta, u, v):
    c = BB1Copula(); c.set_parameters([theta, delta])
    C = lambda a, b: float(c.get_cdf(a, b))
    pdf_num = _mixed_finite_diff(C, u, v, h=2e-4)
    pdf_ana = float(c.get_pdf(u, v))
    assert math.isclose(pdf_ana, pdf_num, rel_tol=3e-2, abs_tol=3e-2)

@pytest.mark.parametrize("theta,delta", [(0.5, 1.2), (1.0, 2.0), (2.0, 2.0), (4.0, 3.0)])
def test_pdf_integrates_to_one(theta, delta):
    c = BB1Copula(); c.set_parameters([theta, delta])
    rng = np.random.default_rng(0)
    n = 200_000
    eps = 1e-4
    u = rng.uniform(eps, 1.0 - eps, size=n)
    v = rng.uniform(eps, 1.0 - eps, size=n)
    pdf = c.get_pdf(u, v)
    est = float(np.mean(pdf))
    assert math.isclose(est, 1.0, rel_tol=2e-2, abs_tol=2e-2)


# =============================================================================
# 5) H-FUNCTIONS
# =============================================================================

@given(theta=valid_theta(), delta=valid_delta(), u=unit_interval(), v=unit_interval())
def test_h_functions_are_probabilities(theta, delta, u, v):
    c = BB1Copula(); c.set_parameters([theta, delta])
    h1 = float(c.partial_derivative_C_wrt_u(u, v))
    h2 = float(c.partial_derivative_C_wrt_v(u, v))
    eps = 1e-12
    assert -eps <= h1 <= 1.0 + eps
    assert -eps <= h2 <= 1.0 + eps

@given(theta=valid_theta(), delta=valid_delta(), u=unit_interval())
def test_h_u_boundary_in_v(theta, delta, u):
    c = BB1Copula(); c.set_parameters([theta, delta])
    assert math.isclose(float(c.partial_derivative_C_wrt_u(u, 0.0)), 0.0, abs_tol=2e-12)
    assert math.isclose(float(c.partial_derivative_C_wrt_u(u, 1.0)), 1.0, abs_tol=2e-6)

@given(theta=valid_theta(), delta=valid_delta(), v=unit_interval())
def test_h_v_boundary_in_u(theta, delta, v):
    c = BB1Copula(); c.set_parameters([theta, delta])
    assert math.isclose(float(c.partial_derivative_C_wrt_v(0.0, v)), 0.0, abs_tol=2e-12)
    assert math.isclose(float(c.partial_derivative_C_wrt_v(1.0, v)), 1.0, abs_tol=2e-6)

@given(theta=valid_theta(), delta=valid_delta(), u=unit_interval(), v=unit_interval())
def test_h_functions_cross_symmetry(theta, delta, u, v):
    c = BB1Copula(); c.set_parameters([theta, delta])
    hu = float(c.partial_derivative_C_wrt_u(u, v))
    hv = float(c.partial_derivative_C_wrt_v(v, u))
    assert math.isclose(hu, hv, rel_tol=1e-10, abs_tol=1e-10)

@given(theta=valid_theta(), delta=valid_delta(), u=unit_interval(), v1=unit_interval(), v2=unit_interval())
def test_h_function_monotone_in_v(theta, delta, u, v1, v2):
    c = BB1Copula(); c.set_parameters([theta, delta])
    if v1 > v2:
        v1, v2 = v2, v1
    h1 = float(c.partial_derivative_C_wrt_u(u, v1))
    h2 = float(c.partial_derivative_C_wrt_u(u, v2))
    assert h1 <= h2 + 1e-10


# =============================================================================
# 6) DERIVATIVES
# =============================================================================

@given(theta=valid_theta(), delta=valid_delta(),
       u=unit_interval(eps=0.05), v=unit_interval(eps=0.05))
@settings(max_examples=60, deadline=None)
def test_partial_derivative_matches_finite_diff(theta, delta, u, v):
    c = BB1Copula(); c.set_parameters([theta, delta])
    C = lambda a, b: float(c.get_cdf(a, b))
    h_num = _finite_diff(C, u, v, h=2e-4)
    h_ana = float(c.partial_derivative_C_wrt_u(u, v))
    assert math.isclose(h_ana, h_num, rel_tol=3e-2, abs_tol=3e-2)


# =============================================================================
# 7) KENDALL'S TAU
# =============================================================================

@given(theta=valid_theta(), delta=valid_delta())
def test_kendall_tau_formula(theta, delta):
    c = BB1Copula(); c.set_parameters([theta, delta])
    tau = float(c.kendall_tau())
    tau_th = 1.0 - 2.0 / (float(delta) * (float(theta) + 2.0))
    assert math.isclose(tau, tau_th, rel_tol=1e-12, abs_tol=0.0)

@given(theta=valid_theta(), delta=valid_delta())
def test_kendall_tau_positive(theta, delta):
    c = BB1Copula(); c.set_parameters([theta, delta])
    assert float(c.kendall_tau()) >= 0.0

@given(theta=valid_theta(), delta=valid_delta())
def test_kendall_tau_range(theta, delta):
    c = BB1Copula(); c.set_parameters([theta, delta])
    tau = float(c.kendall_tau())
    assert 0.0 <= tau <= 1.0

def test_kendall_tau_monotone_in_theta():
    delta = 2.0
    thetas = [0.1, 0.3, 1.0, 3.0, 8.0]
    taus = []
    c = BB1Copula()
    for th in thetas:
        c.set_parameters([th, delta])
        taus.append(float(c.kendall_tau()))
    assert all(taus[i] <= taus[i+1] + 1e-12 for i in range(len(taus)-1))

@pytest.mark.slow
@pytest.mark.parametrize("theta,delta", [(0.4, 1.3), (1.0, 2.0), (2.0, 2.5), (4.0, 3.0)])
def test_kendall_tau_vs_empirical(theta, delta):
    c = BB1Copula(); c.set_parameters([theta, delta])
    n = 8000
    data = c.sample(n, rng=np.random.default_rng(0))
    tau_emp = float(kendalltau(data[:, 0], data[:, 1]).correlation)
    tau_th = float(c.kendall_tau())

    # 4σ band (same spirit as your other suites)
    var_tau = (2 * (2 * n + 5)) / (9 * n * (n - 1)) * (1 - tau_th ** 2) ** 2
    sigma = math.sqrt(var_tau)
    assert abs(tau_emp - tau_th) <= 4.0 * sigma + 0.02


# =============================================================================
# 8) TAIL DEPENDENCE
# =============================================================================

@given(theta=valid_theta(), delta=valid_delta())
def test_tail_dependence_formulas(theta, delta):
    c = BB1Copula(); c.set_parameters([theta, delta])
    lamL = float(c.LTDC())
    lamU = float(c.UTDC())
    lamL_th = float(2.0 ** (-1.0 / (float(delta) * float(theta))))
    lamU_th = float(2.0 - 2.0 ** (1.0 / float(delta)))
    assert math.isclose(lamL, lamL_th, rel_tol=1e-12, abs_tol=0.0)
    assert math.isclose(lamU, lamU_th, rel_tol=1e-12, abs_tol=0.0)

def test_tail_dependence_limits():
    c = BB1Copula()

    # delta -> 1+ => lamU -> 0
    c.set_parameters([2.0, 1.000001])
    assert float(c.UTDC()) < 1e-3

    # delta large => lamU -> 1
    c.set_parameters([2.0, 200.0])
    assert float(c.UTDC()) > 0.9

    # theta large & delta large => lamL -> 1
    c.set_parameters([200.0, 200.0])
    assert float(c.LTDC()) > 0.9


# =============================================================================
# 9) BLOMQVIST β
# =============================================================================

@given(theta=valid_theta(), delta=valid_delta())
def test_blomqvist_beta_matches_definition(theta, delta):
    c = BB1Copula(); c.set_parameters([theta, delta])
    beta_def = 4.0 * float(c.get_cdf(0.5, 0.5)) - 1.0
    beta = float(c.blomqvist_beta())
    assert math.isfinite(beta)
    assert math.isclose(beta, beta_def, rel_tol=5e-8, abs_tol=5e-8)


# =============================================================================
# 11) INIT FROM DATA  (@slow)
# =============================================================================

@pytest.mark.slow
@pytest.mark.parametrize("theta,delta", [(0.6, 1.4), (1.5, 2.0), (3.0, 3.0)])
def test_init_from_data_roundtrip(theta, delta):
    c = BB1Copula(); c.set_parameters([theta, delta])
    data = c.sample(12000, rng=np.random.default_rng(123))

    c2 = BB1Copula()
    res = c2.init_from_data(data[:, 0], data[:, 1])

    # support both patterns
    if isinstance(res, (list, tuple, np.ndarray)) and np.asarray(res).size == 2:
        c2.set_parameters(np.asarray(res, dtype=float))

    th2, de2 = map(float, c2.get_parameters())
    assert th2 > 0.0 and de2 > 1.0
    assert math.isfinite(th2) and math.isfinite(de2)

    # weak sanity: recovered tau & lamU not wildly off
    assert abs(float(c2.kendall_tau()) - float(c.kendall_tau())) < 0.12
    assert abs(float(c2.UTDC()) - float(c.UTDC())) < 0.15


# =============================================================================
# 12) SAMPLING (@slow)
# =============================================================================

@pytest.mark.slow
@pytest.mark.parametrize("theta,delta", [(0.6, 1.4), (1.5, 2.0), (3.0, 3.0)])
def test_empirical_kendall_tau_close(theta, delta):
    c = BB1Copula(); c.set_parameters([theta, delta])
    data = c.sample(10000, rng=np.random.default_rng(0))
    tau_emp = float(kendalltau(data[:, 0], data[:, 1]).correlation)
    tau_th = float(c.kendall_tau())
    assert abs(tau_emp - tau_th) < 0.06


# =============================================================================
# 13) VECTORIZATION
# =============================================================================

def test_vectorised_shapes():
    c = BB1Copula()
    c.set_parameters([2.0, 2.0])
    u = np.array([0.2, 0.8, 0.4])
    v = np.array([0.3, 0.7, 0.6])

    assert c.get_cdf(u, v).shape == (3,)
    assert c.get_pdf(u, v).shape == (3,)
    assert c.partial_derivative_C_wrt_u(u, v).shape == (3,)
    assert c.partial_derivative_C_wrt_v(u, v).shape == (3,)

def test_vectorised_inputs_are_pairwise_not_grid():
    c = BB1Copula()
    c.set_parameters([2.0, 2.0])
    u = np.array([0.2, 0.8])
    v = np.array([0.3, 0.7])

    cdf_vec = c.get_cdf(u, v)
    cdf_pair = np.array([c.get_cdf(float(u[0]), float(v[0])),
                         c.get_cdf(float(u[1]), float(v[1]))])

    assert cdf_vec.shape == (2,)
    assert np.allclose(cdf_vec, cdf_pair, rtol=1e-10, atol=1e-12)