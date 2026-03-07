"""Unit-test suite for the BB4 (Joe–Hu, 1996) Copula.

Run with:  pytest -q  (add -m 'not slow' on CI to skip heavy tests)

BB4 copula properties (Joe 2014, §4.20):
    C(u,v;θ,δ) = (u^{-θ}+v^{-θ}-1 - [(u^{-θ}-1)^{-δ}+(v^{-θ}-1)^{-δ}]^{-1/δ})^{-1/θ}

    Parameters: θ>0, δ>0 (open intervals; parent class enforces bounds).
    Symmetric copula: C(u,v) = C(v,u).
    λ_L = (2 - 2^{-1/δ})^{-1/θ},   λ_U = 2^{-1/δ}.
    τ = 1 - 2·B(1+1/θ, 1+1/δ),  B = beta function.
    β = 4·C(½,½) - 1,  C(½,½) = (2^{θ+1}-1 - 2^{-1/δ}(2^θ-1))^{-1/θ}.

Slow / stochastic tests are marked with @pytest.mark.slow.
"""

import math
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from scipy.stats import kendalltau
from scipy.special import beta as beta_fn

from CopulaFurtif.core.copulas.domain.models.archimedean.BB4 import BB4Copula


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

@st.composite
def valid_theta(draw):
    """θ ∈ (0.05, 8) — strictly inside valid range, numerically safe."""
    return draw(st.floats(min_value=0.05, max_value=8.0,
                          exclude_min=True, exclude_max=True,
                          allow_nan=False, allow_infinity=False,
                          allow_subnormal=False, width=64))


@st.composite
def valid_delta(draw):
    """δ ∈ (0.05, 8) — strictly inside valid range, numerically safe."""
    return draw(st.floats(min_value=0.05, max_value=8.0,
                          exclude_min=True, exclude_max=True,
                          allow_nan=False, allow_infinity=False,
                          allow_subnormal=False, width=64))


@st.composite
def unit_interval(draw, eps=1e-3):
    return draw(st.floats(min_value=eps, max_value=1.0 - eps,
                          exclude_min=True, exclude_max=True,
                          allow_nan=False, allow_infinity=False,
                          allow_subnormal=False, width=64))


@st.composite
def valid_theta_stable(draw):
    """θ ∈ (0.2, 5.0) — stable range for bisection-based sampling."""
    return draw(st.floats(min_value=0.2, max_value=5.0,
                          exclude_min=True, exclude_max=True,
                          allow_nan=False, allow_infinity=False,
                          allow_subnormal=False, width=64))


@st.composite
def valid_delta_stable(draw):
    """δ ∈ (0.2, 5.0) — stable range for bisection-based sampling."""
    return draw(st.floats(min_value=0.2, max_value=5.0,
                          exclude_min=True, exclude_max=True,
                          allow_nan=False, allow_infinity=False,
                          allow_subnormal=False, width=64))
    return min(max(float(x), eps), 1.0 - eps)

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


def _tau_theoretical(theta, delta):
    return 1.0 - 2.0 * beta_fn(1.0 + 1.0 / theta, 1.0 + 1.0 / delta)


def _beta_theoretical(theta, delta):
    lam_U = 2.0 ** (-1.0 / delta)
    inner = 2.0 ** (theta + 1.0) - 1.0 - lam_U * (2.0 ** theta - 1.0)
    c_half = max(inner, 1e-300) ** (-1.0 / theta)
    return 4.0 * c_half - 1.0


# ---------------------------------------------------------------------------
# Parameter tests
# ---------------------------------------------------------------------------

@given(theta=valid_theta(), delta=valid_delta())
def test_parameter_roundtrip(theta, delta):
    """set_parameters then get_parameters should return the same values."""
    c = BB4Copula()
    c.set_parameters([theta, delta])
    p = c.get_parameters()
    assert math.isclose(float(p[0]), theta, rel_tol=0, abs_tol=0)
    assert math.isclose(float(p[1]), delta, rel_tol=0, abs_tol=0)


@given(
    bad=st.one_of(
        st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
        st.just(0.0),
    ),
    ok=valid_theta(),
)
def test_parameter_out_of_bounds_extreme_theta(bad, ok):
    """θ ≤ 0 must be rejected."""
    c = BB4Copula()
    with pytest.raises(ValueError):
        c.set_parameters([bad, ok])


@given(
    bad=st.one_of(
        st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
        st.just(0.0),
    ),
    ok=valid_delta(),
)
def test_parameter_out_of_bounds_extreme_delta(bad, ok):
    """δ ≤ 0 must be rejected."""
    c = BB4Copula()
    with pytest.raises(ValueError):
        c.set_parameters([ok, bad])


@pytest.mark.parametrize("bad_th", [0.0, -1.0])
@pytest.mark.parametrize("bad_de", [0.0, -1.0])
def test_parameter_at_boundary_rejected(bad_th, bad_de):
    """Exact boundary / negative values must be rejected."""
    c = BB4Copula()
    with pytest.raises(ValueError):
        c.set_parameters([bad_th, 1.0])
    with pytest.raises(ValueError):
        c.set_parameters([1.0, bad_de])


def test_parameter_wrong_size():
    """Passing wrong number of parameters must raise ValueError."""
    c = BB4Copula()
    with pytest.raises(ValueError):
        c.set_parameters([1.0])
    with pytest.raises(ValueError):
        c.set_parameters([1.0, 1.5, 2.0])
    with pytest.raises(ValueError):
        c.set_parameters([])


# ---------------------------------------------------------------------------
# CDF invariants
# ---------------------------------------------------------------------------

@given(theta=valid_theta(), delta=valid_delta(), u=unit_interval(), v=unit_interval())
def test_cdf_bounds(theta, delta, u, v):
    """CDF must lie in [0, 1]."""
    c = BB4Copula()
    c.set_parameters([theta, delta])
    assert 0.0 <= float(c.get_cdf(u, v)) <= 1.0


@given(theta=valid_theta(), delta=valid_delta(), u=unit_interval(), v=unit_interval())
def test_cdf_monotone_in_u(theta, delta, u, v):
    """C(u, v) ≤ C(u2, v) for u2 > u."""
    c = BB4Copula()
    c.set_parameters([theta, delta])
    u2 = min(u + 1e-3, 0.999)
    assert c.get_cdf(u, v) <= c.get_cdf(u2, v) + 1e-12


@given(theta=valid_theta(), delta=valid_delta(), u=unit_interval(), v=unit_interval())
def test_cdf_symmetry(theta, delta, u, v):
    """BB4 is a symmetric copula: C(u,v) = C(v,u)."""
    c = BB4Copula()
    c.set_parameters([theta, delta])
    assert math.isclose(float(c.get_cdf(u, v)), float(c.get_cdf(v, u)), rel_tol=1e-12, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# Fréchet–Hoeffding boundary conditions
# ---------------------------------------------------------------------------

@given(theta=valid_theta(), delta=valid_delta(), u=unit_interval())
def test_cdf_boundary_u_zero(theta, delta, u):
    """C(u, 0) = 0 for any copula."""
    c = BB4Copula()
    c.set_parameters([theta, delta])
    assert math.isclose(float(c.get_cdf(u, 0.0)), 0.0, abs_tol=1e-12)


@given(theta=valid_theta(), delta=valid_delta(), v=unit_interval())
def test_cdf_boundary_v_zero(theta, delta, v):
    """C(0, v) = 0 for any copula."""
    c = BB4Copula()
    c.set_parameters([theta, delta])
    assert math.isclose(float(c.get_cdf(0.0, v)), 0.0, abs_tol=1e-12)


@given(theta=valid_theta(), delta=valid_delta(), u=unit_interval())
def test_cdf_boundary_v_one(theta, delta, u):
    """C(u, 1) = u for any copula."""
    c = BB4Copula()
    c.set_parameters([theta, delta])
    assert math.isclose(float(c.get_cdf(u, 1.0)), u, rel_tol=1e-8, abs_tol=1e-8)


@given(theta=valid_theta(), delta=valid_delta(), v=unit_interval())
def test_cdf_boundary_u_one(theta, delta, v):
    """C(1, v) = v for any copula."""
    c = BB4Copula()
    c.set_parameters([theta, delta])
    assert math.isclose(float(c.get_cdf(1.0, v)), v, rel_tol=1e-8, abs_tol=1e-8)


# ---------------------------------------------------------------------------
# PDF invariants
# ---------------------------------------------------------------------------

@given(theta=valid_theta(), delta=valid_delta(), u=unit_interval(), v=unit_interval())
def test_pdf_nonnegative(theta, delta, u, v):
    """PDF must be non-negative."""
    c = BB4Copula()
    c.set_parameters([theta, delta])
    assert float(c.get_pdf(u, v)) >= 0.0


@given(theta=valid_theta(), delta=valid_delta(), u=unit_interval(), v=unit_interval())
def test_pdf_symmetry(theta, delta, u, v):
    """c(u,v) = c(v,u) since C is symmetric."""
    c = BB4Copula()
    c.set_parameters([theta, delta])
    assert math.isclose(float(c.get_pdf(u, v)), float(c.get_pdf(v, u)), rel_tol=1e-8, abs_tol=1e-10)


@given(theta=valid_theta(), delta=valid_delta(), u=unit_interval(eps=0.05), v=unit_interval(eps=0.05))
@settings(max_examples=60, deadline=None)
def test_pdf_matches_mixed_derivative(theta, delta, u, v):
    """c(u,v) ≈ ∂²C/∂u∂v via 2D central finite difference."""
    c = BB4Copula()
    c.set_parameters([theta, delta])
    C = lambda a, b: float(c.get_cdf(a, b))
    pdf_num = _mixed_finite_diff(C, u, v, h=2e-4)
    pdf_ana = float(c.get_pdf(u, v))
    assert math.isfinite(pdf_ana)
    assert math.isclose(pdf_ana, pdf_num, rel_tol=5e-2, abs_tol=5e-2)


@pytest.mark.parametrize("theta,delta", [(1.0, 1.0), (2.0, 1.5), (0.5, 3.0)])
def test_pdf_integrates_to_one(theta, delta):
    """Monte-Carlo: E[c(U,V)] over unit square ≈ 1."""
    c = BB4Copula()
    c.set_parameters([theta, delta])
    rng = np.random.default_rng(0)
    u = rng.uniform(1e-3, 1.0 - 1e-3, 120_000)
    v = rng.uniform(1e-3, 1.0 - 1e-3, 120_000)
    est = float(np.mean(c.get_pdf(u, v)))
    assert math.isclose(est, 1.0, rel_tol=2e-2, abs_tol=2e-2)


# ---------------------------------------------------------------------------
# h-functions (conditional CDFs)
# ---------------------------------------------------------------------------

@given(theta=valid_theta(), delta=valid_delta(), u=unit_interval(), v=unit_interval())
def test_h_functions_are_probabilities(theta, delta, u, v):
    """h-functions are conditional CDFs and must lie in [0, 1]."""
    c = BB4Copula()
    c.set_parameters([theta, delta])
    eps = 1e-12
    assert -eps <= float(c.partial_derivative_C_wrt_u(u, v)) <= 1.0 + eps
    assert -eps <= float(c.partial_derivative_C_wrt_v(u, v)) <= 1.0 + eps


@given(theta=valid_theta(), delta=valid_delta(), u=unit_interval(eps=0.05))
@settings(max_examples=50, deadline=None)
def test_h_u_boundary_in_v(theta, delta, u):
    """∂C/∂u: at v~0 → 0, at v~1 → 1."""
    c = BB4Copula()
    c.set_parameters([theta, delta])
    assert math.isclose(float(c.partial_derivative_C_wrt_u(u, 1e-10)), 0.0, abs_tol=1e-4)
    assert math.isclose(float(c.partial_derivative_C_wrt_u(u, 1 - 1e-10)), 1.0, abs_tol=1e-4)


@given(theta=valid_theta(), delta=valid_delta(), v=unit_interval(eps=0.05))
@settings(max_examples=50, deadline=None)
def test_h_v_boundary_in_u(theta, delta, v):
    """∂C/∂v: at u~0 → 0, at u~1 → 1."""
    c = BB4Copula()
    c.set_parameters([theta, delta])
    assert math.isclose(float(c.partial_derivative_C_wrt_v(1e-10, v)), 0.0, abs_tol=1e-4)
    assert math.isclose(float(c.partial_derivative_C_wrt_v(1 - 1e-10, v)), 1.0, abs_tol=1e-4)


@given(theta=valid_theta(), delta=valid_delta(), u=unit_interval(), v=unit_interval())
def test_h_functions_cross_symmetry(theta, delta, u, v):
    """By symmetry: ∂C/∂u(u,v) = ∂C/∂v(v,u)."""
    c = BB4Copula()
    c.set_parameters([theta, delta])
    a = float(c.partial_derivative_C_wrt_u(u, v))
    b = float(c.partial_derivative_C_wrt_v(v, u))
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)


@given(theta=valid_theta(), delta=valid_delta(), u=unit_interval(), v1=unit_interval(), v2=unit_interval())
@settings(max_examples=50)
def test_h_function_monotone_in_v(theta, delta, u, v1, v2):
    """∂C/∂u is monotone increasing in v (it is a CDF in v)."""
    if v1 > v2:
        v1, v2 = v2, v1
    c = BB4Copula()
    c.set_parameters([theta, delta])
    assert c.partial_derivative_C_wrt_u(u, v1) <= c.partial_derivative_C_wrt_u(u, v2) + 1e-10


# ---------------------------------------------------------------------------
# Derivative cross-check
# ---------------------------------------------------------------------------

@given(theta=valid_theta(), delta=valid_delta(), u=unit_interval(eps=0.05), v=unit_interval(eps=0.05))
@settings(max_examples=80, deadline=None)
def test_partial_derivative_matches_finite_diff(theta, delta, u, v):
    """∂C/∂u should match central finite difference."""
    c = BB4Copula()
    c.set_parameters([theta, delta])
    C = lambda a, b: float(c.get_cdf(a, b))
    h_num = _finite_diff(C, u, v, h=1e-5)
    h_ana = float(c.partial_derivative_C_wrt_u(u, v))
    assert math.isclose(h_ana, h_num, rel_tol=2e-2, abs_tol=2e-2)


# ---------------------------------------------------------------------------
# Kendall's tau
# ---------------------------------------------------------------------------

@given(theta=valid_theta(), delta=valid_delta())
@settings(max_examples=20, deadline=None)
def test_kendall_tau_positive(theta, delta):
    """BB4 is a positively-dependent copula: τ > 0 for all θ,δ > 0."""
    c = BB4Copula()
    c.set_parameters([theta, delta])
    assert float(c.kendall_tau()) > 0.0


@given(theta=valid_theta(), delta=valid_delta())
@settings(max_examples=20, deadline=None)
def test_kendall_tau_range(theta, delta):
    """τ ∈ (0, 1] for any valid (θ, δ) — clips to 1 at extreme params."""
    c = BB4Copula()
    c.set_parameters([theta, delta])
    tau = float(c.kendall_tau())
    assert 0.0 < tau <= 1.0


def test_kendall_tau_monotone_in_theta():
    """For fixed δ, τ increases as θ increases (larger θ → C⁺ limit → τ→1)."""
    delta = 1.5
    thetas = [0.3, 0.7, 1.5, 3.0, 6.0]
    taus = []
    for th in thetas:
        c = BB4Copula()
        c.set_parameters([th, delta])
        taus.append(float(c.kendall_tau()))
    for i in range(len(taus) - 1):
        assert taus[i] < taus[i + 1], f"τ not increasing at θ={thetas[i]}: {taus}"


def test_kendall_tau_monotone_in_delta():
    """For fixed θ, τ increases as δ increases (larger δ → C⁺ limit → τ→1)."""
    theta = 2.0
    deltas = [0.3, 0.7, 1.5, 3.0, 6.0]
    taus = []
    for de in deltas:
        c = BB4Copula()
        c.set_parameters([theta, de])
        taus.append(float(c.kendall_tau()))
    for i in range(len(taus) - 1):
        assert taus[i] < taus[i + 1], f"τ not increasing at δ={deltas[i]}: {taus}"


@pytest.mark.slow
@given(theta=valid_theta_stable(), delta=valid_delta_stable())
@settings(max_examples=10, deadline=None)
def test_kendall_tau_vs_empirical(theta, delta):
    """Empirical Kendall τ from samples should be close to theoretical."""
    c = BB4Copula()
    c.set_parameters([theta, delta])
    data = c.sample(6_000, rng=np.random.default_rng(0))
    tau_emp = float(kendalltau(data[:, 0], data[:, 1]).correlation)
    tau_th = c.kendall_tau()
    assert math.isfinite(tau_emp)
    assert abs(tau_emp - tau_th) < 0.10


# ---------------------------------------------------------------------------
# Tail dependence
# ---------------------------------------------------------------------------

@given(theta=valid_theta(), delta=valid_delta())
def test_tail_dependence_formulas(theta, delta):
    """λ_L = (2-2^{-1/δ})^{-1/θ}, λ_U = 2^{-1/δ}."""
    c = BB4Copula()
    c.set_parameters([theta, delta])
    expected_lt = (2.0 - 2.0 ** (-1.0 / delta)) ** (-1.0 / theta)
    expected_ut = 2.0 ** (-1.0 / delta)
    assert math.isclose(float(c.LTDC()), expected_lt, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(float(c.UTDC()), expected_ut, rel_tol=1e-12, abs_tol=1e-12)


@given(theta=valid_theta(), delta=valid_delta())
def test_tail_dependence_positive(theta, delta):
    """Both λ_L and λ_U must be in (0, 1) for any valid (θ, δ)."""
    c = BB4Copula()
    c.set_parameters([theta, delta])
    assert 0.0 < float(c.LTDC()) < 1.0
    assert 0.0 < float(c.UTDC()) < 1.0


def test_lower_tail_dependence_increases_with_delta():
    """λ_L = (2-2^{-1/δ})^{-1/θ} increases as δ increases (for fixed θ)."""
    theta = 2.0
    deltas = [0.3, 0.7, 1.5, 3.0, 6.0]
    ltdcs = []
    for de in deltas:
        c = BB4Copula()
        c.set_parameters([theta, de])
        ltdcs.append(float(c.LTDC()))
    for i in range(len(ltdcs) - 1):
        assert ltdcs[i] < ltdcs[i + 1]


def test_upper_tail_dependence_increases_with_delta():
    """λ_U = 2^{-1/δ} increases as δ increases (δ→∞ → -1/δ→0 → λ_U→1)."""
    theta = 2.0
    deltas = [0.3, 0.7, 1.5, 3.0, 6.0]
    utdcs = []
    for de in deltas:
        c = BB4Copula()
        c.set_parameters([theta, de])
        utdcs.append(float(c.UTDC()))
    for i in range(len(utdcs) - 1):
        assert utdcs[i] < utdcs[i + 1]


# ---------------------------------------------------------------------------
# Blomqvist beta
# ---------------------------------------------------------------------------

@given(theta=valid_theta(), delta=valid_delta())
def test_blomqvist_beta_matches_definition(theta, delta):
    """β = 4·C(½, ½) - 1 must match the closed-form formula."""
    c = BB4Copula()
    c.set_parameters([theta, delta])
    beta_def = 4.0 * float(c.get_cdf(0.5, 0.5)) - 1.0
    beta_method = float(c.blomqvist_beta())
    assert math.isclose(beta_method, beta_def, rel_tol=1e-10, abs_tol=1e-10)


@given(theta=valid_theta(), delta=valid_delta())
def test_blomqvist_beta_closed_form(theta, delta):
    """Closed-form β = 4·(2^{θ+1}-1 - 2^{-1/δ}(2^θ-1))^{-1/θ} - 1."""
    c = BB4Copula()
    c.set_parameters([theta, delta])
    beta_expected = _beta_theoretical(theta, delta)
    beta_method = float(c.blomqvist_beta())
    assert math.isclose(beta_method, beta_expected, rel_tol=1e-10, abs_tol=1e-10)


# ---------------------------------------------------------------------------
# init_from_data round-trip (slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@given(theta=valid_theta_stable(), delta=valid_delta_stable())
@settings(max_examples=10, deadline=None)
def test_init_from_data_roundtrip(theta, delta):
    """init_from_data must return parameters within the valid domain."""
    c = BB4Copula()
    c.set_parameters([theta, delta])
    data = c.sample(8_000, rng=np.random.default_rng(123))

    c2 = BB4Copula()
    p0 = c2.init_from_data(data[:, 0], data[:, 1])
    c2.set_parameters(p0)

    th0, de0 = float(c2.get_parameters()[0]), float(c2.get_parameters()[1])
    assert th0 > 0.0
    assert de0 > 0.0


# ---------------------------------------------------------------------------
# Sampling sanity check (slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@given(theta=valid_theta_stable(), delta=valid_delta_stable())
@settings(max_examples=10, deadline=None)
def test_sampling_empirical_tau_close(theta, delta):
    """Empirical Kendall τ from samples should be close to theoretical."""
    c = BB4Copula()
    c.set_parameters([theta, delta])
    data = c.sample(6_000, rng=np.random.default_rng(0))
    tau_emp = float(kendalltau(data[:, 0], data[:, 1]).correlation)
    tau_th = c.kendall_tau()
    assert math.isfinite(tau_emp)
    assert abs(tau_emp - tau_th) < 0.10


# ---------------------------------------------------------------------------
# Shape checks (vectorised input)
# ---------------------------------------------------------------------------

def test_vectorised_shapes():
    """CDF, PDF, and sample must return arrays with correct shapes."""
    c = BB4Copula()
    c.set_parameters([2.0, 1.5])
    u = np.linspace(0.05, 0.95, 13)
    v = np.linspace(0.05, 0.95, 13)
    assert np.asarray(c.get_cdf(u, v)).shape == (13,)
    assert np.asarray(c.get_pdf(u, v)).shape == (13,)
    assert c.sample(256).shape == (256, 2)


def test_vectorised_inputs_are_pairwise_not_grid():
    """Vectorized get_cdf/get_pdf operate pairwise, not on the Cartesian grid."""
    c = BB4Copula()
    c.set_parameters([2.0, 1.5])
    u = np.array([0.2, 0.8])
    v = np.array([0.3, 0.7])

    cdf_vec = np.asarray(c.get_cdf(u, v))
    cdf_pair = np.array([
        float(c.get_cdf(float(u[0]), float(v[0]))),
        float(c.get_cdf(float(u[1]), float(v[1]))),
    ])
    assert cdf_vec.shape == (2,)
    assert np.allclose(cdf_vec, cdf_pair, rtol=1e-10, atol=1e-12)