"""Unit-test suite for the Galambos Extreme-Value Copula.

Run with:  pytest -q  (add -m 'not slow' on CI if you skip the heavy bits)

Dependencies (add to requirements-dev.txt):
    pytest
    hypothesis
    scipy

The tests focus on:
    * Parameter validation (inside/outside the admissible interval).
    * Core invariants: symmetry, monotonicity, bounds of CDF/PDF.
    * Fréchet–Hoeffding boundary conditions.
    * Tail dependence: λ_L = 0, λ_U = 2^{−1/δ} > 0.
    * h-functions (conditional CDFs): probability range, boundaries, symmetry, monotonicity.
    * Analytical vs numerical derivatives (spot-check).
    * PDF integrates to 1 (Monte-Carlo, multiple δ values).
    * Kendall's tau: sign, range, monotonicity, empirical check.
    * Independence case (δ → 0+).
    * init_from_data round-trip.
    * Sampling sanity-check: empirical Kendall τ vs theoretical.

Galambos copula properties:
    - δ ∈ (0, 50) in this implementation (strict).
    - Extreme-value copula with upper tail dependence only.
    - As δ → 0+, approaches independence.
    - As δ → ∞, approaches comonotonicity.
    - λ_U = 2^{−1/δ} (different formula from Gumbel/Joe).

Slow / stochastic tests are marked with @pytest.mark.slow so they can be
optionally skipped (-m "not slow").
"""

import math
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings
import scipy.stats as stx

from CopulaFurtif.core.copulas.domain.models.archimedean.galambos import GalambosCopula


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def copula_default():
    """Galambos copula with δ = 2.0 for deterministic tests."""
    c = GalambosCopula()
    c.set_parameters([2.0])
    return c


@st.composite
def valid_delta(draw):
    return draw(st.floats(
        min_value=0.0011, max_value=40.0,
        exclude_min=True, exclude_max=True,
        allow_nan=False, allow_infinity=False,
        allow_subnormal=False,
        width=64,
    ))


@st.composite
def unit_interval(draw, eps=1e-3):
    """Floats strictly inside (eps, 1-eps)."""
    return draw(st.floats(
        min_value=eps, max_value=1.0 - eps,
        exclude_min=True, exclude_max=True,
        allow_nan=False, allow_infinity=False
    ))


# Numerical derivative helpers ------------------------------------------------

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


# ---------------------------------------------------------------------------
# Parameter tests
# ---------------------------------------------------------------------------

@given(delta=valid_delta())
def test_parameter_roundtrip(delta):
    """set_parameters then get_parameters should return the same value."""
    c = GalambosCopula()
    c.set_parameters([delta])
    assert math.isclose(c.get_parameters()[0], delta, rel_tol=1e-12)


@given(delta=st.one_of(
    st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=55.0, allow_nan=False, allow_infinity=False),
))
def test_parameter_out_of_bounds_extreme(delta):
    """Values ≤ 0 or above upper bound must be rejected."""
    c = GalambosCopula()
    with pytest.raises(ValueError):
        c.set_parameters([delta])


@pytest.mark.parametrize("delta", [0.0, -0.5, -1.0])
def test_parameter_at_lower_boundary_rejected(delta):
    """δ ≤ 0 must be rejected (Galambos requires δ > 0 strict)."""
    c = GalambosCopula()
    with pytest.raises(ValueError):
        c.set_parameters([delta])


def test_parameter_wrong_size():
    """Passing wrong number of parameters must raise ValueError."""
    c = GalambosCopula()
    with pytest.raises(ValueError):
        c.set_parameters([1.0, 2.0])
    with pytest.raises(ValueError):
        c.set_parameters([])


# ---------------------------------------------------------------------------
# CDF invariants
# ---------------------------------------------------------------------------

@given(delta=valid_delta(), u=unit_interval(), v=unit_interval())
def test_cdf_bounds(delta, u, v):
    """CDF must lie in [0, 1]."""
    c = GalambosCopula()
    c.set_parameters([delta])
    val = c.get_cdf(u, v)
    assert 0.0 <= val <= 1.0


@given(delta=valid_delta(), u1=unit_interval(), u2=unit_interval(), v=unit_interval())
def test_cdf_monotone_in_u(delta, u1, u2, v):
    """C(u1, v) ≤ C(u2, v) when u1 ≤ u2."""
    if u1 > u2:
        u1, u2 = u2, u1
    c = GalambosCopula()
    c.set_parameters([delta])
    assert c.get_cdf(u1, v) <= c.get_cdf(u2, v) + 1e-12


@given(delta=valid_delta(), u=unit_interval(), v=unit_interval())
def test_cdf_symmetry(delta, u, v):
    """Galambos copula is symmetric: C(u, v) = C(v, u)."""
    c = GalambosCopula()
    c.set_parameters([delta])
    assert math.isclose(c.get_cdf(u, v), c.get_cdf(v, u), rel_tol=1e-12)


# ---------------------------------------------------------------------------
# Fréchet–Hoeffding boundary conditions
# ---------------------------------------------------------------------------

@given(delta=valid_delta(), u=unit_interval())
def test_cdf_boundary_u_zero(delta, u):
    """C(u, 0) = 0 for any copula."""
    c = GalambosCopula()
    c.set_parameters([delta])
    assert math.isclose(c.get_cdf(u, 1e-12), 0.0, abs_tol=1e-6)


@given(delta=valid_delta(), v=unit_interval())
def test_cdf_boundary_v_zero(delta, v):
    """C(0, v) = 0 for any copula."""
    c = GalambosCopula()
    c.set_parameters([delta])
    assert math.isclose(c.get_cdf(1e-12, v), 0.0, abs_tol=1e-6)


@given(delta=valid_delta(), u=unit_interval())
def test_cdf_boundary_v_one(delta, u):
    """C(u, 1) = u for any copula."""
    c = GalambosCopula()
    c.set_parameters([delta])
    assert math.isclose(c.get_cdf(u, 1 - 1e-12), u, rel_tol=1e-4, abs_tol=1e-4)


@given(delta=valid_delta(), v=unit_interval())
def test_cdf_boundary_u_one(delta, v):
    """C(1, v) = v for any copula."""
    c = GalambosCopula()
    c.set_parameters([delta])
    assert math.isclose(c.get_cdf(1 - 1e-12, v), v, rel_tol=1e-4, abs_tol=1e-4)


# ---------------------------------------------------------------------------
# PDF invariants
# ---------------------------------------------------------------------------

@given(delta=valid_delta(), u=unit_interval(), v=unit_interval())
def test_pdf_nonnegative(delta, u, v):
    """PDF must be non-negative (allow tiny IEEE-754 rounding)."""
    c = GalambosCopula()
    c.set_parameters([delta])
    pdf = c.get_pdf(u, v)
    assert pdf >= -1e-12, f"pdf={pdf}"


@given(delta=valid_delta(), u=unit_interval(), v=unit_interval())
def test_pdf_symmetry(delta, u, v):
    """Galambos copula density is symmetric: c(u,v) = c(v,u)."""
    c = GalambosCopula()
    c.set_parameters([delta])
    assert math.isclose(c.get_pdf(u, v), c.get_pdf(v, u), rel_tol=1e-10, abs_tol=1e-10)


@given(delta=valid_delta(), u=unit_interval(), v=unit_interval())
@settings(max_examples=100)
def test_pdf_matches_mixed_derivative(delta, u, v):
    """c(u,v) ≈ ∂²C/∂u∂v via 2D central finite difference."""
    c = GalambosCopula()
    c.set_parameters([delta])

    pdf_num = _mixed_finite_diff(c.get_cdf, u, v)
    pdf_ana = c.get_pdf(u, v)

    assert math.isclose(pdf_ana, pdf_num, rel_tol=3e-2, abs_tol=1e-3), \
        f"δ={delta}, u={u:.4f}, v={v:.4f}: ana={pdf_ana}, num={pdf_num}"


@pytest.mark.parametrize("delta", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
def test_pdf_integrates_to_one(delta):
    """Monte-Carlo check that ∫₀¹∫₀¹ c(u,v) du dv ≈ 1 for various δ."""
    c = GalambosCopula()
    c.set_parameters([delta])
    rng = np.random.default_rng(42)
    u, v = rng.random(100_000), rng.random(100_000)
    pdf_vals = c.get_pdf(u, v)
    integral_mc = pdf_vals.mean()
    assert math.isclose(integral_mc, 1.0, rel_tol=2e-2)


# ---------------------------------------------------------------------------
# h-functions (conditional CDFs)
# ---------------------------------------------------------------------------

@given(delta=valid_delta(), u=unit_interval(), v=unit_interval())
def test_h_functions_are_probabilities(delta, u, v):
    """h-functions are conditional CDFs and must lie in [0, 1]."""
    c = GalambosCopula()
    c.set_parameters([delta])

    h1 = c.partial_derivative_C_wrt_u(u, v)
    h2 = c.partial_derivative_C_wrt_v(u, v)

    eps = 1e-12
    assert -eps <= h1 <= 1.0 + eps
    assert -eps <= h2 <= 1.0 + eps


@given(delta=valid_delta(), u=unit_interval())
@settings(max_examples=50)
def test_h_u_boundary_in_v(delta, u):
    """
    h_{V|U}(v|u) = ∂C/∂u:
      at v ≈ 0 → 0
      at v ≈ 1 → 1
    """
    c = GalambosCopula()
    c.set_parameters([delta])

    h_low = c.partial_derivative_C_wrt_u(u, 1e-10)
    h_high = c.partial_derivative_C_wrt_u(u, 1 - 1e-10)

    assert math.isclose(h_low, 0.0, abs_tol=1e-4)
    assert math.isclose(h_high, 1.0, abs_tol=1e-4)


@given(delta=valid_delta(), v=unit_interval())
@settings(max_examples=50)
def test_h_v_boundary_in_u(delta, v):
    """
    h_{U|V}(u|v) = ∂C/∂v:
      at u ≈ 0 → 0
      at u ≈ 1 → 1
    """
    c = GalambosCopula()
    c.set_parameters([delta])

    h_low = c.partial_derivative_C_wrt_v(1e-10, v)
    h_high = c.partial_derivative_C_wrt_v(1 - 1e-10, v)

    assert math.isclose(h_low, 0.0, abs_tol=1e-4)
    assert math.isclose(h_high, 1.0, abs_tol=1e-4)


@given(delta=valid_delta(), u=unit_interval(), v=unit_interval())
def test_h_functions_cross_symmetry(delta, u, v):
    """For symmetric copulas: ∂C/∂u(u,v) = ∂C/∂v(v,u)."""
    c = GalambosCopula()
    c.set_parameters([delta])

    h_v_given_u = c.partial_derivative_C_wrt_u(u, v)
    h_u_given_v_swapped = c.partial_derivative_C_wrt_v(v, u)

    assert math.isclose(h_v_given_u, h_u_given_v_swapped, rel_tol=1e-8, abs_tol=1e-8)


@given(delta=valid_delta(), u=unit_interval(), v1=unit_interval(), v2=unit_interval())
@settings(max_examples=50)
def test_h_function_monotone_in_v(delta, u, v1, v2):
    """∂C/∂u is monotone increasing in v (it's a CDF in v)."""
    if v1 > v2:
        v1, v2 = v2, v1
    c = GalambosCopula()
    c.set_parameters([delta])
    assert c.partial_derivative_C_wrt_u(u, v1) <= c.partial_derivative_C_wrt_u(u, v2) + 1e-10


# ---------------------------------------------------------------------------
# Derivative cross-check
# ---------------------------------------------------------------------------

@given(delta=st.floats(min_value=0.05, max_value=6.0,
                       allow_nan=False, allow_infinity=False),
       u=unit_interval(), v=unit_interval())
@settings(max_examples=50)
def test_partial_derivative_matches_finite_diff(delta, u, v):
    """Analytical partial derivatives vs numerical finite differences (δ ≤ 6 for stability)."""
    c = GalambosCopula()
    c.set_parameters([delta])

    def C(x, y):
        return c.get_cdf(x, y)

    num_du = _finite_diff(C, u, v)
    num_dv = _finite_diff(lambda x, y: C(y, x), v, u)

    ana_du = c.partial_derivative_C_wrt_u(u, v)
    ana_dv = c.partial_derivative_C_wrt_v(u, v)

    assert math.isclose(ana_du, num_du, rel_tol=1e-2, abs_tol=1e-3)
    assert math.isclose(ana_dv, num_dv, rel_tol=1e-2, abs_tol=1e-3)


# ---------------------------------------------------------------------------
# Kendall's tau
# ---------------------------------------------------------------------------

@given(delta=valid_delta())
def test_kendall_tau_positive(delta):
    """Galambos with δ > 0 implies τ ≥ 0 (positive dependence only)."""
    c = GalambosCopula()
    c.set_parameters([delta])
    assert c.kendall_tau() >= 0.0


@given(delta=valid_delta())
def test_kendall_tau_range(delta):
    """Kendall's τ must lie in [0, 1) for Galambos with δ > 0."""
    c = GalambosCopula()
    c.set_parameters([delta])
    tau = c.kendall_tau()
    assert 0.0 <= tau < 1.0


def test_kendall_tau_monotone_in_delta():
    """τ(δ) is increasing in δ for Galambos copula."""
    deltas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    taus = []
    for delta in deltas:
        c = GalambosCopula()
        c.set_parameters([delta])
        taus.append(c.kendall_tau())
    for i in range(len(taus) - 1):
        assert taus[i] <= taus[i + 1] + 1e-12


def test_kendall_tau_near_zero_at_independence():
    """At small δ, Kendall's τ ≈ 0 (independence)."""
    c = GalambosCopula()
    c.set_parameters([0.0011])
    assert c.kendall_tau() < 0.05


@pytest.mark.slow
@pytest.mark.parametrize("delta", [0.5, 1.0, 2.0, 5.0])
def test_kendall_tau_vs_empirical(delta):
    """
    Generate samples, estimate empirical Kendall τ,
    check it matches theoretical τ within statistical tolerance.
    """
    c = GalambosCopula()
    c.set_parameters([delta])

    data = c.sample(10_000, rng=np.random.default_rng(42))
    tau_emp, _ = stx.kendalltau(data[:, 0], data[:, 1])
    tau_theo = c.kendall_tau()

    n = len(data)
    sigma = math.sqrt(2 * (2 * n + 5) / (9 * n * (n - 1)))
    assert math.isclose(tau_emp, tau_theo, abs_tol=4 * sigma + 0.02)


# ---------------------------------------------------------------------------
# Tail dependence
# ---------------------------------------------------------------------------

@given(delta=valid_delta())
def test_tail_dependence_formulas(delta):
    """Galambos: λ_L = 0, λ_U = 2^{−1/δ}."""
    c = GalambosCopula()
    c.set_parameters([delta])

    assert c.LTDC() == 0.0

    expected_ut = 2.0 ** (-1.0 / delta)
    assert math.isclose(c.UTDC(), expected_ut, rel_tol=1e-12)


@given(delta=valid_delta())
def test_upper_tail_dependence_positive(delta):
    """Galambos always has strictly positive upper tail dependence for δ > 0."""
    c = GalambosCopula()
    c.set_parameters([delta])
    assert c.UTDC() >= 0.0


def test_upper_tail_dependence_increases_with_delta():
    """λ_U = 2^{−1/δ} increases with δ (stronger dependence → higher λ_U)."""
    deltas = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    utdcs = []
    for delta in deltas:
        c = GalambosCopula()
        c.set_parameters([delta])
        utdcs.append(c.UTDC())
    for i in range(len(utdcs) - 1):
        assert utdcs[i] < utdcs[i + 1]


def test_upper_tail_dependence_limits():
    """As δ → 0+, λ_U → 0. As δ → ∞, λ_U → 1."""
    c = GalambosCopula()

    c.set_parameters([0.0011])
    assert c.UTDC() < 0.01  # near zero

    c.set_parameters([49.99])
    assert c.UTDC() > 0.95  # near one



# ---------------------------------------------------------------------------
# Blomqvist beta
# ---------------------------------------------------------------------------

@given(delta=valid_delta())
def test_blomqvist_beta_matches_closed_form(delta):
    """Galambos: β(δ) = 2^{2^{-1/δ}} - 1."""
    c = GalambosCopula()
    c.set_parameters([delta])

    beta = float(c.blomqvist_beta())

    d = float(delta)
    beta_cf = 2.0 ** (2.0 ** (-1.0 / d)) - 1.0

    assert math.isfinite(beta)
    assert -1.0 <= beta <= 1.0
    assert math.isclose(beta, beta_cf, rel_tol=1e-12, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# Independence case (δ → 0+)
# ---------------------------------------------------------------------------

@given(u=unit_interval(eps=0.05), v=unit_interval(eps=0.05))
def test_independence_cdf_near_product(u, v):
    """At small δ, Galambos copula → independence: C(u,v) ≈ u·v."""
    c = GalambosCopula()
    c.set_parameters([0.0011])
    assert math.isclose(c.get_cdf(u, v), u * v, rel_tol=0.1, abs_tol=0.1)


@given(u=unit_interval(eps=0.05), v=unit_interval(eps=0.05))
def test_independence_pdf_near_one(u, v):
    """At small δ, copula density ≈ 1 on (0,1)²."""
    c = GalambosCopula()
    c.set_parameters([0.0011])
    assert math.isclose(c.get_pdf(u, v), 1.0, rel_tol=0.15, abs_tol=0.15)


@given(u=unit_interval(eps=0.05), v=unit_interval(eps=0.05))
def test_independence_h_functions_identity(u, v):
    """At small δ: ∂C/∂u ≈ v and ∂C/∂v ≈ u."""
    c = GalambosCopula()
    c.set_parameters([0.0011])

    h_v_given_u = c.partial_derivative_C_wrt_u(u, v)
    h_u_given_v = c.partial_derivative_C_wrt_v(u, v)

    assert math.isclose(h_v_given_u, v, rel_tol=0.15, abs_tol=0.15)
    assert math.isclose(h_u_given_v, u, rel_tol=0.15, abs_tol=0.15)


# ---------------------------------------------------------------------------
# init_from_data round-trip
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("delta_true", [0.5, 1.0, 2.0, 5.0])
def test_init_from_data_roundtrip(delta_true):
    """
    Generate samples with known δ, then verify init_from_data
    recovers approximately the same δ.
    """
    c = GalambosCopula()
    c.set_parameters([delta_true])
    data = c.sample(10_000, rng=np.random.default_rng(123))

    delta_recovered = c.init_from_data(data[:, 0], data[:, 1])

    assert math.isclose(delta_recovered[0], delta_true, rel_tol=0.2, abs_tol=1.5), \
        f"Expected δ ≈ {delta_true}, got {delta_recovered[0]}"


# ---------------------------------------------------------------------------
# Sampling sanity check (slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@given(delta=valid_delta())
@settings(max_examples=15, deadline=None)
def test_empirical_kendall_tau_close(delta):
    """Empirical Kendall τ from samples should be close to theoretical."""
    c = GalambosCopula()
    c.set_parameters([delta])

    data = c.sample(5000, rng=np.random.default_rng(0))
    tau_emp, _ = stx.kendalltau(data[:, 0], data[:, 1])
    tau_theo = c.kendall_tau()

    sigma = math.sqrt(2 * (2 * 5000 + 5) / (9 * 5000 * 4999))
    assert math.isclose(tau_emp, tau_theo, abs_tol=4 * sigma + 0.02)


# ---------------------------------------------------------------------------
# Shape checks (vectorised input)
# ---------------------------------------------------------------------------

def test_vectorised_shapes(copula_default):
    """CDF, PDF, and sample must return arrays with correct shapes."""
    u = np.linspace(0.05, 0.95, 13)
    v = np.linspace(0.05, 0.95, 13)
    assert copula_default.get_cdf(u, v).shape == (13,)
    assert copula_default.get_pdf(u, v).shape == (13,)

    samples = copula_default.sample(256)
    assert samples.shape == (256, 2)


def test_vectorised_inputs_are_pairwise_not_grid(copula_default):
    """
    Vectorized get_cdf/get_pdf operate pairwise on (u[i], v[i]),
    not on the Cartesian product grid.
    """
    u = np.array([0.2, 0.8])
    v = np.array([0.3, 0.7])

    cdf_vec = copula_default.get_cdf(u, v)
    cdf_pair0 = copula_default.get_cdf(u[0], v[0])
    cdf_pair1 = copula_default.get_cdf(u[1], v[1])

    assert cdf_vec.shape == (2,)
    assert np.allclose(cdf_vec, np.array([cdf_pair0, cdf_pair1]))