"""Unit-test suite for the bivariate Clayton Archimedean Copula.

Run with:  pytest -q  (add -m 'not slow' on CI if you skip the heavy bits)

Dependencies (add to requirements-dev.txt):
    pytest
    hypothesis
    scipy

The tests focus on:
    * Parameter validation (inside/outside the admissible interval).
    * Core invariants: symmetry, monotonicity, bounds of CDF/PDF.
    * Fréchet–Hoeffding boundary conditions.
    * Tail dependence: λ_L = 2^{-1/θ} > 0, λ_U = 0.
    * h-functions (conditional CDFs): probability range, boundaries, symmetry, monotonicity.
    * Analytical vs numerical derivatives (spot-check).
    * PDF integrates to 1 (Monte-Carlo, multiple θ values).
    * Kendall's tau: τ = θ/(θ+2), sign, range, monotonicity.
    * Independence case (θ → 0+).
    * init_from_data round-trip.
    * Sampling sanity-check: empirical Kendall τ vs theoretical.

Clayton copula properties:
    - θ ∈ (0, 30) in this implementation (strict).
    - Asymmetric tail dependence: lower tail dependence only.
    - As θ → 0+, approaches independence.
    - As θ → ∞, approaches comonotonicity.

Slow / stochastic tests are marked with @pytest.mark.slow so they can be
optionally skipped (-m "not slow").
"""

import math
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
import scipy.stats as stx

from CopulaFurtif.core.copulas.domain.models.archimedean.clayton import ClaytonCopula


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def copula_default():
    """Clayton copula with θ = 2.0 for deterministic tests."""
    c = ClaytonCopula()
    c.set_parameters([2.0])
    return c


@st.composite
def valid_theta(draw):
    """θ ∈ (0.05, 29.99) — strictly inside bounds."""
    return draw(st.floats(
        min_value=0.05, max_value=29.99,
        exclude_min=True, exclude_max=True,
        allow_nan=False, allow_infinity=False
    ))


@st.composite
def unit_interval(draw):
    eps = 1e-3
    return draw(st.floats(min_value=eps, max_value=1.0 - eps,
                          exclude_min=True, exclude_max=True))


# Numerical derivative helpers ------------------------------------------------

def _clip01(x, eps=1e-12):
    return min(max(float(x), eps), 1.0 - eps)

def _clipped_C(C, u, v, eps=1e-12):
    return C(_clip01(u, eps), _clip01(v, eps))

def _finite_diff(C, u, v, h=1e-6):
    return (_clipped_C(C, u + h, v) - _clipped_C(C, u - h, v)) / (2*h)


def _mixed_finite_diff(C, u, v, h=1e-5):
    """
    Central 2-D finite difference:
        ∂²C/∂u∂v ≈ [C(u+h,v+h) – C(u+h,v-h) – C(u-h,v+h) + C(u-h,v-h)] / (4h²)
    """
    return (
            _clipped_C(C, u + h, v + h)
            - _clipped_C(C, u + h, v - h)
            - _clipped_C(C, u - h, v + h)
            + _clipped_C(C, u - h, v - h)
    ) / (4 * h * h)


# ---------------------------------------------------------------------------
# Parameter tests
# ---------------------------------------------------------------------------

@given(theta=valid_theta())
def test_parameter_roundtrip(theta):
    """set_parameters then get_parameters should return the same value."""
    c = ClaytonCopula()
    c.set_parameters([theta])
    assert math.isclose(c.get_parameters()[0], theta, rel_tol=1e-12)


@given(theta=st.one_of(
    st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=30.001, allow_nan=False, allow_infinity=False),
))
def test_parameter_out_of_bounds_extreme(theta):
    """Values ≤ 0 or > 30 must be rejected."""
    c = ClaytonCopula()
    with pytest.raises(ValueError):
        c.set_parameters([theta])


@pytest.mark.parametrize("theta", [0.0, -0.5, -1.0])
def test_parameter_at_lower_boundary_rejected(theta):
    """θ ≤ 0 must be rejected (Clayton requires θ > 0)."""
    c = ClaytonCopula()
    with pytest.raises(ValueError):
        c.set_parameters([theta])


def test_parameter_wrong_size():
    """Passing wrong number of parameters must raise ValueError."""
    c = ClaytonCopula()
    with pytest.raises(ValueError):
        c.set_parameters([1.0, 2.0])
    with pytest.raises(ValueError):
        c.set_parameters([])


# ---------------------------------------------------------------------------
# CDF invariants
# ---------------------------------------------------------------------------

@given(theta=valid_theta(), u=unit_interval(), v=unit_interval())
def test_cdf_bounds(theta, u, v):
    """CDF must lie in [0, 1]."""
    c = ClaytonCopula()
    c.set_parameters([theta])
    val = c.get_cdf(u, v)
    assert 0.0 <= val <= 1.0


@given(theta=valid_theta(), u1=unit_interval(), u2=unit_interval(), v=unit_interval())
def test_cdf_monotone_in_u(theta, u1, u2, v):
    """C(u1, v) ≤ C(u2, v) when u1 ≤ u2."""
    if u1 > u2:
        u1, u2 = u2, u1
    c = ClaytonCopula()
    c.set_parameters([theta])
    assert c.get_cdf(u1, v) <= c.get_cdf(u2, v) + 1e-12


@given(theta=valid_theta(), u=unit_interval(), v=unit_interval())
def test_cdf_symmetry(theta, u, v):
    """Clayton copula is symmetric: C(u, v) = C(v, u)."""
    c = ClaytonCopula()
    c.set_parameters([theta])
    assert math.isclose(c.get_cdf(u, v), c.get_cdf(v, u), rel_tol=1e-12)


# ---------------------------------------------------------------------------
# Fréchet–Hoeffding boundary conditions
# ---------------------------------------------------------------------------

@given(theta=valid_theta(), u=unit_interval())
def test_cdf_boundary_u_zero(theta, u):
    """C(u, 0) = 0 for any copula."""
    c = ClaytonCopula()
    c.set_parameters([theta])
    assert math.isclose(c.get_cdf(u, 1e-12), 0.0, abs_tol=1e-6)


@given(theta=valid_theta(), v=unit_interval())
def test_cdf_boundary_v_zero(theta, v):
    """C(0, v) = 0 for any copula."""
    c = ClaytonCopula()
    c.set_parameters([theta])
    assert math.isclose(c.get_cdf(1e-12, v), 0.0, abs_tol=1e-6)


@given(theta=valid_theta(), u=unit_interval())
def test_cdf_boundary_v_one(theta, u):
    """C(u, 1) = u for any copula."""
    c = ClaytonCopula()
    c.set_parameters([theta])
    assert math.isclose(c.get_cdf(u, 1 - 1e-12), u, rel_tol=1e-4, abs_tol=1e-4)


@given(theta=valid_theta(), v=unit_interval())
def test_cdf_boundary_u_one(theta, v):
    """C(1, v) = v for any copula."""
    c = ClaytonCopula()
    c.set_parameters([theta])
    assert math.isclose(c.get_cdf(1 - 1e-12, v), v, rel_tol=1e-4, abs_tol=1e-4)


# ---------------------------------------------------------------------------
# PDF invariants
# ---------------------------------------------------------------------------

@given(theta=valid_theta(), u=unit_interval(), v=unit_interval())
def test_pdf_nonnegative(theta, u, v):
    """PDF must be non-negative."""
    c = ClaytonCopula()
    c.set_parameters([theta])
    assert c.get_pdf(u, v) >= 0.0


@given(theta=valid_theta(), u=unit_interval(), v=unit_interval())
def test_pdf_symmetry(theta, u, v):
    """Clayton copula density is symmetric: c(u,v) = c(v,u)."""
    c = ClaytonCopula()
    c.set_parameters([theta])
    assert math.isclose(c.get_pdf(u, v), c.get_pdf(v, u), rel_tol=1e-10, abs_tol=1e-10)


@given(theta=valid_theta(), u=unit_interval(), v=unit_interval())
@settings(max_examples=100)
def test_pdf_matches_mixed_derivative(theta, u, v):
    """c(u,v) ≈ ∂²C/∂u∂v via 2D central finite difference."""
    c = ClaytonCopula()
    c.set_parameters([theta])

    pdf_num = _mixed_finite_diff(c.get_cdf, u, v)
    pdf_ana = c.get_pdf(u, v)

    assert math.isclose(pdf_ana, pdf_num, rel_tol=3e-2, abs_tol=1e-3), \
        f"θ={theta}, u={u:.4f}, v={v:.4f}: ana={pdf_ana}, num={pdf_num}"


@pytest.mark.parametrize("theta", [0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
def test_pdf_integrates_to_one(theta):
    """Monte-Carlo check that ∫₀¹∫₀¹ c(u,v) du dv ≈ 1 for various θ."""
    c = ClaytonCopula()
    c.set_parameters([theta])
    rng = np.random.default_rng(42)
    u, v = rng.random(100_000), rng.random(100_000)
    pdf_vals = c.get_pdf(u, v)
    integral_mc = pdf_vals.mean()
    assert math.isclose(integral_mc, 1.0, rel_tol=2e-2)


# ---------------------------------------------------------------------------
# h-functions (conditional CDFs)
# ---------------------------------------------------------------------------

@given(theta=valid_theta(), u=unit_interval(), v=unit_interval())
def test_h_functions_are_probabilities(theta, u, v):
    """h-functions are conditional CDFs and must lie in [0, 1]."""
    c = ClaytonCopula()
    c.set_parameters([theta])

    h1 = c.partial_derivative_C_wrt_u(u, v)
    h2 = c.partial_derivative_C_wrt_v(u, v)

    eps = 1e-12

    assert 0.0 <= h1 <= 1.0 + eps, f"∂C/∂u = {h1} out of [0,1]"
    assert 0.0 <= h2 <= 1.0 + eps, f"∂C/∂v = {h2} out of [0,1]"


@given(theta=valid_theta(), u=unit_interval())
@settings(max_examples=50)
def test_h_u_boundary_in_v(theta, u):
    """
    h_{V|U}(v|u) = ∂C/∂u:
      at v ≈ 0 → 0
      at v ≈ 1 → 1
    """
    c = ClaytonCopula()
    c.set_parameters([theta])

    h_low = c.partial_derivative_C_wrt_u(u, 1e-10)
    h_high = c.partial_derivative_C_wrt_u(u, 1 - 1e-10)

    assert math.isclose(h_low, 0.0, abs_tol=1e-4)
    assert math.isclose(h_high, 1.0, abs_tol=1e-4)


@given(theta=valid_theta(), v=unit_interval())
@settings(max_examples=50)
def test_h_v_boundary_in_u(theta, v):
    """
    h_{U|V}(u|v) = ∂C/∂v:
      at u ≈ 0 → 0
      at u ≈ 1 → 1
    """
    c = ClaytonCopula()
    c.set_parameters([theta])

    h_low = c.partial_derivative_C_wrt_v(1e-10, v)
    h_high = c.partial_derivative_C_wrt_v(1 - 1e-10, v)

    assert math.isclose(h_low, 0.0, abs_tol=1e-4)
    assert math.isclose(h_high, 1.0, abs_tol=1e-4)


@given(theta=valid_theta(), u=unit_interval(), v=unit_interval())
def test_h_functions_cross_symmetry(theta, u, v):
    """For symmetric copulas: ∂C/∂u(u,v) = ∂C/∂v(v,u)."""
    c = ClaytonCopula()
    c.set_parameters([theta])

    h_v_given_u = c.partial_derivative_C_wrt_u(u, v)
    h_u_given_v_swapped = c.partial_derivative_C_wrt_v(v, u)

    assert math.isclose(h_v_given_u, h_u_given_v_swapped, rel_tol=1e-8, abs_tol=1e-8)


@given(theta=valid_theta(), u=unit_interval(), v1=unit_interval(), v2=unit_interval())
@settings(max_examples=50)
def test_h_function_monotone_in_v(theta, u, v1, v2):
    """∂C/∂u is monotone increasing in v (it's a CDF in v)."""
    if v1 > v2:
        v1, v2 = v2, v1
    c = ClaytonCopula()
    c.set_parameters([theta])
    assert c.partial_derivative_C_wrt_u(u, v1) <= c.partial_derivative_C_wrt_u(u, v2) + 1e-10


# ---------------------------------------------------------------------------
# Derivative cross-check
# ---------------------------------------------------------------------------

@given(theta=valid_theta(), u=unit_interval(), v=unit_interval())
@settings(max_examples=100)
def test_partial_derivative_matches_finite_diff(theta, u, v):
    """Analytical partial derivatives vs numerical finite differences."""
    c = ClaytonCopula()
    c.set_parameters([theta])

    def C(x, y):
        return c.get_cdf(x, y)

    num_du = _finite_diff(C, u, v)
    num_dv = _finite_diff(lambda x, y: C(y, x), v, u)

    ana_du = c.partial_derivative_C_wrt_u(u, v)
    ana_dv = c.partial_derivative_C_wrt_v(u, v)

    assert math.isfinite(ana_du) and math.isfinite(ana_dv)
    assert math.isclose(ana_du, num_du, rel_tol=1e-3, abs_tol=1e-4)
    assert math.isclose(ana_dv, num_dv, rel_tol=1e-3, abs_tol=1e-4)


# ---------------------------------------------------------------------------
# Kendall's tau
# ---------------------------------------------------------------------------

@given(theta=valid_theta())
def test_kendall_tau_formula(theta):
    """τ = θ/(θ+2) for Clayton copula."""
    c = ClaytonCopula()
    c.set_parameters([theta])
    expected = theta / (theta + 2.0)
    assert math.isclose(c.kendall_tau(), expected, rel_tol=1e-12)


@given(theta=valid_theta())
def test_kendall_tau_positive(theta):
    """Clayton with θ > 0 implies τ > 0 (positive dependence only)."""
    c = ClaytonCopula()
    c.set_parameters([theta])
    assert c.kendall_tau() > 0.0


@given(theta=valid_theta())
def test_kendall_tau_range(theta):
    """Kendall's τ must lie in (0, 1) for Clayton with θ > 0."""
    c = ClaytonCopula()
    c.set_parameters([theta])
    tau = c.kendall_tau()
    assert 0.0 < tau < 1.0


def test_kendall_tau_monotone_in_theta():
    """τ(θ) = θ/(θ+2) is strictly increasing in θ for Clayton."""
    thetas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    taus = []
    for theta in thetas:
        c = ClaytonCopula()
        c.set_parameters([theta])
        taus.append(c.kendall_tau())
    for i in range(len(taus) - 1):
        assert taus[i] < taus[i + 1]


@pytest.mark.slow
@pytest.mark.parametrize("theta", [0.5, 2.0, 5.0, 10.0])
def test_kendall_tau_vs_empirical(theta):
    """
    Generate samples, estimate empirical Kendall τ,
    check it matches theoretical τ within statistical tolerance.
    """
    c = ClaytonCopula()
    c.set_parameters([theta])

    data = c.sample(10_000)
    tau_emp, _ = stx.kendalltau(data[:, 0], data[:, 1])
    tau_theo = c.kendall_tau()

    n = len(data)
    sigma = math.sqrt(2 * (2 * n + 5) / (9 * n * (n - 1)))
    assert math.isclose(tau_emp, tau_theo, abs_tol=4 * sigma)


# ---------------------------------------------------------------------------
# Tail dependence
# ---------------------------------------------------------------------------

@given(theta=valid_theta())
def test_tail_dependence_formulas(theta):
    """Clayton: λ_L = 2^{-1/θ} > 0, λ_U = 0."""
    c = ClaytonCopula()
    c.set_parameters([theta])

    expected_lt = 2.0 ** (-1.0 / theta)
    assert math.isclose(c.LTDC(), expected_lt, rel_tol=1e-12)
    assert c.UTDC() == 0.0


@given(theta=valid_theta())
def test_lower_tail_dependence_positive(theta):
    """Clayton always has strictly positive lower tail dependence."""
    c = ClaytonCopula()
    c.set_parameters([theta])
    assert c.LTDC() > 0.0


def test_lower_tail_dependence_increases_with_theta():
    """λ_L = 2^{-1/θ} increases with θ (stronger dependence → higher λ_L)."""
    thetas = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    ltdcs = []
    for theta in thetas:
        c = ClaytonCopula()
        c.set_parameters([theta])
        ltdcs.append(c.LTDC())
    for i in range(len(ltdcs) - 1):
        assert ltdcs[i] < ltdcs[i + 1]


def test_lower_tail_dependence_limits():
    """As θ → 0+, λ_L → 0. As θ → ∞, λ_L → 1."""
    c = ClaytonCopula()

    c.set_parameters([0.05])
    assert c.LTDC() < 0.01  # near zero

    c.set_parameters([29.0])
    assert c.LTDC() > 0.95  # near one


# ---------------------------------------------------------------------------
# Independence case (θ → 0+)
# ---------------------------------------------------------------------------

@given(u=unit_interval(), v=unit_interval())
def test_independence_cdf_near_product(u, v):
    """At small θ, Clayton copula → independence: C(u,v) ≈ u·v."""
    c = ClaytonCopula()
    c.set_parameters([0.05])
    assert math.isclose(c.get_cdf(u, v), u * v, rel_tol=5e-2, abs_tol=5e-2)


@given(u=unit_interval(), v=unit_interval())
def test_independence_pdf_near_one(u, v):
    """At small θ, copula density ≈ 1 on (0,1)²."""
    c = ClaytonCopula()
    c.set_parameters([0.001])
    assert math.isclose(c.get_pdf(u, v), 1.0, rel_tol=0.1, abs_tol=0.1)


@given(u=unit_interval(), v=unit_interval())
def test_independence_h_functions_identity(u, v):
    """At small θ: ∂C/∂u ≈ v and ∂C/∂v ≈ u."""
    c = ClaytonCopula()
    c.set_parameters([0.001])

    h_v_given_u = c.partial_derivative_C_wrt_u(u, v)
    h_u_given_v = c.partial_derivative_C_wrt_v(u, v)

    assert math.isclose(h_v_given_u, v, rel_tol=0.1, abs_tol=0.1)
    assert math.isclose(h_u_given_v, u, rel_tol=0.1, abs_tol=0.1)


# ---------------------------------------------------------------------------
# init_from_data round-trip
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("theta_true", [0.5, 2.0, 5.0, 10.0])
def test_init_from_data_roundtrip(theta_true):
    """
    Generate samples with known θ, then verify init_from_data
    recovers approximately the same θ.
    """
    c = ClaytonCopula()
    c.set_parameters([theta_true])
    data = c.sample(10_000, rng=np.random.default_rng(123))

    theta_recovered = c.init_from_data(data[:, 0], data[:, 1])

    assert math.isclose(theta_recovered[0], theta_true, rel_tol=0.15, abs_tol=1.0), \
        f"Expected θ ≈ {theta_true}, got {theta_recovered[0]}"


# ---------------------------------------------------------------------------
# Sampling sanity check (slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@given(theta=valid_theta())
@settings(max_examples=15)
def test_empirical_kendall_tau_close(theta):
    """Empirical Kendall τ from samples should be close to theoretical."""
    c = ClaytonCopula()
    c.set_parameters([theta])

    data = c.sample(5000)
    tau_emp, _ = stx.kendalltau(data[:, 0], data[:, 1])
    tau_theo = c.kendall_tau()

    sigma = math.sqrt(2 * (2 * 5000 + 5) / (9 * 5000 * 4999))
    assert math.isclose(tau_emp, tau_theo, abs_tol=4 * sigma)


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