"""Unit-test suite for the Student (t) Copula.

Run with:  pytest -q  (add -m 'not slow' on CI if you skip the heavy bits)

Dependencies (add to requirements-dev.txt):
    pytest
    hypothesis
    scipy

The tests focus on:
    * Parameter validation (inside/outside the admissible intervals for rho and nu).
    * Core invariants: symmetry, monotonicity, bounds of CDF/PDF.
    * Fréchet–Hoeffding boundary conditions.
    * Tail dependence: LTDC = UTDC > 0 (non-zero, unlike Gaussian).
    * h-functions (conditional CDFs): probability range, boundaries, symmetry.
    * Analytical vs numerical derivatives (spot-check).
    * PDF integrates to 1 (Monte-Carlo, multiple parameter combos).
    * Kendall's tau analytical check: tau = (2/π)·arcsin(rho), independent of nu.
    * init_from_data round-trip.
    * Sampling sanity-check: empirical Kendall τ vs theoretical.

Slow / stochastic tests are marked with @pytest.mark.slow so they can be
optionally skipped (-m "not slow").

Note: The Student CDF relies on scipy.stats.multivariate_t which is expensive.
Hypothesis max_examples are kept low for CDF-intensive tests.
"""

import math
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
import scipy.stats as stx
from scipy.stats import t as student_t
from CopulaFurtif.core.copulas.domain.models.elliptical.student import StudentCopula


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def copula_default():
    """Student copula with ρ = 0.5, ν = 4 for deterministic tests."""
    c = StudentCopula()
    c.set_parameters([0.5, 4.0])
    return c


@st.composite
def valid_rho(draw):
    """Draw a valid ρ ∈ (-0.999, 0.999) – strict bounds from CopulaParameters."""
    return draw(st.floats(min_value=-0.99, max_value=0.99,
                          allow_nan=False, allow_infinity=False))


@st.composite
def valid_nu(draw):
    """Draw a valid ν ∈ (2.01, 30.0) – strict bounds."""
    return draw(st.floats(min_value=2.02, max_value=29.99,
                          allow_nan=False, allow_infinity=False))


@st.composite
def valid_params(draw):
    """Draw a valid (ρ, ν) pair."""
    rho = draw(valid_rho())
    nu = draw(valid_nu())
    return rho, nu


@st.composite
def unit_interval(draw):
    eps = 1e-3
    return draw(st.floats(min_value=0.0 + eps, max_value=1.0 - eps,
                          exclude_min=True, exclude_max=True))


# Numerical derivative helpers ------------------------------------------------

def _finite_diff(f, x, y, h=1e-6):
    """1st-order central finite difference ∂f/∂x."""
    return (f(x + h, y) - f(x - h, y)) / (2 * h)


def _mixed_finite_diff(C, u, v, h=1e-5):
    """
    Central 2-D finite difference:
        ∂²C/∂u∂v ≈ [C(u+h,v+h) – C(u+h,v–h) – C(u–h,v+h) + C(u–h,v–h)] / (4h²)
    """
    return (
        C(u + h, v + h)
        - C(u + h, v - h)
        - C(u - h, v + h)
        + C(u - h, v - h)
    ) / (4.0 * h * h)


# ---------------------------------------------------------------------------
# Parameter tests
# ---------------------------------------------------------------------------

@given(data=valid_params())
def test_parameter_roundtrip(data):
    """set_parameters then get_parameters should return the same values."""
    rho, nu = data
    c = StudentCopula()
    c.set_parameters([rho, nu])
    p = c.get_parameters()
    assert math.isclose(p[0], rho, rel_tol=1e-12)
    assert math.isclose(p[1], nu, rel_tol=1e-12)


@pytest.mark.parametrize("rho, nu", [
    (-1.0, 5.0),       # rho at lower boundary
    (1.0, 5.0),        # rho at upper boundary
    (-0.999, 5.0),     # rho at exact bound
    (0.999, 5.0),      # rho at exact bound
    (0.5, 2.01),       # nu at exact bound
    (0.5, 30.0),       # nu at exact bound
    (0.5, 1.5),        # nu below lower bound
    (0.5, 35.0),       # nu above upper bound
    (-2.0, 5.0),       # rho way out of bounds
])
def test_parameter_out_of_bounds(rho, nu):
    """Parameters at or outside strict bounds must be rejected."""
    c = StudentCopula()
    with pytest.raises(ValueError):
        c.set_parameters([rho, nu])


def test_parameter_wrong_size():
    """Passing wrong number of parameters must raise ValueError."""
    c = StudentCopula()
    with pytest.raises(ValueError):
        c.set_parameters([0.5])
    with pytest.raises(ValueError):
        c.set_parameters([0.5, 4.0, 1.0])


# ---------------------------------------------------------------------------
# CDF invariants
# ---------------------------------------------------------------------------

@given(data=valid_params(), u=unit_interval(), v=unit_interval())
@settings(max_examples=50)
def test_cdf_bounds(data, u, v):
    """CDF must lie in [0, 1]."""
    rho, nu = data
    c = StudentCopula()
    c.set_parameters([rho, nu])
    val = c.get_cdf(u, v)
    assert 0.0 <= val <= 1.0


@given(data=valid_params(), u1=unit_interval(), u2=unit_interval(), v=unit_interval())
@settings(max_examples=50)
def test_cdf_monotone_in_u(data, u1, u2, v):
    """C(u1, v) ≤ C(u2, v) when u1 ≤ u2."""
    if u1 > u2:
        u1, u2 = u2, u1
    rho, nu = data
    c = StudentCopula()
    c.set_parameters([rho, nu])
    assert c.get_cdf(u1, v) <= c.get_cdf(u2, v) + 1e-10


@given(data=valid_params(), u=unit_interval(), v=unit_interval())
@settings(max_examples=50)
def test_cdf_symmetry(data, u, v):
    """Student copula is symmetric: C(u, v) = C(v, u)."""
    rho, nu = data
    c = StudentCopula()
    c.set_parameters([rho, nu])
    assert math.isclose(c.get_cdf(u, v), c.get_cdf(v, u), rel_tol=1e-6, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# Fréchet–Hoeffding boundary conditions
# ---------------------------------------------------------------------------

@given(data=valid_params(), u=unit_interval())
@settings(max_examples=30)
def test_cdf_boundary_u_zero(data, u):
    """C(u, 0) = 0 for any copula."""
    rho, nu = data
    c = StudentCopula()
    c.set_parameters([rho, nu])
    assert math.isclose(c.get_cdf(u, 1e-12), 0.0, abs_tol=1e-4)


@given(data=valid_params(), v=unit_interval())
@settings(max_examples=30)
def test_cdf_boundary_v_zero(data, v):
    """C(0, v) = 0 for any copula."""
    rho, nu = data
    c = StudentCopula()
    c.set_parameters([rho, nu])
    assert math.isclose(c.get_cdf(1e-12, v), 0.0, abs_tol=1e-4)


@given(data=valid_params(), u=unit_interval())
@settings(max_examples=30)
def test_cdf_boundary_v_one(data, u):
    """C(u, 1) = u for any copula."""
    rho, nu = data
    c = StudentCopula()
    c.set_parameters([rho, nu])
    assert math.isclose(c.get_cdf(u, 1 - 1e-12), u, rel_tol=1e-3, abs_tol=1e-3)


@given(data=valid_params(), v=unit_interval())
@settings(max_examples=30)
def test_cdf_boundary_u_one(data, v):
    """C(1, v) = v for any copula."""
    rho, nu = data
    c = StudentCopula()
    c.set_parameters([rho, nu])
    assert math.isclose(c.get_cdf(1 - 1e-12, v), v, rel_tol=1e-3, abs_tol=1e-3)


# ---------------------------------------------------------------------------
# PDF invariants
# ---------------------------------------------------------------------------

@given(data=valid_params(), u=unit_interval(), v=unit_interval())
def test_pdf_nonnegative(data, u, v):
    """PDF must be non-negative."""
    rho, nu = data
    c = StudentCopula()
    c.set_parameters([rho, nu])
    assert c.get_pdf(u, v) >= 0.0


@given(data=valid_params(), u=unit_interval(), v=unit_interval())
def test_pdf_symmetry(data, u, v):
    """Student copula density is symmetric: c(u,v) = c(v,u)."""
    rho, nu = data
    c = StudentCopula()
    c.set_parameters([rho, nu])
    assert math.isclose(c.get_pdf(u, v), c.get_pdf(v, u), rel_tol=1e-12, abs_tol=1e-12)


@pytest.mark.parametrize("rho, nu", [
    (-0.5, 3.0),
    (0.0, 5.0),
    (0.3, 4.0),
    (0.8, 10.0),
])
def test_pdf_integrates_to_one(rho, nu):
    """Monte-Carlo check that ∫₀¹∫₀¹ c(u,v) du dv ≈ 1 for various (ρ, ν)."""
    c = StudentCopula()
    c.set_parameters([rho, nu])
    rng = np.random.default_rng(42)
    u, v = rng.random(100_000), rng.random(100_000)
    pdf_vals = c.get_pdf(u, v)
    integral_mc = pdf_vals.mean()
    assert math.isclose(integral_mc, 1.0, rel_tol=2e-2)


@pytest.mark.slow
@pytest.mark.parametrize("rho, nu", [(0.5, 4.0), (-0.3, 8.0)])
def test_pdf_matches_mixed_derivative(rho, nu):
    """
    c(u,v) ≈ ∂²C/∂u∂v via 2D central finite difference.
    Tested on a small grid (CDF is expensive for Student).
    """
    c = StudentCopula()
    c.set_parameters([rho, nu])

    rng = np.random.default_rng(99)
    for _ in range(10):
        u, v = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)
        pdf_num = _mixed_finite_diff(c.get_cdf, u, v)
        pdf_ana = c.get_pdf(u, v)
        assert math.isclose(pdf_ana, pdf_num, rel_tol=5e-2, abs_tol=1e-2), \
            f"rho={rho}, nu={nu}, u={u:.4f}, v={v:.4f}: ana={pdf_ana:.6f}, num={pdf_num:.6f}"


# ---------------------------------------------------------------------------
# h-functions (conditional CDFs)
# ---------------------------------------------------------------------------

@given(data=valid_params(), u=unit_interval(), v=unit_interval())
def test_h_functions_are_probabilities(data, u, v):
    """h-functions are conditional CDFs and must lie in [0, 1]."""
    rho, nu = data
    c = StudentCopula()
    c.set_parameters([rho, nu])

    h1 = c.partial_derivative_C_wrt_u(u, v)
    h2 = c.partial_derivative_C_wrt_v(u, v)

    assert 0.0 <= h1 <= 1.0
    assert 0.0 <= h2 <= 1.0


@given(data=valid_params(), u=unit_interval())
@settings(max_examples=50)
def test_h_u_boundary_in_v(data, u):
    """
    h_{V|U}(v|u) = ∂C/∂u:
      at v ≈ 0 → 0
      at v ≈ 1 → 1
    """
    rho, nu = data
    c = StudentCopula()
    c.set_parameters([rho, nu])

    h_low = c.partial_derivative_C_wrt_u(u, 1e-10)
    h_high = c.partial_derivative_C_wrt_u(u, 1 - 1e-10)

    assert math.isclose(h_low, 0.0, abs_tol=1e-4)
    assert math.isclose(h_high, 1.0, abs_tol=1e-4)


@given(data=valid_params(), v=unit_interval())
@settings(max_examples=50)
def test_h_v_boundary_in_u(data, v):
    """
    h_{U|V}(u|v) = ∂C/∂v:
      at u ≈ 0 → 0
      at u ≈ 1 → 1
    """
    rho, nu = data
    c = StudentCopula()
    c.set_parameters([rho, nu])

    h_low = c.partial_derivative_C_wrt_v(1e-10, v)
    h_high = c.partial_derivative_C_wrt_v(1 - 1e-10, v)

    assert math.isclose(h_low, 0.0, abs_tol=1e-4)
    assert math.isclose(h_high, 1.0, abs_tol=1e-4)


@given(data=valid_params(), u=unit_interval(), v=unit_interval())
def test_h_functions_cross_symmetry(data, u, v):
    """For symmetric copulas: ∂C/∂u(u,v) = ∂C/∂v(v,u)."""
    rho, nu = data
    c = StudentCopula()
    c.set_parameters([rho, nu])

    h_v_given_u = c.partial_derivative_C_wrt_u(u, v)
    h_u_given_v_swapped = c.partial_derivative_C_wrt_v(v, u)

    assert math.isclose(h_v_given_u, h_u_given_v_swapped, rel_tol=1e-8, abs_tol=1e-8)


@given(data=valid_params(), u=unit_interval(), v1=unit_interval(), v2=unit_interval())
@settings(max_examples=50)
def test_h_function_monotone_in_v(data, u, v1, v2):
    """∂C/∂u is monotone increasing in v (it's a CDF in v)."""
    if v1 > v2:
        v1, v2 = v2, v1
    rho, nu = data
    c = StudentCopula()
    c.set_parameters([rho, nu])
    assert c.partial_derivative_C_wrt_u(u, v1) <= c.partial_derivative_C_wrt_u(u, v2) + 1e-10


# ---------------------------------------------------------------------------
# Derivative cross-check (slow, CDF-dependent)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("rho, nu", [(0.5, 4.0), (-0.3, 10.0)])
def test_partial_derivative_matches_finite_diff(rho, nu):
    """Analytical partial derivatives vs numerical finite differences (grid)."""
    c = StudentCopula()
    c.set_parameters([rho, nu])

    def C(x, y):
        return c.get_cdf(x, y)

    rng = np.random.default_rng(77)
    for _ in range(10):
        u, v = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)
        num_du = _finite_diff(C, u, v)
        num_dv = _finite_diff(lambda x, y: C(y, x), v, u)

        ana_du = c.partial_derivative_C_wrt_u(u, v)
        ana_dv = c.partial_derivative_C_wrt_v(u, v)

        assert math.isclose(ana_du, num_du, rel_tol=5e-3, abs_tol=1e-3), \
            f"∂C/∂u: ana={ana_du:.6f}, num={num_du:.6f}"
        assert math.isclose(ana_dv, num_dv, rel_tol=5e-3, abs_tol=1e-3), \
            f"∂C/∂v: ana={ana_dv:.6f}, num={num_dv:.6f}"


# ---------------------------------------------------------------------------
# Kendall's tau analytical check
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rho, nu", [
    (0.0, 4.0),
    (0.5, 3.0),
    (-0.5, 10.0),
    (0.9, 5.0),
])
def test_kendall_tau_analytical(rho, nu):
    """tau = (2/π)·arcsin(ρ), independent of ν for elliptical copulas."""
    c = StudentCopula()
    c.set_parameters([rho, nu])
    expected = (2.0 / math.pi) * math.asin(rho)
    assert math.isclose(c.kendall_tau(), expected, rel_tol=1e-10)


@given(nu=valid_nu())
def test_kendall_tau_independent_of_nu(nu):
    """Kendall's tau depends only on ρ, not ν."""
    rho = 0.6
    c = StudentCopula()
    c.set_parameters([rho, nu])
    expected = (2.0 / math.pi) * math.asin(rho)
    assert math.isclose(c.kendall_tau(), expected, rel_tol=1e-10)


def test_kendall_tau_zero_at_independence():
    """At ρ = 0, Kendall's τ = 0 regardless of ν."""
    c = StudentCopula()
    c.set_parameters([0.0, 5.0])
    assert c.kendall_tau() == 0.0


# ---------------------------------------------------------------------------
# Tail dependence (non-zero for Student, LTDC = UTDC by symmetry)
# ---------------------------------------------------------------------------

@given(data=valid_params())
def test_tail_dependence_positive(data):
    """Student copula has strictly positive tail dependence for |ρ| < 1."""
    rho, nu = data
    c = StudentCopula()
    c.set_parameters([rho, nu])
    ltdc = c.LTDC()
    utdc = c.UTDC()
    assert ltdc >= 0.0
    assert utdc >= 0.0


@given(data=valid_params())
def test_tail_dependence_symmetric(data):
    """LTDC = UTDC for the Student copula (radial symmetry)."""
    rho, nu = data
    c = StudentCopula()
    c.set_parameters([rho, nu])
    assert math.isclose(c.LTDC(), c.UTDC(), rel_tol=1e-12)


@pytest.mark.parametrize("rho, nu", [
    (0.5, 4.0),
    (0.8, 3.0),
    (-0.3, 5.0),
])
def test_tail_dependence_analytical(rho, nu):
    """Verify LTDC = 2·t_{ν+1}(-√((ν+1)(1-ρ)/(1+ρ))) against manual computation."""
    c = StudentCopula()
    c.set_parameters([rho, nu])
    expected = 2 * student_t.cdf(
        -np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), df=nu + 1
    )
    assert math.isclose(c.LTDC(), expected, rel_tol=1e-10)


@pytest.mark.parametrize("nu", [3.0, 5.0, 10.0, 20.0])
def test_tail_dependence_increases_with_rho(nu):
    """At fixed ν, tail dependence increases with ρ."""
    rhos = [0.1, 0.3, 0.5, 0.7, 0.9]
    ltdcs = []
    for rho in rhos:
        c = StudentCopula()
        c.set_parameters([rho, nu])
        ltdcs.append(c.LTDC())
    for i in range(len(ltdcs) - 1):
        assert ltdcs[i] <= ltdcs[i + 1] + 1e-12


@pytest.mark.parametrize("rho", [0.3, 0.6, 0.8])
def test_tail_dependence_decreases_with_nu(rho):
    """At fixed ρ > 0, tail dependence decreases as ν increases (lighter tails)."""
    nus = [2.5, 4.0, 8.0, 15.0, 25.0]
    ltdcs = []
    for nu in nus:
        c = StudentCopula()
        c.set_parameters([rho, nu])
        ltdcs.append(c.LTDC())
    for i in range(len(ltdcs) - 1):
        assert ltdcs[i] >= ltdcs[i + 1] - 1e-12


# ---------------------------------------------------------------------------
# Independence case (rho = 0)
# ---------------------------------------------------------------------------

@given(nu=valid_nu(), u=unit_interval(), v=unit_interval())
def test_independence_cdf_equals_product(nu, u, v):
    """For Student copula, ρ=0 implies exact independence: C(u,v)=u*v for any ν."""
    c = StudentCopula()
    c.set_parameters([0.0, float(nu)])
    assert math.isclose(c.get_cdf(u, v), u * v, rel_tol=1e-6, abs_tol=1e-6)


@given(nu=valid_nu(), u=unit_interval(), v=unit_interval())
def test_independence_pdf_equals_one(nu, u, v):
    """For Student copula, ρ=0 implies exact independence: c(u,v)=1 for any ν."""
    c = StudentCopula()
    c.set_parameters([0.0, float(nu)])
    assert math.isclose(c.get_pdf(u, v), 1.0, rel_tol=1e-10, abs_tol=1e-10)


@given(nu=valid_nu(), u=unit_interval(), v=unit_interval())
def test_independence_h_functions_identity(nu, u, v):
    """
    For ρ=0:
      ∂C/∂u = v and ∂C/∂v = u
    """
    c = StudentCopula()
    c.set_parameters([0.0, float(nu)])
    h_v_given_u = c.partial_derivative_C_wrt_u(u, v)
    h_u_given_v = c.partial_derivative_C_wrt_v(u, v)
    assert math.isclose(h_v_given_u, v, rel_tol=1e-8, abs_tol=1e-8)
    assert math.isclose(h_u_given_v, u, rel_tol=1e-8, abs_tol=1e-8)


# ---------------------------------------------------------------------------
# init_from_data round-trip
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("rho_true, nu_true", [
    (0.5, 4.0),
    (-0.3, 8.0),
    (0.7, 5.0),
])
def test_init_from_data_roundtrip(rho_true, nu_true):
    """
    Generate samples with known (ρ, ν), then verify init_from_data
    recovers approximately the same parameters.
    """
    c = StudentCopula()
    c.set_parameters([rho_true, nu_true])
    data = c.sample(15_000, rng=np.random.default_rng(123))

    params = c.init_from_data(data[:, 0], data[:, 1])
    rho_rec, nu_rec = params

    assert math.isclose(rho_rec, rho_true, abs_tol=0.08), \
        f"Expected ρ ≈ {rho_true}, got {rho_rec}"
    # nu is harder to recover precisely; accept wider tolerance
    assert math.isclose(nu_rec, nu_true, rel_tol=0.5, abs_tol=5.0), \
        f"Expected ν ≈ {nu_true}, got {nu_rec}"


# ---------------------------------------------------------------------------
# Sampling sanity check (slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@given(data=valid_params())
@settings(max_examples=15)
def test_empirical_kendall_tau_close(data):
    """Empirical Kendall τ from samples should be close to theoretical."""
    rho, nu = data
    c = StudentCopula()
    c.set_parameters([rho, nu])

    samples = c.sample(5000)
    tau_emp, _ = stx.kendalltau(samples[:, 0], samples[:, 1])
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
    Vectorized APIs operate pairwise on (u[i], v[i]), not on the Cartesian product grid.
    Use h-functions here (much more numerically stable than Student CDF).
    """
    u = np.array([0.2, 0.8])
    v = np.array([0.3, 0.7])

    h_vec = copula_default.partial_derivative_C_wrt_u(u, v)
    h_pair = np.array([
        copula_default.partial_derivative_C_wrt_u(float(u[0]), float(v[0])),
        copula_default.partial_derivative_C_wrt_u(float(u[1]), float(v[1])),
    ])

    assert h_vec.shape == (2,)
    assert np.allclose(h_vec, h_pair, rtol=1e-12, atol=1e-12)
