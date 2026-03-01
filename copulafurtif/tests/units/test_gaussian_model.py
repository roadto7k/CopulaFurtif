"""Unit-test suite for the Gaussian Copula.

Run with:  pytest -q  (add -m 'not slow' on CI if you skip the heavy bits)

Dependencies (add to requirements-dev.txt):
    pytest
    hypothesis
    scipy

The tests focus on:
    * Parameter validation (inside/outside the admissible interval).
    * Core invariants: symmetry, monotonicity, bounds of CDF/PDF.
    * Fréchet–Hoeffding boundary conditions.
    * Tail-dependence values (analytical zeros for Gaussian).
    * Analytical vs numerical derivatives (spot-check).
    * PDF integrates to 1 (Monte-Carlo, multiple rho values).
    * Kendall's tau analytical check.
    * init_from_data round-trip.
    * Sampling sanity-check: empirical Kendall τ vs theoretical.

Slow / stochastic tests are marked with @pytest.mark.slow so they can be
optionally skipped (-m "not slow").
"""

import math
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
import scipy.stats as stx
from CopulaFurtif.core.copulas.domain.models.elliptical.gaussian import GaussianCopula
import math


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def copula_default():
    """Gaussian copula with ρ = 0.3 for deterministic tests."""
    c = GaussianCopula()
    c.set_parameters([0.3])
    return c


@st.composite
def valid_rho(draw):
    """Draw a valid ρ ∈ (-1, 1) – strict, matching CopulaParameters bounds."""
    return draw(st.floats(min_value=-0.999, max_value=0.999,
                          allow_nan=False, allow_infinity=False))


@st.composite
def unit_interval(draw):
    eps = 1e-3
    return draw(st.floats(min_value=0.0 + eps, max_value=1.0 - eps,
                          exclude_min=True, exclude_max=True))


# Numerical derivative helpers ------------------------------------------------

def _clip01(x, eps=1e-12):
    return min(max(float(x), eps), 1.0 - eps)

def _clipped_f(f, x, y, eps=1e-12):
    return f(_clip01(x, eps), _clip01(y, eps))


def _finite_diff(f, x, y, h=1e-5, eps=1e-12):
    """1st-order central finite difference ∂f/∂x with clipping to (0,1)."""
    return (_clipped_f(f, x + h, y, eps) - _clipped_f(f, x - h, y, eps)) / (2.0 * h)


def _mixed_finite_diff(C, u, v, h=1e-5, eps=1e-12):
    """
    Central 2-D finite difference with clipping to (0,1):
      ∂²C/∂u∂v ≈ [C(u+h,v+h) – C(u+h,v-h) – C(u-h,v+h) + C(u-h,v-h)] / (4h²)
    """
    return (
        _clipped_f(C, u + h, v + h, eps)
        - _clipped_f(C, u + h, v - h, eps)
        - _clipped_f(C, u - h, v + h, eps)
        + _clipped_f(C, u - h, v - h, eps)
    ) / (4.0 * h * h)


# ---------------------------------------------------------------------------
# Parameter tests
# ---------------------------------------------------------------------------

@given(rho=valid_rho())
def test_parameter_roundtrip(rho):
    """set_parameters then get_parameters should return the same value."""
    c = GaussianCopula()
    c.set_parameters([rho])
    assert math.isclose(c.get_parameters()[0], rho, rel_tol=1e-12)


@given(rho=st.one_of(
    st.floats(max_value=-1.0, allow_infinity=False, allow_nan=False),
    st.floats(min_value=1.0, allow_infinity=False, allow_nan=False),
))
def test_parameter_out_of_bounds_extreme(rho):
    """Values ≤ -1 or ≥ 1 must be rejected (bounds are (-1, 1) strict)."""
    c = GaussianCopula()
    with pytest.raises(ValueError):
        c.set_parameters([rho])


@pytest.mark.parametrize("rho", [-1.0, 1.0])
def test_parameter_at_boundary_rejected(rho):
    """Exact boundary values -1 and 1 must be rejected (strict inequality)."""
    c = GaussianCopula()
    with pytest.raises(ValueError):
        c.set_parameters([rho])


def test_parameter_wrong_size():
    """Passing wrong number of parameters must raise ValueError."""
    c = GaussianCopula()
    with pytest.raises(ValueError):
        c.set_parameters([0.3, 0.5])
    with pytest.raises(ValueError):
        c.set_parameters([])


# ---------------------------------------------------------------------------
# CDF invariants
# ---------------------------------------------------------------------------

@given(rho=valid_rho(), u=unit_interval(), v=unit_interval())
@settings(deadline=None)
def test_cdf_bounds(rho, u, v):
    """CDF must lie in [0, 1]."""
    c = GaussianCopula()
    c.set_parameters([rho])
    val = c.get_cdf(u, v)
    assert 0.0 <= val <= 1.0


@given(rho=valid_rho(), u1=unit_interval(), u2=unit_interval(), v=unit_interval())
def test_cdf_monotone_in_u(rho, u1, u2, v):
    """C(u1, v) ≤ C(u2, v) when u1 ≤ u2."""
    if u1 > u2:
        u1, u2 = u2, u1
    c = GaussianCopula()
    c.set_parameters([rho])
    assert c.get_cdf(u1, v) <= c.get_cdf(u2, v) + 1e-12


@given(rho=valid_rho(), u=unit_interval(), v=unit_interval())
def test_cdf_symmetry(rho, u, v):
    """Gaussian copula is symmetric: C(u, v) = C(v, u)."""
    c = GaussianCopula()
    c.set_parameters([rho])
    assert math.isclose(c.get_cdf(u, v), c.get_cdf(v, u), rel_tol=1e-9)

@given(rho=valid_rho(), u=unit_interval(), v=unit_interval())
def test_pdf_symmetry(rho, u, v):
    """Gaussian copula density is symmetric: c(u,v)=c(v,u)."""
    c = GaussianCopula()
    c.set_parameters([rho])
    assert math.isclose(c.get_pdf(u, v), c.get_pdf(v, u), rel_tol=1e-12, abs_tol=1e-12)

# ---------------------------------------------------------------------------
# Fréchet–Hoeffding boundary conditions
# ---------------------------------------------------------------------------

@given(rho=valid_rho(), u=unit_interval())
def test_cdf_boundary_u_zero(rho, u):
    """C(u, 0) = 0 for any copula."""
    c = GaussianCopula()
    c.set_parameters([rho])
    assert math.isclose(c.get_cdf(u, 1e-12), 0.0, abs_tol=1e-6)


@given(rho=valid_rho(), v=unit_interval())
def test_cdf_boundary_v_zero(rho, v):
    """C(0, v) = 0 for any copula."""
    c = GaussianCopula()
    c.set_parameters([rho])
    assert math.isclose(c.get_cdf(1e-12, v), 0.0, abs_tol=1e-6)


@given(rho=valid_rho(), u=unit_interval())
def test_cdf_boundary_v_one(rho, u):
    """C(u, 1) = u for any copula."""
    c = GaussianCopula()
    c.set_parameters([rho])
    assert math.isclose(c.get_cdf(u, 1 - 1e-12), u, rel_tol=1e-4, abs_tol=1e-4)


@given(rho=valid_rho(), v=unit_interval())
def test_cdf_boundary_u_one(rho, v):
    """C(1, v) = v for any copula."""
    c = GaussianCopula()
    c.set_parameters([rho])
    assert math.isclose(c.get_cdf(1 - 1e-12, v), v, rel_tol=1e-4, abs_tol=1e-4)


# ---------------------------------------------------------------------------
# PDF invariants
# ---------------------------------------------------------------------------

@given(rho=valid_rho(), u=unit_interval(), v=unit_interval())
def test_pdf_nonnegative(rho, u, v):
    """PDF must be non-negative."""
    c = GaussianCopula()
    c.set_parameters([rho])
    pdf = c.get_pdf(u, v)
    assert pdf >= 0.0


@given(rho=valid_rho(), u=unit_interval(), v=unit_interval())
@settings(max_examples=100)
def test_pdf_matches_mixed_derivative(rho, u, v):
    """c(u,v) ≈ ∂²C/∂u∂v via 2D central finite difference (adaptive tolerances)."""
    c = GaussianCopula()
    c.set_parameters([rho])

    ar = abs(float(rho))
    u = float(u)
    v = float(v)

    # --- adaptive finite-diff step (bigger step when problem is ill-conditioned) ---
    # CDF is numerically integrated, and 2D finite differences amplify noise.
    # Larger h reduces numerical cancellation.
    if ar <= 0.95:
        h = 1e-5
    elif ar <= 0.98:
        h = 2e-5
    else:
        h = 5e-5

    pdf_num = _mixed_finite_diff(c.get_cdf, u, v, h=h)
    pdf_ana = float(c.get_pdf(u, v))

    # --- adaptive tolerances ---
    # Tail factor: relax tolerance when u or v close to 0/1 (ppf blows up in tails)
    tail = min(u, 1.0 - u, v, 1.0 - v)
    if tail < 1e-3:
        tail_factor = 50.0
    elif tail < 1e-2:
        tail_factor = 10.0
    elif tail < 5e-2:
        tail_factor = 3.0
    else:
        tail_factor = 1.0

    # Correlation factor: relax as |rho| approaches 1
    if ar <= 0.90:
        corr_factor = 1.0
    elif ar <= 0.95:
        corr_factor = 2.0
    elif ar <= 0.98:
        corr_factor = 5.0
    else:
        corr_factor = 15.0

    abs_tol = 1e-6 * corr_factor * tail_factor
    rel_tol = 3e-4 * corr_factor * tail_factor

    # Robust isclose (handles small/large magnitudes sanely)
    diff = abs(pdf_ana - pdf_num)
    scale = max(1.0, abs(pdf_ana), abs(pdf_num))
    assert diff <= max(abs_tol, rel_tol * scale), (
        f"rho={rho:.6f}, u={u:.6f}, v={v:.6f}, h={h:g}\n"
        f"ana={pdf_ana:.12g}, num={pdf_num:.12g}\n"
        f"diff={diff:.3g}, abs_tol={abs_tol:.3g}, rel_tol={rel_tol:.3g}, scale={scale:.3g}"
    )


@pytest.mark.parametrize("rho", [-0.7, 0.0, 0.3, 0.9])
def test_pdf_integrates_to_one(rho):
    """Monte-Carlo check that ∫₀¹∫₀¹ c(u,v) du dv ≈ 1 for various ρ."""
    c = GaussianCopula()
    c.set_parameters([rho])
    rng = np.random.default_rng(42)
    u, v = rng.random(100_000), rng.random(100_000)
    pdf_vals = c.get_pdf(u, v)
    integral_mc = pdf_vals.mean()
    assert math.isclose(integral_mc, 1.0, rel_tol=1e-2)


# ---------------------------------------------------------------------------
# h-functions (conditional CDFs)
# ---------------------------------------------------------------------------

@given(rho=valid_rho(), u=unit_interval(), v=unit_interval())
def test_h_functions_are_probabilities(rho, u, v):
    """h-functions are conditional CDFs and must lie in [0,1]."""
    c = GaussianCopula()
    c.set_parameters([rho])

    h1 = c.partial_derivative_C_wrt_u(u, v)  # P(V<=v | U=u)
    h2 = c.partial_derivative_C_wrt_v(u, v)  # P(U<=u | V=v)

    assert 0.0 <= h1 <= 1.0
    assert 0.0 <= h2 <= 1.0

@given(rho=valid_rho(), u=unit_interval())
def test_h_u_boundary_in_v(rho, u):
    """
    h_{V|U}(v|u)=∂C/∂u:
      at v≈0 -> 0
      at v≈1 -> 1
    """
    c = GaussianCopula()
    c.set_parameters([rho])

    h_low = c.partial_derivative_C_wrt_u(u, 1e-12)
    h_high = c.partial_derivative_C_wrt_u(u, 1 - 1e-12)

    assert math.isclose(h_low, 0.0, abs_tol=1e-8)
    assert math.isclose(h_high, 1.0, abs_tol=1e-8)


@given(rho=valid_rho(), v=unit_interval())
def test_h_v_boundary_in_u(rho, v):
    """
    h_{U|V}(u|v)=∂C/∂v:
      at u≈0 -> 0
      at u≈1 -> 1
    """
    c = GaussianCopula()
    c.set_parameters([rho])

    h_low = c.partial_derivative_C_wrt_v(1e-12, v)
    h_high = c.partial_derivative_C_wrt_v(1 - 1e-12, v)

    assert math.isclose(h_low, 0.0, abs_tol=1e-8)
    assert math.isclose(h_high, 1.0, abs_tol=1e-8)

@given(rho=valid_rho(), u=unit_interval(), v=unit_interval())
def test_h_functions_cross_symmetry(rho, u, v):
    """For symmetric copulas: ∂C/∂u(u,v) = ∂C/∂v(v,u)."""
    c = GaussianCopula()
    c.set_parameters([rho])

    h_v_given_u = c.partial_derivative_C_wrt_u(u, v)
    h_u_given_v_swapped = c.partial_derivative_C_wrt_v(v, u)

    assert math.isclose(h_v_given_u, h_u_given_v_swapped, rel_tol=1e-12, abs_tol=1e-12)

@given(rho=valid_rho(), u=unit_interval(), v1=unit_interval(), v2=unit_interval())
def test_h_function_monotone_in_v(rho, u, v1, v2):
    """∂C/∂u is monotone increasing in v (it's a CDF in v)."""
    if v1 > v2:
        v1, v2 = v2, v1
    c = GaussianCopula()
    c.set_parameters([rho])
    assert c.partial_derivative_C_wrt_u(u, v1) <= c.partial_derivative_C_wrt_u(u, v2) + 1e-12


# ---------------------------------------------------------------------------
# Derivative cross-check
# ---------------------------------------------------------------------------

@given(rho=valid_rho(), u=unit_interval(), v=unit_interval())
@settings(max_examples=100)
def test_partial_derivative_matches_finite_diff(rho, u, v):
    """Analytical partial derivatives vs numerical finite differences."""
    c = GaussianCopula()
    c.set_parameters([rho])

    def C(x, y):
        return c.get_cdf(x, y)

    num_du = _finite_diff(C, u, v)
    num_dv = _finite_diff(lambda x, y: C(y, x), v, u)

    ana_du = c.partial_derivative_C_wrt_u(u, v)
    ana_dv = c.partial_derivative_C_wrt_v(u, v)

    assert math.isclose(ana_du, num_du, rel_tol=1e-3, abs_tol=1e-4)
    assert math.isclose(ana_dv, num_dv, rel_tol=1e-3, abs_tol=1e-4)


# ---------------------------------------------------------------------------
# Kendall's tau analytical check
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rho, expected_tau", [
    (0.0, 0.0),
    (0.5, (2.0 / math.pi) * math.asin(0.5)),
    (-0.5, (2.0 / math.pi) * math.asin(-0.5)),
    (0.99, (2.0 / math.pi) * math.asin(0.99)),
])
def test_kendall_tau_analytical(rho, expected_tau):
    """Verify kendall_tau matches (2/π)·arcsin(ρ) for known values."""
    c = GaussianCopula()
    c.set_parameters([rho])
    assert math.isclose(c.kendall_tau(), expected_tau, rel_tol=1e-10)


def test_kendall_tau_zero_at_independence():
    """At ρ = 0, Kendall's τ = 0 (independence)."""
    c = GaussianCopula()
    c.set_parameters([0.0])
    assert c.kendall_tau() == 0.0


@given(rho=valid_rho())
def test_kendall_tau_range(rho):
    """Kendall's τ must lie in (-1, 1) for Gaussian copula."""
    c = GaussianCopula()
    c.set_parameters([rho])
    tau = c.kendall_tau()
    assert -1.0 < tau < 1.0


def test_kendall_tau_monotone_in_rho():
    """τ = (2/π)·arcsin(ρ) is strictly increasing in ρ."""
    rhos = [-0.9, -0.5, -0.2, 0.0, 0.2, 0.5, 0.9]
    taus = []
    for rho in rhos:
        c = GaussianCopula()
        c.set_parameters([rho])
        taus.append(c.kendall_tau())
    for i in range(len(taus) - 1):
        assert taus[i] < taus[i + 1]


# ---------------------------------------------------------------------------
# Tail dependence (always zero for Gaussian)
# ---------------------------------------------------------------------------

@given(rho=valid_rho())
def test_tail_dependence_zero(rho):
    """Gaussian copula has no tail dependence for any ρ ∈ (-1, 1)."""
    c = GaussianCopula()
    c.set_parameters([rho])
    assert c.LTDC() == 0.0
    assert c.UTDC() == 0.0


# ---------------------------------------------------------------------------
# Blomqvist beta
# ---------------------------------------------------------------------------

@given(rho=valid_rho())
def test_blomqvist_beta_matches_definition(rho):
    """β = 4·C(½, ½) - 1 for any copula (definition-based check)."""
    c = GaussianCopula()
    c.set_parameters([rho])
    beta_def = 4.0 * float(c.get_cdf(0.5, 0.5)) - 1.0
    beta = float(c.blomqvist_beta())
    assert math.isfinite(beta)
    assert math.isclose(beta, beta_def, rel_tol=1e-12, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# Independence case (ρ = 0)
# ---------------------------------------------------------------------------

@given(u=unit_interval(), v=unit_interval())
def test_independence_cdf_equals_product(u, v):
    """At ρ = 0, Gaussian copula reduces to independence: C(u,v) = u·v."""
    c = GaussianCopula()
    c.set_parameters([0.0])
    assert math.isclose(c.get_cdf(u, v), u * v, rel_tol=1e-6, abs_tol=1e-6)


@given(u=unit_interval(), v=unit_interval())
def test_independence_pdf_equals_one(u, v):
    """At ρ = 0, copula density is 1 everywhere on (0,1)²."""
    c = GaussianCopula()
    c.set_parameters([0.0])
    assert math.isclose(c.get_pdf(u, v), 1.0, rel_tol=1e-10, abs_tol=1e-10)


@given(u=unit_interval(), v=unit_interval())
def test_independence_h_functions_identity(u, v):
    """At ρ = 0: ∂C/∂u = v and ∂C/∂v = u."""
    c = GaussianCopula()
    c.set_parameters([0.0])
    h_v_given_u = c.partial_derivative_C_wrt_u(u, v)
    h_u_given_v = c.partial_derivative_C_wrt_v(u, v)
    assert math.isclose(h_v_given_u, v, rel_tol=1e-8, abs_tol=1e-8)
    assert math.isclose(h_u_given_v, u, rel_tol=1e-8, abs_tol=1e-8)


# ---------------------------------------------------------------------------
# init_from_data round-trip
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("rho_true", [-0.7, -0.3, 0.0, 0.3, 0.7])
def test_init_from_data_roundtrip(rho_true):
    """
    Generate samples with known ρ, then verify init_from_data
    recovers approximately the same ρ.
    """
    c = GaussianCopula()
    c.set_parameters([rho_true])
    data = c.sample(10_000, rng=np.random.default_rng(123))

    rho_recovered = c.init_from_data(data[:, 0], data[:, 1])

    assert math.isclose(rho_recovered[0], rho_true, abs_tol=0.05), \
        f"Expected ρ ≈ {rho_true}, got {rho_recovered[0]}"


# ---------------------------------------------------------------------------
# Sampling sanity check (slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@given(rho=valid_rho())
@settings(max_examples=20)
def test_empirical_kendall_tau_close(rho):
    """Empirical Kendall τ from samples should be close to theoretical."""
    c = GaussianCopula()
    c.set_parameters([rho])

    data = c.sample(5000)
    tau_emp, _ = stx.kendalltau(data[:, 0], data[:, 1])
    tau_theo = c.kendall_tau()

    # Tolerance ≈ 3σ under H0
    sigma = math.sqrt(2 * (2 * len(data) + 5) / (9 * len(data) * (len(data) - 1)))
    assert math.isclose(tau_emp, tau_theo, abs_tol=3 * sigma)


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
    Vectorized get_cdf/get_pdf operate pairwise on (u[i], v[i]) via zip,
    not on the Cartesian product grid.
    """
    u = np.array([0.2, 0.8])
    v = np.array([0.3, 0.7])

    cdf_vec = copula_default.get_cdf(u, v)
    cdf_pair0 = copula_default.get_cdf(u[0], v[0])
    cdf_pair1 = copula_default.get_cdf(u[1], v[1])

    assert cdf_vec.shape == (2,)
    assert np.allclose(cdf_vec, np.array([cdf_pair0, cdf_pair1]))