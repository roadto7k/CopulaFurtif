"""Comprehensive unit‑test suite for the bivariate **Clayton** Archimedean Copula.

Inspired by the AMH test file created earlier; same structure & rigor.

Run with:  pytest -q  (add -m 'not slow' on CI to skip the heavy sampling test).

Dev dependencies (requirements‑dev.txt):
    pytest
    hypothesis
    scipy       # only for the optional empirical Kendall τ check

Key checks implemented:
    • Parameter validation (inside/outside admissible interval).
    • Core invariants: symmetry, monotonicity, CDF/PDF bounds.
    • Tail‑dependence formulas (λ_L > 0, λ_U = 0 for Clayton).
    • Analytical vs. numerical partial derivatives.
    • Sampling sanity: empirical Kendall τ ≈ theoretical (slow).
"""

import math
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from CopulaFurtif.core.copulas.domain.models.archimedean.clayton import ClaytonCopula

# -----------------------------------------------------------------------------
# Fixtures & helpers
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def copula_default():
    """Default Clayton copula with θ = 2.0 (moderate positive dependence)."""
    c = ClaytonCopula()
    c.set_parameters([2])
    return c


# In the library implementation, θ is constrained to (0, 30] (0 not allowed, >30 raises).

@st.composite
def valid_theta(draw):
    return draw(
        st.floats(
            min_value=0.05, max_value=29.99,  # strictly positive
            exclude_min=True,
            exclude_max=False,
            allow_nan=False,
            allow_infinity=False,
        )
    )


@st.composite
def unit_interval(draw):
    return draw(
        st.floats(
            min_value=0.0, max_value=1.0,
            exclude_min=True, exclude_max=True,
        )
    )


# Numerical derivative helper --------------------------------------------------

def _finite_diff(f, x, y, h=1e-6):
    """Central finite difference approximation to ∂f/∂x."""
    return (f(x + h, y) - f(x - h, y)) / (2 * h)


# -----------------------------------------------------------------------------
# Parameter tests
# -----------------------------------------------------------------------------

@given(theta=valid_theta())
def test_parameter_roundtrip(theta):
    c = ClaytonCopula()
    c.set_parameters([theta])
    assert math.isclose(c.get_parameters()[0], theta, rel_tol=1e-12)


@given(theta=st.one_of(
    st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),  # ≤ 0 not allowed
    st.floats(min_value=30.0001, allow_nan=False, allow_infinity=False),  # above max allowed
))
def test_parameter_out_of_bounds(theta):
    c = ClaytonCopula()
    with pytest.raises(ValueError):
        c.set_parameters([theta])


# -----------------------------------------------------------------------------
# CDF invariants
# -----------------------------------------------------------------------------

@given(theta=valid_theta(), u=unit_interval(), v=unit_interval())
def test_cdf_bounds(theta, u, v):
    c = ClaytonCopula()
    c.set_parameters([theta])
    val = c.get_cdf(u, v)
    assert 0.0 <= val <= 1.0


@given(theta=valid_theta(), u1=unit_interval(), u2=unit_interval(), v=unit_interval())
def test_cdf_monotone_in_u(theta, u1, u2, v):
    if u1 > u2:
        u1, u2 = u2, u1
    c = ClaytonCopula()
    c.set_parameters([theta])
    assert c.get_cdf(u1, v) <= c.get_cdf(u2, v)


@given(theta=valid_theta(), u=unit_interval(), v=unit_interval())
def test_cdf_symmetry(theta, u, v):
    c = ClaytonCopula()
    c.set_parameters([theta])
    assert math.isclose(c.get_cdf(u, v), c.get_cdf(v, u), rel_tol=1e-12)


# -----------------------------------------------------------------------------
# PDF invariants
# -----------------------------------------------------------------------------

@given(theta=valid_theta(), u=unit_interval(), v=unit_interval())
def test_pdf_nonnegative(theta, u, v):
    c = ClaytonCopula()
    c.set_parameters([theta])
    assert c.get_pdf(u, v) >= 0.0


# -----------------------------------------------------------------------------
# Derivative cross‑check
# -----------------------------------------------------------------------------

EPS = 1e-4

open_unit = st.floats(min_value=EPS, max_value=1.0-EPS,
                      allow_nan=False, allow_infinity=False)

@given(theta=valid_theta(), u=open_unit, v=open_unit)
@settings(max_examples=100)
def test_partial_derivative_matches_finite_diff(theta, u, v):
    c = ClaytonCopula()
    c.set_parameters([theta])

    ana_du = c.partial_derivative_C_wrt_u(u, v)
    ana_dv = c.partial_derivative_C_wrt_v(u, v)

    # finite-difference helpers
    num_du = _finite_diff(c.get_cdf, u, v)
    num_dv = _finite_diff(lambda x, y: c.get_cdf(y, x), v, u)

    assert math.isfinite(ana_du) and math.isfinite(ana_dv)
    assert math.isclose(ana_du, num_du, rel_tol=1e-3, abs_tol=1e-4)
    assert math.isclose(ana_dv, num_dv, rel_tol=1e-3, abs_tol=1e-4)


# -----------------------------------------------------------------------------
# Tail dependence
# -----------------------------------------------------------------------------

@given(theta=valid_theta())
def test_tail_dependence(theta):
    c = ClaytonCopula()
    c.set_parameters([theta])

    # Clayton lower‑tail dependence: λ_L = 2^(-1/θ)
    expected_lt = 2 ** (-1.0 / theta)

    assert math.isclose(c.LTDC(), expected_lt, rel_tol=1e-12)
    assert c.UTDC() == 0.0


# -----------------------------------------------------------------------------
# Kendall τ formula
# -----------------------------------------------------------------------------

@given(theta=valid_theta())
def test_kendall_tau_formula(theta):
    c = ClaytonCopula()
    c.set_parameters([theta])
    expected = theta / (theta + 2.0)
    assert math.isclose(c.kendall_tau(), expected, rel_tol=1e-12)


# -----------------------------------------------------------------------------
# Sampling sanity check (slow)
# -----------------------------------------------------------------------------

@pytest.mark.slow
@given(theta=valid_theta())
@settings(max_examples=20)
def test_empirical_kendall_tau_close(theta):
    import scipy.stats as stx  # optional dependency
    c = ClaytonCopula()
    c.set_parameters([theta])

    data = c.sample(10000)
    tau_emp, _ = stx.kendalltau(data[:, 0], data[:, 1])
    tau_theo = c.kendall_tau()

    # σ for τ̂ under H₀ (no ties), cf. Kendall 1949
    n = len(data)
    var_tau = (2 * (2 * n + 5)) / (9 * n * (n - 1)) * (1 - tau_theo ** 2) ** 2
    sigma = math.sqrt(var_tau)
    assert abs(tau_emp - tau_theo) <= 4 * sigma


# -----------------------------------------------------------------------------
# IAD / AD disabled behaviour
# -----------------------------------------------------------------------------

def test_iad_ad_disabled(copula_default):
    assert np.isnan(copula_default.IAD(None))
    assert np.isnan(copula_default.AD(None))


# -----------------------------------------------------------------------------
# Vectorised shape checks
# -----------------------------------------------------------------------------

def test_vectorised_shapes(copula_default):
    u = np.linspace(0.05, 0.95, 13)
    v = np.linspace(0.05, 0.95, 13)
    assert copula_default.get_cdf(u, v).shape == (13,)
    assert copula_default.get_pdf(u, v).shape == (13,)

    samples = copula_default.sample(256)
    assert samples.shape == (256, 2)
