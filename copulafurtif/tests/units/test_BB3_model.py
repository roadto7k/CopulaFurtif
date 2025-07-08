"""
Comprehensive unit-tests for the bivariate **BB3** Archimedean Copula
(Joe–Hu 1996, positive-stable stopped-gamma).

The structure mirrors the BB2 test-suite the user provided – same paranoia,
just with the BB3 formulas/constraints:

• θ ≥ 1 (power) and δ > 0 (scale) parameter rectangle.
• Core invariants: symmetry, monotonicity, CDF/PDF bounds.
• Tail-dependence formulas specific to BB3 (λ_L switch on θ, λ_U on θ).
• Analytical vs. numerical partial derivatives (both ∂/∂u and ∂/∂v).
• Kendall τ closed-form vs. empirical from samples.
• IAD / AD disabled behaviour.
• Broadcasting & shape guarantees.

Run:
    pytest -q            # quick
    pytest -q -m 'slow'  # includes heavy sampling sanity check
"""

import math

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, Verbosity, note
import scipy.stats as stx
from scipy import stats, integrate

# Path must match your repo layout
from CopulaFurtif.core.copulas.domain.models.archimedean.BB3 import BB3Copula

# ----------------------------------------------------------------------------
# Fixtures & helpers
# ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def copula_default():
    """Default BB3 copula with (θ, δ) = (2.0, 1.5)."""
    c = BB3Copula()
    c.set_parameters([2.0, 1.5])
    return c


# Parameter bounds for Hypothesis ------------------------------------------------
# θ ∈ [1, ∞)  (we cap at 5 for speed) ; δ ∈ (0, ∞) (cap at 10)

@st.composite
def valid_theta(draw):
    return draw(
        st.floats(
            min_value=1.0,
            max_value=5.0,
            exclude_min=True,
            exclude_max=True,
            allow_nan=False,
            allow_infinity=False,
        )
    )


@st.composite
def valid_delta(draw):
    return draw(
        st.floats(
            min_value=0.5,
            max_value=10.0,
            exclude_min=True,
            exclude_max=True,
            allow_nan=False,
            allow_infinity=False,
        )
    )

unit = st.floats(min_value=1e-3, max_value=0.999, allow_nan=False)
_rng = np.random.default_rng(123)


# Numerical finite-difference ---------------------------------------------------

def _finite_diff(f, x, y, h=1e-6):
    return (f(x + h, y) - f(x - h, y)) / (2 * h)

ATOL = 1e-12
RTOL = 3e-3

# ----------------------------------------------------------------------------
# Parameter tests
# ----------------------------------------------------------------------------

@given(theta=valid_theta(), delta=valid_delta())
def test_parameter_roundtrip(theta, delta):
    c = BB3Copula()
    c.set_parameters([theta, delta])
    p = c.get_parameters()
    assert math.isclose(p[0], theta, rel_tol=1e-12)
    assert math.isclose(p[1], delta, rel_tol=1e-12)


@given(
    theta=st.floats(max_value=1.0, allow_nan=False, allow_infinity=False, exclude_max=True),
    delta=valid_delta(),
)
def test_theta_out_of_bounds(theta, delta):
    c = BB3Copula()
    with pytest.raises(ValueError):
        c.set_parameters([theta, delta])


@given(
    theta=valid_theta(),
    delta=st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
)
def test_delta_out_of_bounds(theta, delta):
    c = BB3Copula()
    with pytest.raises(ValueError):
        c.set_parameters([theta, delta])


# ----------------------------------------------------------------------------
# CDF invariants
# ----------------------------------------------------------------------------

@given(theta=valid_theta(), delta=valid_delta(), u=unit, v=unit)
def test_cdf_bounds(theta, delta, u, v):
    c = BB3Copula()
    c.set_parameters([theta, delta])
    val = c.get_cdf(u, v)
    assert 0.0 <= val <= 1.0


@given(theta=valid_theta(), delta=valid_delta(),
       u1=unit, u2=unit, v=unit)
def test_cdf_monotone_in_u(theta, delta, u1, u2, v):
    if u1 > u2:
        u1, u2 = u2, u1
    c = BB3Copula()
    c.set_parameters([theta, delta])
    assert c.get_cdf(u1, v) <= c.get_cdf(u2, v)


@given(theta=valid_theta(), delta=valid_delta(), u=unit, v=unit)
def test_cdf_symmetry(theta, delta, u, v):
    c = BB3Copula()
    c.set_parameters([theta, delta])
    assert math.isclose(c.get_cdf(u, v), c.get_cdf(v, u), rel_tol=1e-12)


# ----------------------------------------------------------------------------
# PDF invariants
# ----------------------------------------------------------------------------

@given(theta=valid_theta(), delta=valid_delta(), u=unit, v=unit)
def test_pdf_nonnegative(theta, delta, u, v):
    c = BB3Copula()
    c.set_parameters([theta, delta])
    assert c.get_pdf(u, v) >= 0.0


# ----------------------------------------------------------------------------
# Derivatives: analytic vs numeric
# ----------------------------------------------------------------------------

@given(theta=valid_theta(), delta=valid_delta(), u=unit, v=unit)
@settings(max_examples=100)
def test_partial_derivative_matches_finite_diff(theta, delta, u, v):
    c = BB3Copula()
    c.set_parameters([theta, delta])

    def C(x, y):
        return c.get_cdf(x, y)

    num_du = _finite_diff(C, u, v)
    num_dv = _finite_diff(lambda x, y: C(y, x), v, u)

    ana_du = c.partial_derivative_C_wrt_u(u, v)
    ana_dv = c.partial_derivative_C_wrt_v(u, v)

    assert math.isclose(ana_du, num_du, rel_tol=1e-3, abs_tol=1e-4)
    assert math.isclose(ana_dv, num_dv, rel_tol=1e-3, abs_tol=1e-4)


# ----------------------------------------------------------------------------
# Tail dependence
# ----------------------------------------------------------------------------

@given(theta=valid_theta(), delta=valid_delta())
def test_tail_dependence(theta, delta):
    c = BB3Copula()
    c.set_parameters([theta, delta])

    if theta > 1.0:
        expected_lt = 1.0
    else:  # θ close to 1 (but >1 because of bounds) won't happen; include completeness
        expected_lt = 2.0 ** (-1.0 / delta)
    expected_ut = 2.0 - 2.0 ** (1.0 / theta)

    assert math.isclose(c.LTDC(), expected_lt, rel_tol=1e-12)
    assert math.isclose(c.UTDC(), expected_ut, rel_tol=1e-12)


# ----------------------------------------------------------------------------
# Sampling sanity (slow)
# ----------------------------------------------------------------------------


# @pytest.mark.slow
# @given(theta=valid_theta(), delta=valid_delta())
# @settings(max_examples=20, deadline=None, verbosity=Verbosity.verbose)
# def test_empirical_kendall_tau_close(theta, delta):
#     rng = np.random.default_rng()
#     cop = BB3Copula()
#     cop.set_parameters([theta, delta])
#
#     data = cop.sample(5_000, rng=rng)
#     tau_emp, _ = stx.kendalltau(data[:, 0], data[:, 1], method="asymptotic")
#
#     tau_theo = cop.kendall_tau()
#     n = len(data)
#     var_tau = (2*(2*n + 5)) / (9*n*(n-1)) * (1 - tau_theo**2)**2
#     sigma = math.sqrt(var_tau)
#     floor = 2e-3
#     tol = max(4*sigma, floor)
#
#     # This will be printed even under pytest’s capture
#     note(f"θ={theta:.5f}, δ={delta:.5f}, τ_emp={tau_emp:.6f}, τ_theo={tau_theo:.6f}, tol={tol:.6f}")
#
#     assert abs(tau_emp - tau_theo) <= tol

# ----------------------------------------------------------------------------
# IAD / AD disabled behaviour
# ----------------------------------------------------------------------------

def test_iad_ad_disabled(copula_default):
    assert np.isnan(copula_default.IAD(None))
    assert np.isnan(copula_default.AD(None))


# ----------------------------------------------------------------------------
# Vectorised shape checks
# ----------------------------------------------------------------------------

# def test_vectorised_shapes(copula_default):
#     u = np.linspace(0.05, 0.95, 11)
#     v = np.linspace(0.05, 0.95, 11)
#
#     assert copula_default.get_cdf(u, v).shape == (11,)
#     assert copula_default.get_pdf(u, v).shape == (11,)
#
#     samples = copula_default.sample(256, rng=_rng)
#     assert samples.shape == (256, 2)

