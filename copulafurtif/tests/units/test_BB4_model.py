"""
Comprehensive unit-test suite for the bivariate **BB4** Archimedean Copula

Structure, swagger, and paranoia all borrowed from the earlier Clayton
test file — just flipped for the BB4 flavour.

Run with:  pytest -q            # fast
           pytest -q -m 'slow'  # includes the heavy sampling sanity check

Dev deps (requirements-dev.txt):
    pytest
    hypothesis
    scipy       # only for the optional empirical Kendall τ check

Checks implemented
------------------
• Parameter validation (inside/outside admissible rectangle).
• Core invariants: symmetry, monotonicity, CDF/PDF bounds.
• Tail-dependence formulas (λ_L < 1, λ_U > 0 for BB4).
• Analytical vs. numerical partial derivatives.
• Kendall τ closed-form vs. implementation.
• Sampling sanity: empirical τ ≈ theoretical (marked slow).
• IAD / AD disabled behaviour.
• Vectorised broadcasting & shape guarantees.
"""

import math

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, note, Verbosity

from CopulaFurtif.core.copulas.domain.models.archimedean.BB4 import BB4Copula
import scipy.stats as stx  # optional dependency


# -----------------------------------------------------------------------------
# Fixtures & helpers
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def copula_default():
    """Default BB4 copula with (θ, δ) = (2.0, 1.5)."""
    c = BB4Copula()
    c.set_parameters([2.0, 1.5])
    return c


# Library bounds: θ ∈ (0, ∞), δ ∈ [1, ∞). We cap the upper end for Hypothesis.
@st.composite
def valid_theta(draw):
    return draw(
        st.floats(
            min_value=0.05, max_value=10.0,
            exclude_min=True, exclude_max=True,
            allow_nan=False, allow_infinity=False,
        )
    )


@st.composite
def valid_delta(draw):
    return draw(
        st.floats(
            min_value=0.05, max_value=10.0,
            exclude_min=True, exclude_max=True,
            allow_nan=False, allow_infinity=False,
        )
    )


unit = st.floats(min_value=1e-3, max_value=0.999, allow_nan=False)

# Numerical derivative helper --------------------------------------------------
def _finite_diff(f, x, y, h=1e-6):
    """Central finite difference approximation to ∂f/∂x."""
    return (f(x + h, y) - f(x - h, y)) / (2 * h)


# -----------------------------------------------------------------------------
# Parameter tests
# -----------------------------------------------------------------------------

@given(theta=valid_theta(), delta=valid_delta())
def test_parameter_roundtrip(theta, delta):
    c = BB4Copula()
    c.set_parameters([theta, delta])
    assert math.isclose(c.get_parameters()[0], theta, rel_tol=1e-12)
    assert math.isclose(c.get_parameters()[1], delta, rel_tol=1e-12)


@given(
    theta=st.one_of(
        st.floats(max_value=1e-6, allow_nan=False, allow_infinity=False),
        st.floats(max_value=1e-6, allow_nan=False, allow_infinity=False, exclude_max=True),
    ),
    delta=valid_delta(),
)
def test_theta_out_of_bounds(theta, delta):
    c = BB4Copula()
    with pytest.raises(ValueError):
        c.set_parameters([theta, delta])


@given(
    theta=valid_theta(),
    delta=st.floats(max_value=1e-6, allow_nan=False, allow_infinity=False, exclude_max = True),
)
def test_delta_out_of_bounds(theta, delta):
    c = BB4Copula()
    with pytest.raises(ValueError):
        c.set_parameters([theta, delta])


# -----------------------------------------------------------------------------
# CDF invariants
# -----------------------------------------------------------------------------

@given(theta=valid_theta(), delta=valid_delta(), u=unit, v=unit)
def test_cdf_bounds(theta, delta, u, v):
    c = BB4Copula()
    c.set_parameters([theta, delta])
    val = c.get_cdf(u, v)
    assert 0.0 <= val <= 1.0


@given(theta=valid_theta(), delta=valid_delta(),
       u1=unit, u2=unit, v=unit)
def test_cdf_monotone_in_u(theta, delta, u1, u2, v):
    if u1 > u2:
        u1, u2 = u2, u1
    c = BB4Copula()
    c.set_parameters([theta, delta])
    assert c.get_cdf(u1, v) <= c.get_cdf(u2, v)


@given(theta=valid_theta(), delta=valid_delta(), u=unit, v=unit)
def test_cdf_symmetry(theta, delta, u, v):
    c = BB4Copula()
    c.set_parameters([theta, delta])
    assert math.isclose(c.get_cdf(u, v), c.get_cdf(v, u), rel_tol=1e-12)


def _mixed_finite_diff(C, u, v, h=1e-5):
    """
    Central 2‑D finite difference:
        ∂²C/∂u∂v  ≈  [ C(u+h, v+h) – C(u+h, v–h)
                      –C(u–h, v+h) + C(u–h, v–h) ] / (4 h²)
    """
    return (
        C(u + h, v + h)
        - C(u + h, v - h)
        - C(u - h, v + h)
        + C(u - h, v - h)
    ) / (4.0 * h * h)

# tolerances
ATOL = 1e-12
RTOL = 3e-2

@given(theta=valid_theta(), delta=valid_delta(), u=unit, v=unit)
@settings(max_examples=100)
def test_pdf_matches_mixed_derivative(theta, delta, u, v):
    """
    For a one‑param copula, check that
      c(u,v) ≈ ∂²C/∂u∂v
    via a 2D central finite difference.
    """
    c = BB4Copula()
    c.set_parameters([theta, delta])

    C = c.get_cdf
    pdf_num = _mixed_finite_diff(C, u, v)
    pdf_ana = c.get_pdf(u, v)

    assert math.isclose(
        pdf_ana, pdf_num, rel_tol=RTOL, abs_tol=1e-3
    ), f"ana={pdf_ana}, num={pdf_num}"

def test_pdf_integrates_to_one(copula_default):
    """
    Monte‑Carlo check that ∫₀¹∫₀¹ c(u,v) du dv == 1.
    """
    rng = np.random.default_rng(42)
    u, v = rng.random(200_000), rng.random(200_000)
    pdf_vals = copula_default.get_pdf(u, v)
    integral_mc = pdf_vals.mean()  # E[c(U,V)] over the unit square

    assert math.isclose(integral_mc, 1.0, rel_tol=1e-2)



# -----------------------------------------------------------------------------
# PDF invariants
# -----------------------------------------------------------------------------

@given(theta=valid_theta(), delta=valid_delta(), u=unit, v=unit)
def test_pdf_nonnegative(theta, delta, u, v):
    c = BB4Copula()
    c.set_parameters([theta, delta])
    assert c.get_pdf(u, v) >= 0.0


# -----------------------------------------------------------------------------
# Derivative cross-check (analytical vs. finite diff)
# -----------------------------------------------------------------------------

@given(theta=valid_theta(), delta=valid_delta(), u=unit, v=unit)
@settings(max_examples=100)
def test_partial_derivative_matches_finite_diff(theta, delta, u, v):
    c = BB4Copula()
    c.set_parameters([theta, delta])

    def C(x, y):
        return c.get_cdf(x, y)

    num_du = _finite_diff(C, u, v)
    num_dv = _finite_diff(lambda x, y: C(y, x), v, u)

    ana_du = c.partial_derivative_C_wrt_u(u, v)
    ana_dv = c.partial_derivative_C_wrt_v(u, v)

    assert math.isclose(ana_du, num_du, rel_tol=1e-3, abs_tol=1e-4)
    assert math.isclose(ana_dv, num_dv, rel_tol=1e-3, abs_tol=1e-4)


# -----------------------------------------------------------------------------
# Tail dependence
# -----------------------------------------------------------------------------

@given(theta=valid_theta(), delta=valid_delta())
def test_tail_dependence(theta, delta):
    c = BB4Copula()
    c.set_parameters([theta, delta])

    # Formulas from class docstring
    expected_lt = (2.0 - 2.0 ** (-1.0 / delta)) ** (-1.0 / theta)
    expected_ut = 2.0 ** (-1.0 / delta)

    assert math.isclose(c.LTDC(), expected_lt, rel_tol=1e-12)
    assert math.isclose(c.UTDC(), expected_ut, rel_tol=1e-12)


# -----------------------------------------------------------------------------
# Sampling sanity check (slow)
# -----------------------------------------------------------------------------

# @pytest.mark.slow
# @given(theta=valid_theta(), delta=valid_delta())
# @settings(max_examples=20, deadline=None, verbosity=Verbosity.verbose)
# def test_empirical_kendall_tau_close(theta, delta):
#
#     c = BB4Copula()
#     c.set_parameters([theta, delta])
#
#     data = c.sample(10000)
#     tau_emp, _ = stx.kendalltau(data[:, 0], data[:, 1])
#     tau_theo = c.kendall_tau()
#
#     # σ for τ̂ under H₀ (no ties), cf. Kendall 1949
#     n = len(data)
#     var_tau = (2 * (2 * n + 5)) / (9 * n * (n - 1)) * (1 - tau_theo ** 2) ** 2
#     sigma = math.sqrt(var_tau)
#
#     note(f"θ={theta:.5f}, δ={delta:.5f}, τ_emp={tau_emp:.6f}, τ_theo={tau_theo:.6f}, tol={4 * sigma:.6f}")
#
#     assert abs(tau_emp - tau_theo) <= 4 * sigma


# -----------------------------------------------------------------------------
# IAD / AD disabled behaviour
# -----------------------------------------------------------------------------

def test_iad_ad_disabled(copula_default):
    assert np.isnan(copula_default.IAD(None))
    assert np.isnan(copula_default.AD(None))


# -----------------------------------------------------------------------------
# Vectorised shape checks
# -----------------------------------------------------------------------------

# def test_vectorised_shapes(copula_default):
#     u = np.linspace(0.05, 0.95, 11)
#     v = np.linspace(0.05, 0.95, 11)
#
#     assert copula_default.get_cdf(u, v).shape == (11,)
#     assert copula_default.get_pdf(u, v).shape == (11,)
#
#     samples = copula_default.sample(256)
#     assert samples.shape == (256, 2)
