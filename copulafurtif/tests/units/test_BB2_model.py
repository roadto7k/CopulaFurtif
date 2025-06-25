"""
Comprehensive unit-test suite for the bivariate **BB2** Archimedean Copula
(the 180-degree survival rotation of BB1).

Structure, swagger, and paranoia all borrowed from the earlier Clayton
test file — just flipped for the BB2 flavour.

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
• Tail-dependence formulas (λ_L < 1, λ_U > 0 for BB2).
• Analytical vs. numerical partial derivatives.
• Kendall τ closed-form vs. implementation.
• Sampling sanity: empirical τ ≈ theoretical (marked slow).
• IAD / AD disabled behaviour.
• Vectorised broadcasting & shape guarantees.
"""

import math
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from CopulaFurtif.core.copulas.domain.models.archimedean.BB1 import BB1Copula
from CopulaFurtif.core.copulas.domain.models.archimedean.BB2 import BB2Copula

# -----------------------------------------------------------------------------
# Fixtures & helpers
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def copula_default():
    """Default BB2 copula with (θ, δ) = (2.0, 1.5)."""
    c = BB2Copula()
    c.parameters = [2.0, 1.5]
    return c


# Library bounds: θ ∈ (0, ∞), δ ∈ [1, ∞). We cap the upper end for Hypothesis.
@st.composite
def valid_theta(draw):
    return draw(
        st.floats(
            min_value=0.05, max_value=30.0,
            exclude_min=True, exclude_max=False,
            allow_nan=False, allow_infinity=False,
        )
    )


@st.composite
def valid_delta(draw):
    return draw(
        st.floats(
            min_value=1.05, max_value=10.0,
            exclude_min=True, exclude_max=False,
            allow_nan=False, allow_infinity=False,
        )
    )


@st.composite
def unit_interval(draw):
    return draw(
        st.floats(
            0.0, 1.0,
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

@given(theta=valid_theta(), delta=valid_delta())
def test_parameter_roundtrip(theta, delta):
    c = BB2Copula()
    c.parameters = [theta, delta]
    assert math.isclose(c.parameters[0], theta, rel_tol=1e-12)
    assert math.isclose(c.parameters[1], delta, rel_tol=1e-12)


@given(
    theta=st.one_of(
        st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
        st.floats(max_value=0.99, allow_nan=False, allow_infinity=False),
    ),
    delta=valid_delta(),
)
def test_theta_out_of_bounds(theta, delta):
    c = BB2Copula()
    with pytest.raises(ValueError):
        c.parameters = [theta, delta]


@given(
    theta=valid_theta(),
    delta=st.floats(max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_delta_out_of_bounds(theta, delta):
    c = BB2Copula()
    with pytest.raises(ValueError):
        c.parameters = [theta, delta]


# -----------------------------------------------------------------------------
# CDF invariants
# -----------------------------------------------------------------------------

@given(theta=valid_theta(), delta=valid_delta(), u=unit_interval(), v=unit_interval())
def test_cdf_bounds(theta, delta, u, v):
    c = BB2Copula()
    c.parameters = [theta, delta]
    val = c.get_cdf(u, v)
    assert 0.0 <= val <= 1.0


@given(theta=valid_theta(), delta=valid_delta(),
       u1=unit_interval(), u2=unit_interval(), v=unit_interval())
def test_cdf_monotone_in_u(theta, delta, u1, u2, v):
    if u1 > u2:
        u1, u2 = u2, u1
    c = BB2Copula()
    c.parameters = [theta, delta]
    assert c.get_cdf(u1, v) <= c.get_cdf(u2, v)


@given(theta=valid_theta(), delta=valid_delta(), u=unit_interval(), v=unit_interval())
def test_cdf_symmetry(theta, delta, u, v):
    c = BB2Copula()
    c.parameters = [theta, delta]
    assert math.isclose(c.get_cdf(u, v), c.get_cdf(v, u), rel_tol=1e-12)


# -----------------------------------------------------------------------------
# PDF invariants
# -----------------------------------------------------------------------------

@given(theta=valid_theta(), delta=valid_delta(), u=unit_interval(), v=unit_interval())
def test_pdf_nonnegative(theta, delta, u, v):
    c = BB2Copula()
    c.parameters = [theta, delta]
    assert c.get_pdf(u, v) >= 0.0


# -----------------------------------------------------------------------------
# Derivative cross-check (analytical vs. finite diff)
# -----------------------------------------------------------------------------

@given(theta=valid_theta(), delta=valid_delta(), u=unit_interval(), v=unit_interval())
@settings(max_examples=100)
def test_partial_derivative_matches_finite_diff(theta, delta, u, v):
    c = BB2Copula()
    c.parameters = [theta, delta]

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
    c = BB2Copula()
    c.parameters = [theta, delta]

    # Formulas from class docstring
    expected_lt = 2.0 - 2.0 ** (1.0 / delta)
    expected_ut = 2.0 ** (-1.0 / (delta * theta))

    assert math.isclose(c.LTDC(), expected_lt, rel_tol=1e-12)
    assert math.isclose(c.UTDC(), expected_ut, rel_tol=1e-12)


# -----------------------------------------------------------------------------
# Kendall τ formula
# -----------------------------------------------------------------------------

@given(theta=valid_theta(), delta=valid_delta())
def test_kendall_tau_formula(theta, delta):
    c = BB2Copula()
    c.parameters = [theta, delta]
    # Closed-form from implementation: τ = 1 − (2/δ)(1 − 1/θ) B(1 − 1/θ, 2/δ + 1)
    from scipy.special import beta
    expected = 1.0 - (2.0 / delta) * (1.0 - 1.0 / theta) * beta(1.0 - 1.0 / theta,
                                                                 2.0 / delta + 1.0)
    assert math.isclose(c.kendall_tau(), expected, rel_tol=1e-12)


# -----------------------------------------------------------------------------
# Sampling sanity check (slow)
# -----------------------------------------------------------------------------

@pytest.mark.slow
@given(theta=valid_theta(), delta=valid_delta())
@settings(max_examples=20)
def test_empirical_kendall_tau_close(theta, delta):
    import scipy.stats as stx  # optional dependency
    c = BB2Copula()
    c.parameters = [theta, delta]

    data = c.sample(5000)
    tau_emp, _ = stx.kendalltau(data[:, 0], data[:, 1])
    tau_theo = c.kendall_tau()

    # σ for τ̂ under H₀ (no ties), cf. Kendall 1949
    n = len(data)
    sigma = math.sqrt(2 * (2 * n + 5) / (9 * n * (n - 1)))
    assert math.isclose(tau_emp, tau_theo, abs_tol=3 * sigma)


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
    u = np.linspace(0.05, 0.95, 11)
    v = np.linspace(0.05, 0.95, 11)

    assert copula_default.get_cdf(u, v).shape == (11,)
    assert copula_default.get_pdf(u, v).shape == (11,)

    samples = copula_default.sample(256)
    assert samples.shape == (256, 2)
