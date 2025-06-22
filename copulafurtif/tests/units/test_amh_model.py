"""Improved unit‑test suite for the AMH Archimedean Copula.

Run with:  pytest -q  (add -m 'not slow' on CI if you skip the heavy bits)

Dependencies (add to requirements‑dev.txt):
    pytest
    hypothesis
    scipy  # only used for an optional goodness‑of‑fit check

The tests focus on:
    * Parameter validation (inside/outside the admissible interval).
    * Core invariants: symmetry, monotonicity, bounds of CDF/PDF.
    * Tail‑dependence values (analytical zeros for AMH).
    * Analytical vs numerical derivatives (spot‑check).
    * Sampling sanity‑check: empirical Kendall τ vs theoretical.

Slow / stochastic tests are marked with @pytest.mark.slow so they can be
optionally skipped (‑m "not slow").
"""

import math
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from CopulaFurtif.core.copulas.domain.models.archimedean.AMH import AMHCopula

# -----------------------------------------------------------------------------
# Fixtures & helpers
# -----------------------------------------------------------------------------

@pytest.fixture
def copula_default():
    """Default AMH copula with θ = 0.0 (independence)."""
    return AMHCopula()

@st.composite
def valid_theta(draw):
    # AMH admits θ ∈ (‑1, 1)  – exclude the boundaries strictly.
    return draw(st.floats(min_value=-0.999, max_value=0.999, allow_nan=False, allow_infinity=False))

@st.composite
def unit_interval(draw):
    return draw(st.floats(min_value=0.0, max_value=1.0, exclude_min=True, exclude_max=True))

# -----------------------------------------------------------------------------
# Parameter tests
# -----------------------------------------------------------------------------

@given(theta=valid_theta())
def test_parameter_roundtrip(theta):
    c = AMHCopula()
    c.parameters = [theta]
    assert math.isclose(c.parameters[0], theta, rel_tol=1e-12)

@given(theta=st.one_of(
    st.floats(max_value=-1.0, allow_infinity=False, allow_nan=False),
    st.floats(min_value=1.0, allow_infinity=False, allow_nan=False),
))
def test_parameter_out_of_bounds(theta):
    c = AMHCopula()
    with pytest.raises(ValueError):
        c.parameters = [theta]

# Numerical derivative helper --------------------------------------------------

def _finite_diff(f, x, y, h=1e-6):
    """1st‑order central finite difference ∂f/∂x."""
    return (f(x + h, y) - f(x - h, y)) / (2 * h)


# -----------------------------------------------------------------------------
# CDF invariants
# -----------------------------------------------------------------------------

@given(theta=valid_theta(), u=unit_interval(), v=unit_interval())
def test_cdf_bounds(theta, u, v):
    c = AMHCopula()
    c.parameters = [theta]
    val = c.get_cdf(u, v)
    assert 0.0 <= val <= 1.0


@given(theta=valid_theta(), u1=unit_interval(), u2=unit_interval(), v=unit_interval())
def test_cdf_monotone_in_u(theta, u1, u2, v):
    assume = pytest.importorskip("hypothesis")  # ensure Hypothesis is available
    if u1 > u2:
        u1, u2 = u2, u1  # enforce order
    c = AMHCopula()
    c.parameters = [theta]
    assert c.get_cdf(u1, v) <= c.get_cdf(u2, v)


@given(theta=valid_theta(), u=unit_interval(), v=unit_interval())
def test_cdf_symmetry(theta, u, v):
    c = AMHCopula()
    c.parameters = [theta]
    assert math.isclose(c.get_cdf(u, v), c.get_cdf(v, u), rel_tol=1e-12)

# -----------------------------------------------------------------------------
# PDF invariants
# -----------------------------------------------------------------------------

@given(theta=valid_theta(), u=unit_interval(), v=unit_interval())
def test_pdf_nonnegative(theta, u, v):
    c = AMHCopula()
    c.parameters = [theta]
    pdf = c.get_pdf(u, v)
    assert pdf >= 0.0


# -----------------------------------------------------------------------------
# Derivative cross‑check
# -----------------------------------------------------------------------------

@given(theta=valid_theta(), u=unit_interval(), v=unit_interval())
@settings(max_examples=100)
def test_partial_derivative_matches_finite_diff(theta, u, v):
    c = AMHCopula()
    c.parameters = [theta]

    def C(x, y):
        return c.get_cdf(x, y)

    num_du = _finite_diff(C, u, v)
    num_dv = _finite_diff(lambda x, y: C(y, x), v, u)  # exploit symmetry

    ana_du = c.partial_derivative_C_wrt_u(u, v)
    ana_dv = c.partial_derivative_C_wrt_v(u, v)

    assert math.isclose(ana_du, num_du, rel_tol=1e-3, abs_tol=1e-4)
    assert math.isclose(ana_dv, num_dv, rel_tol=1e-3, abs_tol=1e-4)

# -----------------------------------------------------------------------------
# Tail dependence (closed‑form zeros for AMH)
# -----------------------------------------------------------------------------

@given(theta=valid_theta())
def test_tail_dependence_zero(theta):
    c = AMHCopula()
    c.parameters = [theta]
    assert c.LTDC() == 0.0
    assert c.UTDC() == 0.0

# -----------------------------------------------------------------------------
# Sampling sanity check (slow)
# -----------------------------------------------------------------------------

@pytest.mark.slow
@given(theta=valid_theta())
@settings(max_examples=20)
def test_empirical_kendall_tau_close(theta):
    import scipy.stats as stx  # local import for optional dependency

    c = AMHCopula()
    c.parameters = [theta]

    # Draw a moderately‑sized sample and compute empirical τ
    data = c.sample(5000)
    tau_emp, _ = stx.kendalltau(data[:, 0], data[:, 1])
    tau_theo = c.kendall_tau()

    # The variance of τ under H0 ~ 2(2n+5)/(9n(n‑1)); tolerance ≈ 3*σ
    sigma = math.sqrt(2 * (2 * len(data) + 5) / (9 * len(data) * (len(data) - 1)))
    assert math.isclose(tau_emp, tau_theo, abs_tol=3 * sigma)

# -----------------------------------------------------------------------------
# Shape checks (vectorised input)
# -----------------------------------------------------------------------------

def test_vectorised_shapes(copula_default):
    u = np.linspace(0.05, 0.95, 13)
    v = np.linspace(0.05, 0.95, 13)
    assert copula_default.get_cdf(u, v).shape == (13,)
    assert copula_default.get_pdf(u, v).shape == (13,)

    samples = copula_default.sample(256)
    assert samples.shape == (256, 2)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# IAD / AD disabled behavior
# -----------------------------------------------------------------------------

def test_iad_ad_disabled(copula_default):
    """
    When IAD (Integrated Absolute Deviation) and AD (Absolute Deviation) are not configured
    (i.e., parameters not set for these diagnostics), both methods should return NaN,
    indicating the metric is disabled or undefined.
    """
    # Using the default copula (θ=0), IAD and AD methods are not active
    # Expect NaN outputs to signal 'disabled' state rather than numeric results
    assert np.isnan(copula_default.IAD(None))
    assert np.isnan(copula_default.AD(None))
