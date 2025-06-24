"""Unit‑tests for the Plackett copula implementation.

The sampler is placeholder — we only test shape/invariants, not empirical τ.
"""

import math
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings
from CopulaFurtif.core.copulas.domain.models.archimedean.plackett import PlackettCopula

# -----------------------------------------------------------------------------
# Strategies
# -----------------------------------------------------------------------------

# keep tiny margin away from exactly 0.01 / 100 to avoid sqrt cancelation error
θ_valid = st.floats(min_value=0.02, max_value=99.0, allow_nan=False, allow_infinity=False)
θ_invalid = st.one_of(
    st.floats(max_value=0.01, exclude_max=True, allow_nan=False),
    st.floats(min_value=100.0, allow_nan=False)
)
unit_interior = st.floats(min_value=1e-3, max_value=0.999, allow_nan=False)


# Finite difference helper

def _fd(f, x, y, h=1e-5):
    return (f(x + h, y) - f(x - h, y)) / (2 * h)

# -----------------------------------------------------------------------------
# Parameter tests
# -----------------------------------------------------------------------------

@given(theta=θ_valid)
def test_param_roundtrip(theta):
    c = PlackettCopula(); c.parameters = [theta]
    assert math.isclose(c.parameters[0], theta, rel_tol=1e-12)

@given(theta=θ_invalid)
def test_param_out_of_bounds(theta):
    c = PlackettCopula()
    with pytest.raises(ValueError):
        c.parameters = [theta]

# -----------------------------------------------------------------------------
# CDF / PDF invariants
# -----------------------------------------------------------------------------

@given(theta=θ_valid, u=unit_interior, v=unit_interior)
def test_cdf_bounds(theta, u, v):
    c = PlackettCopula(); c.parameters = [theta]
    val = c.get_cdf(u, v)
    assert 0.0 <= val <= 1.0

@given(theta=θ_valid, u=unit_interior, v=unit_interior)
def test_pdf_nonneg(theta, u, v):
    c = PlackettCopula(); c.parameters = [theta]
    assert c.get_pdf(u, v) >= 0.0

@given(theta=θ_valid, u=unit_interior, v=unit_interior)
def test_cdf_symmetry(theta, u, v):
    c = PlackettCopula(); c.parameters = [theta]
    assert math.isclose(c.get_cdf(u, v), c.get_cdf(v, u), rel_tol=1e-12)

# -----------------------------------------------------------------------------
# Derivative cross‑check (θ limited to avoid ill‑conditioning)
# -----------------------------------------------------------------------------

@given(theta=st.floats(min_value=0.05, max_value=20.0, allow_nan=False),
       u=unit_interior, v=unit_interior)
@settings(max_examples=50)
def test_partial_derivatives(theta, u, v):
    c = PlackettCopula(); c.parameters = [theta]
    def C(x, y):
        return c.get_cdf(x, y)

    num_du = _fd(C, u, v)
    num_dv = _fd(lambda x, y: C(y, x), v, u)
    ana_du = c.partial_derivative_C_wrt_u(u, v)
    ana_dv = c.partial_derivative_C_wrt_v(u, v)

    assert math.isclose(ana_du, num_du, rel_tol=1e-2, abs_tol=1e-3)
    assert math.isclose(ana_dv, num_dv, rel_tol=1e-2, abs_tol=1e-3)

# -----------------------------------------------------------------------------
# Kendall τ & tail dependence
# -----------------------------------------------------------------------------

@given(theta=θ_valid)
def test_tau_tail(theta):
    c = PlackettCopula(); c.parameters = [theta]
    tau = c.kendall_tau()
    expected_tau = (theta - 1) / (theta + 1)
    assert math.isclose(tau, expected_tau, rel_tol=1e-12)
    assert c.LTDC() == 0.0 and c.UTDC() == 0.0

# -----------------------------------------------------------------------------
# Sample & disabled metrics
# -----------------------------------------------------------------------------

def test_sample_disabled():
    c = PlackettCopula(); c.parameters = [5.0]
    samp = c.sample(400)
    assert samp.shape == (400, 2)
    assert np.isnan(c.IAD(None))
    assert np.isnan(c.AD(None))
