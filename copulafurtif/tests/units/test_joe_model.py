"""Unit‑tests for the Joe copula implementation.

Sampler est placeholder i.i.d. uniform ⇒ pas de test empirique.
"""

import math
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings
from CopulaFurtif.core.copulas.domain.models.archimedean.joe import JoeCopula

# ----------------------------------------------------------------------------
# Strategies
# ----------------------------------------------------------------------------

theta_valid = st.floats(min_value=1.05, max_value=29.5, allow_nan=False, allow_infinity=False)

theta_invalid = st.one_of(
    st.floats(max_value=1.01, exclude_max=True, allow_nan=False),
    st.floats(min_value=30.0, allow_nan=False)
)

unit_interior = st.floats(min_value=1e-3, max_value=0.999, allow_nan=False)


# Finite diff helper

def _fd(f, x, y, h=1e-5):
    return (f(x + h, y) - f(x - h, y)) / (2 * h)

# ----------------------------------------------------------------------------
# Parameter validation
# ----------------------------------------------------------------------------

@given(theta=theta_valid)
def test_roundtrip(theta):
    c = JoeCopula(); c.parameters = [theta]
    assert math.isclose(c.parameters[0], theta, rel_tol=1e-12)


@given(theta=theta_invalid)
def test_out_of_bounds(theta):
    c = JoeCopula()
    with pytest.raises(ValueError):
        c.parameters = [theta]

# ----------------------------------------------------------------------------
# CDF/PDF invariants
# ----------------------------------------------------------------------------

@given(theta=theta_valid, u=unit_interior, v=unit_interior)
def test_cdf_bounds(theta, u, v):
    c = JoeCopula(); c.parameters = [theta]
    val = c.get_cdf(u, v)
    assert 0.0 <= val <= 1.0


@given(theta=theta_valid, u=unit_interior, v=unit_interior)
def test_pdf_nonneg(theta, u, v):
    c = JoeCopula(); c.parameters = [theta]
    assert c.get_pdf(u, v) >= 0.0


@given(theta=theta_valid, u=unit_interior, v=unit_interior)
def test_cdf_symmetry(theta, u, v):
    c = JoeCopula(); c.parameters = [theta]
    assert math.isclose(c.get_cdf(u, v), c.get_cdf(v, u), rel_tol=1e-12)

# ----------------------------------------------------------------------------
# Derivative cross‑check (θ up to 10 for stability)
# ----------------------------------------------------------------------------

@given(theta=st.floats(min_value=1.1, max_value=10.0), u=unit_interior, v=unit_interior)
@settings(max_examples=40)
def test_partial_derivatives(theta, u, v):
    c = JoeCopula(); c.parameters = [theta]
    def C(x, y):
        return c.get_cdf(x, y)
    num_du = _fd(C, u, v)
    num_dv = _fd(lambda x, y: C(y, x), v, u)
    ana_du = c.partial_derivative_C_wrt_u(u, v)
    ana_dv = c.partial_derivative_C_wrt_v(u, v)
    assert math.isclose(ana_du, num_du, rel_tol=5e-2, abs_tol=5e-3)
    assert math.isclose(ana_dv, num_dv, rel_tol=5e-2, abs_tol=5e-3)

# ----------------------------------------------------------------------------
# Tau & tail dependence
# ----------------------------------------------------------------------------

@given(theta=theta_valid)
def test_tau_tail(theta):
    c = JoeCopula(); c.parameters = [theta]
    assert math.isclose(c.kendall_tau(), 1 - 1/theta, rel_tol=1e-12)
    assert c.LTDC() == 0.0
    expected = 2 - 2 ** (1/theta)
    assert math.isclose(c.UTDC(), expected, rel_tol=1e-12)

# ----------------------------------------------------------------------------
# Sample & disabled metrics
# ----------------------------------------------------------------------------

def test_sample_disabled():
    c = JoeCopula(); c.parameters = [2.5]
    samp = c.sample(300)
    assert samp.shape == (300, 2)
    assert np.isnan(c.IAD(None))
    assert np.isnan(c.AD(None))
