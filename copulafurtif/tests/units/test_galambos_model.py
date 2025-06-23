"""Unit‑test suite for the Galambos extreme‑value copula implementation.

The sampler shipped in CopulaFurtif is i.i.d. uniform (placeholder) —
therefore we test only shape and invariants, pas de validation empirique.

Run: pytest tests/test_galambos_copula.py -q
"""

import math
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings
from CopulaFurtif.core.copulas.domain.models.archimedean.galambos import GalambosCopula

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

@st.composite
def theta_valid(draw):
    return draw(st.floats(min_value=0.05, max_value=9.5, allow_nan=False, allow_infinity=False))

@st.composite
def theta_invalid(draw):
    return draw(st.one_of(
        st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),    # ≤0
        st.floats(min_value=10.0, allow_nan=False, allow_infinity=False),   # ≥10
    ))

@st.composite
def unit_interval(draw):
    return draw(st.floats(min_value=1e-3, max_value=0.999, allow_nan=False, allow_infinity=False))


def _finite_diff(f, x, y, h=1e-5):
    return (f(x + h, y) - f(x - h, y)) / (2 * h)

# -----------------------------------------------------------------------------
# Parameter tests
# -----------------------------------------------------------------------------

@given(theta=theta_valid())
def test_set_get_parameters(theta):
    cop = GalambosCopula(); cop.parameters = [theta]
    assert math.isclose(cop.parameters[0], theta, rel_tol=1e-12)


@given(theta=theta_invalid())
def test_parameter_out_of_bounds(theta):
    cop = GalambosCopula()
    with pytest.raises(ValueError):
        cop.parameters = [theta]

# -----------------------------------------------------------------------------
# CDF & PDF invariants
# -----------------------------------------------------------------------------

@given(theta=theta_valid(), u=unit_interval(), v=unit_interval())
def test_cdf_bounds(theta, u, v):
    cop = GalambosCopula(); cop.parameters = [theta]
    val = cop.get_cdf(u, v)
    assert 0.0 <= val <= 1.0


@given(theta=theta_valid(), u=unit_interval(), v=unit_interval())
def test_pdf_nonnegative(theta, u, v):
    cop = GalambosCopula(); cop.parameters = [theta]
    assert cop.get_pdf(u, v) >= 0.0


@given(theta=theta_valid(), u=unit_interval(), v=unit_interval())
def test_cdf_symmetry(theta, u, v):
    cop = GalambosCopula(); cop.parameters = [theta]
    assert math.isclose(cop.get_cdf(u, v), cop.get_cdf(v, u), rel_tol=1e-12)

# -----------------------------------------------------------------------------
# Derivative cross‑check (moderate θ)
# -----------------------------------------------------------------------------

@given(theta=st.floats(min_value=0.1, max_value=5.0, allow_nan=False),
       u=unit_interval(), v=unit_interval())
@settings(max_examples=40)
def test_partial_derivative_matches_finite_diff(theta, u, v):
    cop = GalambosCopula(); cop.parameters = [theta]

    def C(x, y):
        return cop.get_cdf(x, y)

    num_du = _finite_diff(C, u, v)
    num_dv = _finite_diff(lambda x, y: C(y, x), v, u)
    ana_du = cop.partial_derivative_C_wrt_u(u, v)
    ana_dv = cop.partial_derivative_C_wrt_v(u, v)

    assert math.isclose(ana_du, num_du, rel_tol=1e-2, abs_tol=1e-3)
    assert math.isclose(ana_dv, num_dv, rel_tol=1e-2, abs_tol=1e-3)

# -----------------------------------------------------------------------------
# Kendall τ & tail dependence
# -----------------------------------------------------------------------------

@given(theta=theta_valid())
def test_tau_and_tail(theta):
    cop = GalambosCopula(); cop.parameters = [theta]
    tau = cop.kendall_tau()
    expected_tau = theta / (theta + 2)
    assert math.isclose(tau, expected_tau, rel_tol=1e-12)

    ltdc = cop.LTDC()
    expected_lambda = 2 - 2 ** (1 / theta)
    assert math.isclose(ltdc, expected_lambda, rel_tol=1e-12)
    assert cop.UTDC() == ltdc

# -----------------------------------------------------------------------------
# Sample shape & disabled metrics
# -----------------------------------------------------------------------------

def test_sample_and_disabled_metrics():
    cop = GalambosCopula(); cop.parameters = [2.0]
    samp = cop.sample(256)
    assert samp.shape == (256, 2)
    assert np.isnan(cop.IAD(None))
    assert np.isnan(cop.AD(None))
