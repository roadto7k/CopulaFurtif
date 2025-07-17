"""Unit‑test suite for the Gumbel (Archimedean) copula implementation.

* θ ∈ (1.01, 30).  We validate inside/outside bounds.
* The provided sampler is exact (Hougaard construction with exponentials),
  so we can trust shapes but we skip an empirical Kendall‑τ test for speed.

Run:  pytest tests/test_gumbel_copula.py -q
"""

import math
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings
from CopulaFurtif.core.copulas.domain.models.archimedean.gumbel import GumbelCopula

# -----------------------------------------------------------------------------
# Strategies helpers
# -----------------------------------------------------------------------------

@st.composite
def theta_valid(draw):
    """θ strictly between bounds to avoid numeric blow‑ups at 1.01 & 30."""
    return draw(st.floats(min_value=1, max_value=29.5, exclude_min=True, allow_nan=False, allow_infinity=False))

@st.composite
def theta_invalid(draw):
    return draw(st.one_of(
        st.floats(max_value=1.0, allow_nan=False, allow_infinity=False, exclude_max=False),  # θ ≤ 1.01
        st.floats(min_value=30.0, allow_nan=False, allow_infinity=False, exclude_min=False), # θ ≥ 30
    ))

@st.composite
def unit_interval(draw):
    # Keep away from 0/1 to stabilise logs
    return draw(st.floats(min_value=1e-3, max_value=0.999, allow_nan=False, allow_infinity=False))


def _finite_diff(f, x, y, h=1e-5):
    return (f(x + h, y) - f(x - h, y)) / (2 * h)

# -----------------------------------------------------------------------------
# Parameter tests
# -----------------------------------------------------------------------------

@given(theta=theta_valid())
def test_parameter_roundtrip(theta):
    cop = GumbelCopula()
    cop.set_parameters([theta])
    assert math.isclose(cop.get_parameters()[0], theta, rel_tol=1e-12)


@given(theta=theta_invalid())
def test_parameter_out_of_bounds(theta):
    cop = GumbelCopula()
    with pytest.raises(ValueError):
        cop.set_parameters([theta])

# -----------------------------------------------------------------------------
# CDF / PDF invariants
# -----------------------------------------------------------------------------

@given(theta=theta_valid(), u=unit_interval(), v=unit_interval())
def test_cdf_bounds(theta, u, v):
    cop = GumbelCopula()
    cop.set_parameters([theta])
    val = cop.get_cdf(u, v)
    assert 0.0 <= val <= 1.0


@given(theta=theta_valid(), u=unit_interval(), v=unit_interval())
def test_pdf_nonnegative(theta, u, v):
    cop = GumbelCopula()
    cop.set_parameters([theta])
    assert cop.get_pdf(u, v) >= 0.0


@given(theta=theta_valid(), u=unit_interval(), v=unit_interval())
def test_cdf_symmetry(theta, u, v):
    cop = GumbelCopula()
    cop.set_parameters([theta])
    assert math.isclose(cop.get_cdf(u, v), cop.get_cdf(v, u), rel_tol=1e-12)

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

@given(theta=theta_valid(), u=unit_interval(), v=unit_interval())
@settings(max_examples=100)
def test_pdf_matches_mixed_derivative(theta, u, v):
    """
    For a one‑param copula, check that
      c(u,v) ≈ ∂²C/∂u∂v
    via a 2D central finite difference.
    """
    c = GumbelCopula()
    c.set_parameters([theta])

    C = c.get_cdf
    pdf_num = _mixed_finite_diff(C, u, v)
    pdf_ana = c.get_pdf(u, v)

    assert math.isclose(
        pdf_ana, pdf_num, rel_tol=RTOL, abs_tol=1e-3
    ), f"ana={pdf_ana}, num={pdf_num}"

@pytest.fixture(scope="module")
def copula_default():
    c = GumbelCopula()
    c.set_parameters([2.0])   # pick a valid default θ
    return c

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
# Derivative cross‑check (moderate θ only)
# -----------------------------------------------------------------------------

@given(theta=st.floats(min_value=1.1, max_value=10.0, allow_nan=False),
       u=unit_interval(), v=unit_interval())
@settings(max_examples=40)
def test_partial_derivative_matches_finite_diff(theta, u, v):
    cop = GumbelCopula()
    cop.set_parameters([theta])

    def C(x, y):
        return cop.get_cdf(x, y)

    num_du = _finite_diff(C, u, v)
    num_dv = _finite_diff(lambda x, y: C(y, x), v, u)
    ana_du = cop.partial_derivative_C_wrt_u(u, v)
    ana_dv = cop.partial_derivative_C_wrt_v(u, v)

    assert math.isclose(ana_du, num_du, rel_tol=2e-2, abs_tol=2e-3)
    assert math.isclose(ana_dv, num_dv, rel_tol=2e-2, abs_tol=2e-3)

# -----------------------------------------------------------------------------
# Kendall τ & tail dependence
# -----------------------------------------------------------------------------

@given(theta=theta_valid())
def test_kendall_tau_and_tail(theta):
    cop = GumbelCopula()
    cop.set_parameters([theta])
    tau = cop.kendall_tau()
    assert math.isclose(tau, 1 - 1/theta, rel_tol=1e-12)

    expected_lambda = 2 - 2 ** (1/theta)
    assert math.isclose(cop.LTDC(), 0.0)  # Gumbel has **upper** tail dependence only
    assert math.isclose(cop.UTDC(), expected_lambda, rel_tol=1e-12)

# -----------------------------------------------------------------------------
# Sample shape & disabled metrics
# -----------------------------------------------------------------------------

def test_sample_and_disabled_metrics():
    cop = GumbelCopula()
    cop.parameters = [2.5]
    samp = cop.sample(512)
    assert samp.shape == (512, 2)
    assert np.isnan(cop.IAD(None))
    assert np.isnan(cop.AD(None))
