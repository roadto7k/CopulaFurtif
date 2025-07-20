"""Unit tests for the two-parameter BB1 copula.

* θ > 0, δ ≥ 1   (per implementation bounds)
* Focus: parameters, invariants, analytic vs numeric partials, τ & tail dependence, sample shapes.
"""

import math
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings
from CopulaFurtif.core.copulas.domain.models.archimedean.BB1 import BB1Copula
from hypothesis import assume
from scipy.stats import kendalltau
from hypothesis import assume

# -----------------------------------------------------------------------------
# Strategies
# -----------------------------------------------------------------------------

theta_valid = st.floats(min_value=0.05, max_value=5.0, allow_nan=False, allow_infinity=False)
delta_valid = st.floats(min_value=1.05, max_value=5.0, allow_nan=False, allow_infinity=False)

theta_invalid = st.floats(max_value=0.0, allow_nan=False)  # θ ≤ 0
delta_invalid = st.floats(max_value=1.0, exclude_max=True, allow_nan=False)  # δ < 1

unit = st.floats(min_value=1e-3, max_value=0.999, allow_nan=False)

# helper finite diff

def _fd(f, x, y, h=1e-5):
    return (f(x + h, y) - f(x - h, y)) / (2 * h)

# -----------------------------------------------------------------------------
# Parameter tests
# -----------------------------------------------------------------------------

@given(theta=theta_valid, delta=delta_valid)
def test_param_roundtrip(theta, delta):
    c = BB1Copula()
    c.set_parameters([theta, delta])
    assert math.isclose(c.get_parameters()[0], theta, rel_tol=1e-12)
    assert math.isclose(c.get_parameters()[1], delta, rel_tol=1e-12)


@given(theta=theta_invalid, delta=delta_valid)
def test_theta_out_of_bounds(theta, delta):
    c = BB1Copula()
    with pytest.raises(ValueError):
        c.set_parameters([theta, delta])


@given(theta=theta_valid, delta=delta_invalid)
def test_delta_out_of_bounds(theta, delta):
    c = BB1Copula()
    with pytest.raises(ValueError):
        c.set_parameters([theta, delta])

# -----------------------------------------------------------------------------
# CDF/PDF invariants
# -----------------------------------------------------------------------------

@given(theta=theta_valid, delta=delta_valid, u=unit, v=unit)
def test_cdf_bounds(theta, delta, u, v):
    cop = BB1Copula(); cop.set_parameters([theta, delta])
    val = cop.get_cdf(u, v)
    assert 0.0 <= val <= 1.0


@given(theta=theta_valid, delta=delta_valid, u=unit, v=unit)
def test_pdf_nonneg(theta, delta, u, v):
    cop = BB1Copula()
    cop.set_parameters([theta, delta])
    assert cop.get_pdf(u, v) >= 0.0


@given(theta=theta_valid, delta=delta_valid, u=unit, v=unit)
def test_cdf_symmetry(theta, delta, u, v):
    cop = BB1Copula()
    cop.set_parameters([theta, delta])
    assert math.isclose(cop.get_cdf(u, v), cop.get_cdf(v, u), rel_tol=1e-12)

@pytest.fixture(scope="module")
def copula_default():
    c = BB1Copula()
    c.set_parameters([2.0, 2.0])
    return c

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

@given(theta=theta_valid, delta=delta_valid, u=unit, v=unit)
@settings(max_examples=100)
def test_pdf_matches_mixed_derivative(theta, delta, u, v):
    """
    For a one‑param copula, check that
      c(u,v) ≈ ∂²C/∂u∂v
    via a 2D central finite difference.
    """
    c = BB1Copula()
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
# Derivative cross-check (restrict θ & δ for stability)
# -----------------------------------------------------------------------------

@given(theta=st.floats(min_value=0.1, max_value=4.0),
       delta=st.floats(min_value=1.1, max_value=4.0),
       u=unit, v=unit)
@settings(max_examples=40)
def test_partial_derivatives(theta, delta, u, v):
    cop = BB1Copula()
    cop.set_parameters([theta, delta])

    def C(x, y):
        return cop.get_cdf(x, y)

    num_du = _fd(C, u, v)
    num_dv = _fd(lambda x, y: C(y, x), v, u)
    ana_du = cop.partial_derivative_C_wrt_u(u, v)
    ana_dv = cop.partial_derivative_C_wrt_v(u, v)

    assert math.isclose(ana_du, num_du, rel_tol=2e-2, abs_tol=2e-3)
    assert math.isclose(ana_dv, num_dv, rel_tol=2e-2, abs_tol=2e-3)

# -----------------------------------------------------------------------------
# Kendall τ & tail dependence
# -----------------------------------------------------------------------------

@given(theta=theta_valid, delta=delta_valid)
def test_tail(theta, delta):

    cop = BB1Copula(); cop.set_parameters([theta, delta])
    # Check LTDC/UTDC formulas
    lamL = cop.LTDC()
    lamU = cop.UTDC()
    assert 0.0 < lamL < 1.0 and 0.0 < lamU < 1.0

@given(
        theta=st.floats(min_value=0.1, max_value=4.0),
        delta=st.floats(min_value=1.1, max_value=4.0)
    )
@settings(max_examples=20)
def test_kendall_tau_montecarlo(theta, delta):
    cop = BB1Copula()
    cop.set_parameters([theta, delta])
    n = 10000
    data = cop.sample(n)
    tau_emp, _ = kendalltau(data[:,0], data[:,1])
    tau_theo   = cop.kendall_tau()

    assume(math.isfinite(tau_theo))

    # borne 4 σ
    var_tau = (2 * (2 * n + 5)) / (9 * n * (n - 1)) * (1 - tau_theo ** 2) ** 2
    sigma = math.sqrt(var_tau)
    assert abs(tau_emp - tau_theo) <= 4*sigma

# -----------------------------------------------------------------------------
# Sample & disabled metrics
# -----------------------------------------------------------------------------

def test_sample_disabled():
    cop = BB1Copula()
    cop.set_parameters([0.8, 1.8])
    samp = cop.sample(500)
    assert samp.shape == (500, 2)
    assert np.isnan(cop.IAD(None)) and np.isnan(cop.AD(None))
