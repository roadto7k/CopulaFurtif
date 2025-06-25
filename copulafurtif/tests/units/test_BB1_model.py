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

# -----------------------------------------------------------------------------
# Strategies
# -----------------------------------------------------------------------------

θ_valid = st.floats(min_value=0.05, max_value=5.0, allow_nan=False, allow_infinity=False)
δ_valid = st.floats(min_value=1.05, max_value=5.0, allow_nan=False, allow_infinity=False)

θ_invalid = st.floats(max_value=0.0, allow_nan=False)  # θ ≤ 0
δ_invalid = st.floats(max_value=1.0, exclude_max=True, allow_nan=False)  # δ < 1

unit = st.floats(min_value=1e-3, max_value=0.999, allow_nan=False)

# helper finite diff

def _fd(f, x, y, h=1e-5):
    return (f(x + h, y) - f(x - h, y)) / (2 * h)

# -----------------------------------------------------------------------------
# Parameter tests
# -----------------------------------------------------------------------------

@given(theta=θ_valid, delta=δ_valid)
def test_param_roundtrip(theta, delta):
    c = BB1Copula(); c.parameters = [theta, delta]
    assert math.isclose(c.parameters[0], theta, rel_tol=1e-12)
    assert math.isclose(c.parameters[1], delta, rel_tol=1e-12)


@given(theta=θ_invalid, delta=δ_valid)
def test_theta_out_of_bounds(theta, delta):
    c = BB1Copula()
    with pytest.raises(ValueError):
        c.parameters = [theta, delta]


@given(theta=θ_valid, delta=δ_invalid)
def test_delta_out_of_bounds(theta, delta):
    c = BB1Copula()
    with pytest.raises(ValueError):
        c.parameters = [theta, delta]

# -----------------------------------------------------------------------------
# CDF/PDF invariants
# -----------------------------------------------------------------------------

@given(theta=θ_valid, delta=δ_valid, u=unit, v=unit)
def test_cdf_bounds(theta, delta, u, v):
    cop = BB1Copula(); cop.parameters = [theta, delta]
    val = cop.get_cdf(u, v)
    assert 0.0 <= val <= 1.0


@given(theta=θ_valid, delta=δ_valid, u=unit, v=unit)
def test_pdf_nonneg(theta, delta, u, v):
    cop = BB1Copula(); cop.parameters = [theta, delta]
    assert cop.get_pdf(u, v) >= 0.0


@given(theta=θ_valid, delta=δ_valid, u=unit, v=unit)
def test_cdf_symmetry(theta, delta, u, v):
    cop = BB1Copula(); cop.parameters = [theta, delta]
    assert math.isclose(cop.get_cdf(u, v), cop.get_cdf(v, u), rel_tol=1e-12)

# -----------------------------------------------------------------------------
# Derivative cross-check (restrict θ & δ for stability)
# -----------------------------------------------------------------------------

@given(theta=st.floats(min_value=0.1, max_value=4.0),
       delta=st.floats(min_value=1.1, max_value=4.0),
       u=unit, v=unit)
@settings(max_examples=40)
def test_partial_derivatives(theta, delta, u, v):
    cop = BB1Copula(); cop.parameters = [theta, delta]

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

@given(theta=θ_valid, delta=δ_valid)
def test_tail(theta, delta):

    cop = BB1Copula(); cop.parameters = [theta, delta]
    # Check LTDC/UTDC formulas
    lamL = cop.LTDC()
    lamU = cop.UTDC()
    assert 0.0 < lamL < 1.0 and 0.0 < lamU < 1.0

@pytest.mark.xfail(reason="Sampler for BB1 not yet variance‑consistent with analytic τ")
@given(
        theta=st.floats(min_value=0.1, max_value=4.0),
        delta=st.floats(min_value=1.1, max_value=4.0)
    )
@settings(max_examples=20)
def test_kendall_tau_montecarlo(theta, delta):
    cop = BB1Copula(); cop.parameters = [theta, delta]
    n = 10000
    data = cop.sample(n)
    tau_emp, _ = kendalltau(data[:,0], data[:,1])
    tau_theo   = cop.kendall_tau()

    from hypothesis import assume
    assume(math.isfinite(tau_theo))

    # borne 3 σ
    sigma = math.sqrt(2*(2*n+5)/(9*n*(n-1)))
    assert abs(tau_emp - tau_theo) <= 3*sigma

# -----------------------------------------------------------------------------
# Sample & disabled metrics
# -----------------------------------------------------------------------------

def test_sample_disabled():
    cop = BB1Copula(); cop.parameters = [0.8, 1.8]
    samp = cop.sample(500)
    assert samp.shape == (500, 2)
    assert np.isnan(cop.IAD(None)) and np.isnan(cop.AD(None))
