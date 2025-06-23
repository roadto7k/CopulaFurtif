"""Unit‑test suite for the Frank Archimedean Copula (CopulaFurtif).

We keep it pragmatic: cover the essentials while avoiding numeric traps near
θ ≈ 0 and |θ|→35 where derivatives blow up.

Run: pytest tests/test_frank_copula.py -q
"""

import math
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings
from CopulaFurtif.core.copulas.domain.models.archimedean.frank import FrankCopula

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

# Safe interval — skip the singular band around 0 and avoid the extreme ±35.
SAFE_THETA_MIN = -30.0
SAFE_THETA_MAX = 30.0
EPS_ZERO = 0.01  # exclude <|0.01|

@st.composite
def safe_theta(draw):
    neg = st.floats(SAFE_THETA_MIN, -EPS_ZERO, allow_nan=False, allow_infinity=False)
    pos = st.floats(EPS_ZERO, SAFE_THETA_MAX, allow_nan=False, allow_infinity=False)
    return draw(st.one_of(neg, pos))

@st.composite
def unit_interval(draw):
    # Avoid 0/1 to keep pdf finite. Being too close makes also non linear error
    return draw(st.floats(1e-2, 1.0 - 1e-2, allow_nan=False, allow_infinity=False))

eps = np.finfo(float).eps

def optimal_eps(u: float, v: float, h0: float = 1e-5) -> float:
    """
    Compute a differentiation step centered proportionally to the local scale.
    Starts on a basis h0 (1e-3 ≈ √[5]{ε_machine}) that we expand
    with max(u, v) to avoid being too small when u or v ≈ 0.
    """
    return eps**(1/2) * max(1.0, u, v)


def _finite_diff(f, x, y, h=1e-3):
    return (f(x + h, y) - f(x - h, y)) / (2 * h)

# -----------------------------------------------------------------------------
# Parameter round‑trip & validation
# -----------------------------------------------------------------------------

@given(theta=safe_theta())
def test_set_get_parameters(theta):
    cop = FrankCopula()
    cop.parameters = [theta]
    assert math.isclose(cop.parameters[0], theta, rel_tol=1e-12)


@given(theta=st.one_of(
    st.floats(max_value=-35.0, exclude_max=True),
    st.floats(min_value=35.0,  exclude_min=True),
))
def test_parameters_out_of_bounds(theta):
    cop = FrankCopula()
    with pytest.raises(ValueError):
        cop.parameters = [theta]

# -----------------------------------------------------------------------------
# Basic CDF/PDF sanity
# -----------------------------------------------------------------------------

@given(theta=safe_theta(), u=unit_interval(), v=unit_interval())
def test_cdf_bounds(theta, u, v):
    cop = FrankCopula(); cop.parameters = [theta]
    val = cop.get_cdf(u, v)
    assert 0.0 <= val <= 1.0


@given(theta=safe_theta(), u=unit_interval(), v=unit_interval())
def test_pdf_nonnegative(theta, u, v):
    cop = FrankCopula(); cop.parameters = [theta]
    assert cop.get_pdf(u, v) >= 0.0

# -----------------------------------------------------------------------------
# Derivative cross‑check (moderate θ only for stability)
# -----------------------------------------------------------------------------

@given(
    theta=safe_theta(),
    u=unit_interval(),
    v=unit_interval(),
)
@settings(max_examples=50)
def test_partial_derivative_matches_finite_diff(theta, u, v):
    cop = FrankCopula()
    cop.parameters = [theta]

    def C(x, y):
        return cop.get_cdf(x, y)

    # adaptative epsilon
    h = optimal_eps(u, v)

    num_du = _finite_diff(C, u, v, h)
    num_dv = _finite_diff(lambda x, y: C(y, x), v, u, h)

    ana_du = cop.partial_derivative_C_wrt_u(u, v)
    ana_dv = cop.partial_derivative_C_wrt_v(u, v)

    assert math.isclose(ana_du, num_du, rel_tol=1e-2, abs_tol=1e-3)
    assert math.isclose(ana_dv, num_dv, rel_tol=1e-2, abs_tol=1e-3)
# -----------------------------------------------------------------------------
# Kendall τ range & formula sanity
# -----------------------------------------------------------------------------

@given(theta=safe_theta())
def test_kendall_tau_range(theta):
    cop = FrankCopula(); cop.parameters = [theta]
    tau = cop.kendall_tau()
    assert math.isfinite(tau)
    # Slight overshoot possible for large |θ| due to spence numerical error
    assert -1.001 <= tau <= 1.001

# -----------------------------------------------------------------------------
# Tail dependence zeros
# -----------------------------------------------------------------------------

def test_tail_dependence_zero():
    cop = FrankCopula(); cop.parameters = [10.0]
    assert cop.LTDC() == 0.0
    assert cop.UTDC() == 0.0

# -----------------------------------------------------------------------------
# Sampling sanity (quick)
# -----------------------------------------------------------------------------

def test_sample_shape():
    cop = FrankCopula(); cop.parameters = [3.0]
    samp = cop.sample(512)
    assert samp.shape == (512, 2)

# -----------------------------------------------------------------------------
# IAD / AD disabled
# -----------------------------------------------------------------------------

def test_iad_ad_disabled():
    cop = FrankCopula()
    assert np.isnan(cop.IAD(None))
    assert np.isnan(cop.AD(None))
