"""Unit‑tests for the Plackett copula implementation.

The sampler is placeholder — we only test shape/invariants, not empirical τ.
"""

import math
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings
from hypothesis import assume
from CopulaFurtif.core.copulas.domain.models.archimedean.plackett import PlackettCopula
import scipy.stats as stx

# -----------------------------------------------------------------------------
# Strategies
# -----------------------------------------------------------------------------

# keep tiny margin away from exactly 0.01 / 100 to avoid sqrt cancelation error
theta_valid = st.floats(min_value=0.02, max_value=80.0,
                    allow_nan=False, allow_infinity=False) \
          .filter(lambda x: abs(x - 1.0) > 1e-4)

theta_invalid = st.one_of(
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

@given(theta=theta_valid)
def test_param_roundtrip(theta):
    c = PlackettCopula()
    c.set_parameters([theta])
    assert math.isclose(c.get_parameters()[0], theta, rel_tol=1e-12)

@given(theta=theta_invalid)
def test_param_out_of_bounds(theta):
    c = PlackettCopula()
    with pytest.raises(ValueError):
        c.set_parameters([theta]) 

# -----------------------------------------------------------------------------
# CDF / PDF invariants
# -----------------------------------------------------------------------------

@given(theta=theta_valid, u=unit_interior, v=unit_interior)
def test_cdf_bounds(theta, u, v):
    c = PlackettCopula()
    c.set_parameters([theta])
    val = c.get_cdf(u, v)
    assert 0.0 <= val <= 1.0

@given(theta=theta_valid, u=unit_interior, v=unit_interior)
def test_pdf_nonneg(theta, u, v):
    c = PlackettCopula()
    c.set_parameters([theta])
    assert c.get_pdf(u, v) >= 0.0

@given(theta=theta_valid, u=unit_interior, v=unit_interior)
def test_cdf_symmetry(theta, u, v):
    c = PlackettCopula()
    c.set_parameters([theta])
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

@given(theta=theta_valid, u=unit_interior, v=unit_interior)
@settings(max_examples=100)
def test_pdf_matches_mixed_derivative(theta, u, v):
    """
    For a one‑param copula, check that
      c(u,v) ≈ ∂²C/∂u∂v
    via a 2D central finite difference.
    """
    c = PlackettCopula()
    c.set_parameters([theta])

    C = c.get_cdf
    pdf_num = _mixed_finite_diff(C, u, v)
    pdf_ana = c.get_pdf(u, v)

    assert math.isclose(
        pdf_ana, pdf_num, rel_tol=RTOL, abs_tol=1e-3
    ), f"ana={pdf_ana}, num={pdf_num}"

@pytest.fixture(scope="module")
def copula_default():
    c = PlackettCopula()
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
# Derivative cross‑check (θ limited to avoid ill‑conditioning)
# -----------------------------------------------------------------------------

@given(theta=st.floats(min_value=0.05, max_value=20.0, allow_nan=False),
       u=unit_interior, v=unit_interior)
@settings(max_examples=50)
def test_partial_derivatives(theta, u, v):
    c = PlackettCopula()
    c.set_parameters([theta])
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

@given(theta=theta_valid)
def test_tail_dependence_zero(theta):
    c = PlackettCopula()
    c.set_parameters([theta])
    assert c.LTDC() == 0.0
    assert c.UTDC() == 0.0


@pytest.mark.slow
@settings(max_examples=30, deadline=None)
@given(theta=theta_valid)
def test_kendall_tau_monte_carlo(theta):
    n   = 50_000
    rng = np.random.default_rng(123)

    cop = PlackettCopula(); cop.set_parameters([theta])
    uv  = cop.sample(n, rng=rng)

    tau_emp, _ = stx.kendalltau(uv[:, 0], uv[:, 1])
    tau_th     = cop.kendall_tau()

    se = math.sqrt(2 * (2 * n + 5) / (9 * n * (n - 1)))
    assert math.isclose(tau_emp, tau_th, abs_tol=3 * se + 0.01), (
        f"θ={theta:.3f}: emp τ={tau_emp:.4f}, theo τ={tau_th:.4f}"
    )
# -----------------------------------------------------------------------------
# Sample & disabled metrics
# -----------------------------------------------------------------------------

def test_sample_disabled():
    c = PlackettCopula(); c.set_parameters([5.0]) 
    samp = c.sample(400)
    assert samp.shape == (400, 2)
    assert np.isnan(c.IAD(None))
    assert np.isnan(c.AD(None))
