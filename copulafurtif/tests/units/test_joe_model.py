"""Unit‑tests for the Joe copula implementation.

Sampler est placeholder i.i.d. uniform ⇒ pas de test empirique.
"""

import math
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings
from CopulaFurtif.core.copulas.domain.models.archimedean.joe import JoeCopula
import scipy.stats as stx

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
    c = JoeCopula()
    c.set_parameters([theta])
    assert math.isclose(c.get_parameters()[0], theta, rel_tol=1e-12)


@given(theta=theta_invalid)
def test_out_of_bounds(theta):
    c = JoeCopula()
    with pytest.raises(ValueError):
        c.set_parameters([theta])

# ----------------------------------------------------------------------------
# CDF/PDF invariants
# ----------------------------------------------------------------------------

@given(theta=theta_valid, u=unit_interior, v=unit_interior)
def test_cdf_bounds(theta, u, v):
    c = JoeCopula(); c.set_parameters([theta])
    val = c.get_cdf(u, v)
    assert 0.0 <= val <= 1.0


@given(theta=theta_valid, u=unit_interior, v=unit_interior)
def test_pdf_nonneg(theta, u, v):
    c = JoeCopula()
    c.set_parameters([theta])
    assert c.get_pdf(u, v) >= 0.0


@given(theta=theta_valid, u=unit_interior, v=unit_interior)
def test_cdf_symmetry(theta, u, v):
    c = JoeCopula(); c.set_parameters([theta])
    assert math.isclose(c.get_cdf(u, v), c.get_cdf(v, u), rel_tol=1e-12)

# ----------------------------------------------------------------------------
# Derivative cross‑check (θ up to 10 for stability)
# ----------------------------------------------------------------------------

@given(theta=st.floats(min_value=1.1, max_value=10.0), u=unit_interior, v=unit_interior)
@settings(max_examples=40)
def test_partial_derivatives(theta, u, v):
    c = JoeCopula()
    c.set_parameters([theta])
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
def test_tail_dependence(theta):
    c = JoeCopula()
    c.set_parameters([theta])

    # Lower tail (Joe is upper-tail only)
    assert c.LTDC() == 0.0

    # Upper tail λ_U = 2 − 2^{1/θ}
    expected_u = 2.0 - 2.0**(1.0/theta)
    assert math.isclose(c.UTDC(), expected_u, rel_tol=1e-12)


@pytest.mark.slow
@settings(max_examples=20, deadline=None)
@given(theta=theta_valid)
def test_kendall_tau_monte_carlo(theta):
    n = 10_000
    c = JoeCopula()
    c.set_parameters([theta])

    rng  = np.random.default_rng(seed=0)
    data = c.sample(n, rng=rng)
    tau_emp, _ = stx.kendalltau(data[:, 0], data[:, 1])

    tau_theo = c.kendall_tau()

    se = math.sqrt(2 * (2 * n + 5) / (9 * n * (n - 1)))
    sigma_eff = se + 0.5 * (1.0 - tau_theo) ** 2

    assert math.isclose(tau_emp, tau_theo, abs_tol=3 * sigma_eff + 0.005), (
        f"θ={theta:.3f}: emp τ={tau_emp:.4f}, theo τ={tau_theo:.4f}"
    )

# ----------------------------------------------------------------------------
# Sample & disabled metrics
# ----------------------------------------------------------------------------

def test_sample_disabled():
    c = JoeCopula()
    c.set_parameters([2.5])
    samp = c.sample(300)
    assert samp.shape == (300, 2)
    assert np.isnan(c.IAD(None))
    assert np.isnan(c.AD(None))
