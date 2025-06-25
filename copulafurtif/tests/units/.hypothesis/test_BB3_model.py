"""
Unit-test suite for the two-parameter **BB3** (Joe–Hu) Archimedean copula.

Run with:
    pytest -q            # all fast checks
    pytest -q -m slow    # + heavy sampling test

Dev deps:
    pytest
    hypothesis
    scipy                # only for empirical Kendall τ in the slow test
"""

import math
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from CopulaFurtif.core.copulas.domain.models.archimedean.BB3 import BB3Copula

# -----------------------------------------------------------------------------
# Fixtures & Hypothesis helpers
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def copula_default():
    """A moderate-dependence BB3 (δ=2, θ=1.2)."""
    c = BB3Copula()
    c.parameters = [2.0, 1.2]
    return c


@st.composite
def valid_delta(draw):
    # δ = d > 0
    return draw(
        st.floats(
            min_value=0.05,
            max_value=30.0,
            exclude_min=True,
            exclude_max=True,
            allow_nan=False,
            allow_infinity=False,
        )
    )


@st.composite
def valid_theta(draw):
    # θ = q > 1
    return draw(
        st.floats(
            min_value=1.0,
            max_value=5.0,
            exclude_min=True,
            exclude_max=True,
            allow_nan=False,
            allow_infinity=False,
        )
    )


@st.composite
def unit_interval(draw):
    return draw(
        st.floats(
            min_value=1e-5,
            max_value=1.0 - 1e-3,
            exclude_min=True,
            exclude_max=True,
        )
    )


def _finite_diff(f, x, y, h=None):
    """
    Fourth-order central difference ∂f/∂x using a 5-point stencil.

    Error  ~ O(h⁴).  We scale h with the magnitude of x to stay away
    from the clip-floor and avoid catastrophic cancellation.
    """
    # adaptive step: 1e-4 of the scale of x, but at least 1e-8
    if h is None:
        h = max(1e-8, 1e-4 * max(x, 1e-2))

    return (
        -f(x + 2*h, y) + 8*f(x + h, y)
        - 8*f(x - h, y) + f(x - 2*h, y)
    ) / (12 * h)

# -----------------------------------------------------------------------------
# Parameter validation
# -----------------------------------------------------------------------------

@given(delta=valid_delta(), theta=valid_theta())
def test_parameter_roundtrip(delta, theta):
    c = BB3Copula()
    c.parameters = [delta, theta]
    assert math.isclose(c.parameters[0], delta, rel_tol=1e-12)
    assert math.isclose(c.parameters[1], theta, rel_tol=1e-12)


@given(
    delta=st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
    theta=valid_theta(),
)
def test_delta_out_of_bounds(delta, theta):
    c = BB3Copula()
    with pytest.raises(ValueError):
        c.parameters = [delta, theta]


@given(
    delta=valid_delta(),
    theta=st.floats(min_value=0.9, max_value=0.999, allow_nan=False, allow_infinity=False),
)
def test_theta_out_of_bounds(delta, theta):
    c = BB3Copula()
    with pytest.raises(ValueError):
        c.parameters = [delta, theta]

# -----------------------------------------------------------------------------
# CDF invariants
# -----------------------------------------------------------------------------

@given(delta=valid_delta(), theta=valid_theta(), u=unit_interval(), v=unit_interval())
def test_cdf_bounds(delta, theta, u, v):
    c = BB3Copula()
    c.parameters = [delta, theta]
    val = c.get_cdf(u, v)
    assert 0.0 <= val <= 1.0


@given(delta=valid_delta(), theta=valid_theta(),
       u1=unit_interval(), u2=unit_interval(), v=unit_interval())
def test_cdf_monotone_in_u(delta, theta, u1, u2, v):
    if u1 > u2:
        u1, u2 = u2, u1
    c = BB3Copula()
    c.parameters = [delta, theta]
    assert c.get_cdf(u1, v) <= c.get_cdf(u2, v)


@given(delta=valid_delta(), theta=valid_theta(), u=unit_interval(), v=unit_interval())
def test_cdf_symmetry(delta, theta, u, v):
    c = BB3Copula()
    c.parameters = [delta, theta]
    assert math.isclose(c.get_cdf(u, v), c.get_cdf(v, u), rel_tol=1e-12)

# -----------------------------------------------------------------------------
# PDF invariants
# -----------------------------------------------------------------------------

@given(delta=valid_delta(), theta=valid_theta(), u=unit_interval(), v=unit_interval())
def test_pdf_nonnegative(delta, theta, u, v):
    c = BB3Copula()
    c.parameters = [delta, theta]
    assert c.get_pdf(u, v) >= 0.0

# -----------------------------------------------------------------------------
# Derivative cross-check
# -----------------------------------------------------------------------------

@given(delta=valid_delta(), theta=valid_theta(), u=unit_interval(), v=unit_interval())
@settings(max_examples=100)
def test_partial_derivative_matches_finite_diff(delta, theta, u, v):
    c = BB3Copula()
    c.parameters = [delta, theta]

    def C(x, y):
        return c.get_cdf(x, y)

    num_du = _finite_diff(C, u, v)
    num_dv = _finite_diff(lambda x, y: C(y, x), v, u)

    ana_du = c.partial_derivative_C_wrt_u(u, v)
    ana_dv = c.partial_derivative_C_wrt_v(u, v)

    # allow up to 5% of the finite difference when u or v is tiny
    dyn_abs_du = max(1e-4, 0.05 * abs(num_du))
    dyn_abs_dv = max(1e-4, 0.05 * abs(num_dv))
    assert math.isclose(ana_du, num_du, rel_tol=1e-2, abs_tol=dyn_abs_du)
    assert math.isclose(ana_dv, num_dv, rel_tol=1e-2, abs_tol=dyn_abs_dv)

# -----------------------------------------------------------------------------
# Tail dependence
# -----------------------------------------------------------------------------

@given(delta=valid_delta(), theta=valid_theta())
def test_tail_dependence(delta, theta):
    c = BB3Copula()
    c.parameters = [delta, theta]

    expected_ut = 2.0 - 2.0 ** (1.0 / theta)
    expected_lt = 1.0 if theta > 1.0 else 2.0 ** (-1.0 / delta)

    assert math.isclose(c.UTDC(), expected_ut, rel_tol=1e-12)
    assert math.isclose(c.LTDC(), expected_lt, rel_tol=1e-12)

# -----------------------------------------------------------------------------
# Kendall tau – numerical double-integral cross-check
# -----------------------------------------------------------------------------

@given(delta=valid_delta(), theta=valid_theta())
def test_kendall_tau_numerical(delta, theta):
    c = BB3Copula()
    c.parameters = [delta, theta]

    # recompute with a *different* grid to avoid perfect agreement by construction
    eps, n = 1e-6, 801
    u = np.linspace(eps, 1 - eps, n)
    U, V = np.meshgrid(u, u)
    Cvals = c.get_cdf(U, V)
    integral = np.trapz(np.trapz(Cvals, u, axis=1), u)
    tau_num = 4 * integral - 1

    assert math.isclose(c.kendall_tau(), tau_num, rel_tol=1e-2)

# -----------------------------------------------------------------------------
# Sampling sanity check (slow)
# -----------------------------------------------------------------------------

# @pytest.mark.slow
# @given(delta=valid_delta(), theta=valid_theta())
# @settings(max_examples=20)
# def test_empirical_kendall_tau_close(delta, theta):
#     import scipy.stats as stx  # optional dependency
#     c = BB3Copula()
#     c.parameters = [delta, theta]
#
#     data = c.sample(4000)
#     tau_emp, _ = stx.kendalltau(data[:, 0], data[:, 1])
#     tau_theo = c.kendall_tau()
#
#     n = len(data)
#     sigma = math.sqrt(2 * (2 * n + 5) / (9 * n * (n - 1)))
#     assert math.isclose(tau_emp, tau_theo, abs_tol=3 * sigma)

# -----------------------------------------------------------------------------
# Disabled IAD / AD
# -----------------------------------------------------------------------------

def test_iad_ad_disabled(copula_default):
    assert np.isnan(copula_default.IAD(None))
    assert np.isnan(copula_default.AD(None))

# -----------------------------------------------------------------------------
# Broadcasting / shape checks
# -----------------------------------------------------------------------------

def test_vectorised_shapes(copula_default):
    u = np.linspace(0.05, 0.95, 11)
    v = np.linspace(0.05, 0.95, 11)
    assert copula_default.get_cdf(u, v).shape == (11,)
    assert copula_default.get_pdf(u, v).shape == (11,)

    samples = copula_default.sample(256)
    assert samples.shape == (256, 2)
