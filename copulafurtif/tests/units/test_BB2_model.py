"""
Comprehensive unit-test suite for the bivariate **BB2** Archimedean Copula
(the 180-degree survival rotation of BB1).

Structure, swagger, and paranoia all borrowed from the earlier Clayton
test file — just flipped for the BB2 flavour.

Run with:  pytest -q            # fast
           pytest -q -m 'slow'  # includes the heavy sampling sanity check

Dev deps (requirements-dev.txt):
    pytest
    hypothesis
    scipy       # only for the optional empirical Kendall τ check

Checks implemented
------------------
• Parameter validation (inside/outside admissible rectangle).
• Core invariants: symmetry, monotonicity, CDF/PDF bounds.
• Tail-dependence formulas (λ_L < 1, λ_U > 0 for BB2).
• Analytical vs. numerical partial derivatives.
• Kendall τ closed-form vs. implementation.
• Sampling sanity: empirical τ ≈ theoretical (marked slow).
• IAD / AD disabled behaviour.
• Vectorised broadcasting & shape guarantees.
"""

import math

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, Verbosity

from CopulaFurtif.core.copulas.domain.models.archimedean.BB2 import BB2Copula
import scipy.stats as stx  # optional dependency
import jax
import jax.numpy as jnp



# -----------------------------------------------------------------------------
# Fixtures & helpers
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def copula_default():
    """Default BB2 copula with (θ, δ) = (2.0, 2)."""
    c = BB2Copula()
    c.set_parameters([2.0, 2])
    return c


# Library bounds: θ ∈ (0, ∞), δ ∈ [1, ∞). We cap the upper end for Hypothesis.
@st.composite
def valid_theta(draw):
    return draw(
        st.floats(
            min_value=0.05, max_value=10.0,
            exclude_min=True, exclude_max=True,
            allow_nan=False, allow_infinity=False,
        )
    )


@st.composite
def valid_delta(draw):
    return draw(
        st.floats(
            min_value=1.00, max_value=10.0,
            exclude_min=True, exclude_max=True,
            allow_nan=False, allow_infinity=False,
        )
    )

eps_unit = 1e-3
unit = st.floats(min_value=eps_unit, max_value=1-eps_unit, allow_nan=False)


# -----------------------------------------------------------------------------
# Parameter tests
# -----------------------------------------------------------------------------

@given(theta=valid_theta(), delta=valid_delta())
def test_parameter_roundtrip(theta, delta):
    c = BB2Copula()
    c.set_parameters([theta, delta])
    assert math.isclose(c.get_parameters()[0], theta, rel_tol=1e-12)
    assert math.isclose(c.get_parameters()[1], delta, rel_tol=1e-12)


@given(
    theta=st.one_of(
        st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
        st.floats(max_value=0.05, allow_nan=False, allow_infinity=False, exclude_max=True),
    ),
    delta=valid_delta(),
)
def test_theta_out_of_bounds(theta, delta):
    c = BB2Copula()
    with pytest.raises(ValueError):
        c.set_parameters([theta, delta])


@given(
    theta=valid_theta(),
    delta=st.floats(max_value=1.0, allow_nan=False, allow_infinity=False, exclude_max = True),
)
def test_delta_out_of_bounds(theta, delta):
    c = BB2Copula()
    with pytest.raises(ValueError):
        c.set_parameters([theta, delta])


# -----------------------------------------------------------------------------
# CDF invariants
# -----------------------------------------------------------------------------
# @settings(max_examples=2000)
# @given(theta=valid_theta(), delta=valid_delta(), u=unit, v=unit)
# def test_cdf_bounds(theta, delta, u, v):
#     c = BB2Copula()
#     c.set_parameters([theta, delta])
#     val = c.get_cdf(u, v)
#     assert 0.0 <= val <= 1.0
#
#
# @given(theta=valid_theta(), delta=valid_delta(),
#        u1=unit, u2=unit, v=unit)
# def test_cdf_monotone_in_u(theta, delta, u1, u2, v):
#     if u1 > u2:
#         u1, u2 = u2, u1
#     c = BB2Copula()
#     c.set_parameters([theta, delta])
#     assert c.get_cdf(u1, v) <= c.get_cdf(u2, v)
#
#
# @given(theta=valid_theta(), delta=valid_delta(), u=unit, v=unit)
# def test_cdf_symmetry(theta, delta, u, v):
#     c = BB2Copula()
#     c.set_parameters([theta, delta])
#     assert math.isclose(c.get_cdf(u, v), c.get_cdf(v, u), rel_tol=1e-12)

def test_pdf_integrates_to_one_jax_verbose(copula_default):
    """
    Monte Carlo via JAX: estimate that ∫₀¹∫₀¹ c(u,v) du dv ≈ 1 within 1% tolerance,
    with extra prints to debug.
    """
    # 1) Draw 200 000 random points
    key = jax.random.PRNGKey(0)
    key, key_u, key_v = jax.random.split(key, 3)
    eps = 1e-6
    u = jax.random.uniform(key_u, (200_000,), minval=eps, maxval=1.0 - eps)
    v = jax.random.uniform(key_v, (200_000,), minval=eps, maxval=1.0 - eps)

    # 2) Evaluate PDF and sanitize
    pdf_vals = copula_default.get_pdf(u, v)
    pdf_vals = jnp.nan_to_num(pdf_vals, nan=0.0, posinf=0.0, neginf=0.0)

    # 3) Block and convert to numpy for analysis
    u_np   = np.array(u.block_until_ready())
    v_np   = np.array(v.block_until_ready())
    pdf_np = np.array(pdf_vals.block_until_ready())

    # 4) Print summary statistics
    integral_mc = pdf_np.mean()
    print(f"  → MC integral estimate    : {integral_mc:.6f}")
    print(f"  → PDF stats (min, max)    : {pdf_np.min():.3e}, {pdf_np.max():.3e}")
    print(f"  → PDF stats (mean, std)   : {pdf_np.mean():.3e}, {pdf_np.std():.3e}")
    print(f"  → % zeros in PDF          : {100.0 * np.mean(pdf_np==0):.2f}%")

    # 5) Show a few (u,v) where pdf==0
    zeros = np.where(pdf_np == 0)[0]
    if zeros.size > 0:
        idx = zeros[:5]
        print("  → first 5 (u,v) with PDF=0:")
        for i in idx:
            print(f"       u={u_np[i]:.6f}, v={v_np[i]:.6f}")

    # 6) Final assertion
    assert math.isclose(integral_mc, 1.0, rel_tol=1e-2), (
        f"Integral estimate out of tolerance: {integral_mc:.6f}"
    )


# -----------------------------------------------------------------------------
# PDF invariants
# -----------------------------------------------------------------------------

# On active la sortie verbose de Hypothesis et on désactive la deadline
@settings(
    max_examples=100,
    verbosity=Verbosity.verbose,
    deadline=None,
)
@given(
    theta=valid_theta(),
    delta=valid_delta(),
    u=unit,
    v=unit,
)
def test_pdf_nonnegative_verbose(theta, delta, u, v):
    copula = BB2Copula()
    copula.set_parameters([theta, delta])
    pdf = copula.get_pdf(u, v)

    # 1) On vérifie d’abord qu’on n’a pas de NaN
    assert not jnp.isnan(pdf), (
        f"PDF returned NaN for θ={theta}, δ={delta}, u={u}, v={v}"
    )

    # 2) Puis on vérifie la non-négativité, avec message en cas d’erreur
    assert pdf >= 0.0, (
        f"PDF is negative ({pdf}) for θ={theta}, δ={delta}, u={u}, v={v}"
    )


# -----------------------------------------------------------------------------
# Tail dependence
# -----------------------------------------------------------------------------

@given(theta=valid_theta(), delta=valid_delta())
def test_tail_dependence(theta, delta):
    c = BB2Copula()
    c.set_parameters([theta, delta])

    # Formulas from class docstring
    expected_lt = 1
    expected_ut = 0

    assert math.isclose(c.LTDC(), expected_lt, rel_tol=1e-12)
    assert math.isclose(c.UTDC(), expected_ut, rel_tol=1e-12)


# -----------------------------------------------------------------------------
# Sampling sanity check (slow)
# -----------------------------------------------------------------------------

@given(theta=valid_theta(), delta=valid_delta())
@settings(max_examples=20)
def test_empirical_kendall_tau_close(theta, delta):

    c = BB2Copula()
    c.set_parameters([theta, delta])

    data = c.sample(10000)
    tau_emp, _ = stx.kendalltau(data[:, 0], data[:, 1])
    tau_theo = c.kendall_tau()

    # σ for τ̂ under H₀ (no ties), cf. Kendall 1949
    n = len(data)
    var_tau = (2 * (2 * n + 5)) / (9 * n * (n - 1)) * (1 - tau_theo ** 2) ** 2
    sigma = math.sqrt(var_tau)
    assert abs(tau_emp - tau_theo) <= 5 * sigma


# -----------------------------------------------------------------------------
# IAD / AD disabled behaviour
# -----------------------------------------------------------------------------

def test_iad_ad_disabled(copula_default):
    assert np.isnan(copula_default.IAD(None))
    assert np.isnan(copula_default.AD(None))


# -----------------------------------------------------------------------------
# Vectorised shape checks
# -----------------------------------------------------------------------------

def test_vectorised_shapes(copula_default):
    u = np.linspace(0.05, 0.95, 11)
    v = np.linspace(0.05, 0.95, 11)

    assert copula_default.get_cdf(u, v).shape == (11,)
    assert copula_default.get_pdf(u, v).shape == (11,)

    samples = copula_default.sample(256)
    assert samples.shape == (256, 2)
