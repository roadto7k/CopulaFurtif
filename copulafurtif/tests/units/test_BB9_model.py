"""
Comprehensive unit-test suite for the bivariate **BB9** (Crowder) copula.

BB9 is an Archimedean copula (Joe 2014 §4.25.1):

    C(u,v; ϑ,δ) = exp{−[(δ⁻¹−log u)^ϑ + (δ⁻¹−log v)^ϑ − δ⁻ϑ]^{1/ϑ} + δ⁻¹}

where x = δ⁻¹−log u, y = δ⁻¹−log v, ϑ ≥ 1, δ > 0.

Properties (Joe 2014, §4.25.1 / p.206):
  • Archimedean generator: φ(t) = (δ⁻¹ − log t)^ϑ − δ⁻ϑ.
  • Gumbel subfamily when δ→∞ or δ⁻¹→0⁺.
  • C⊥ when δ→0⁺ or ϑ=1; C⁺ when ϑ→∞.
  • No tail dependence (λ_L = λ_U = 0) for ϑ ≥ 1.
  • Concordance increases with both ϑ and δ.

Run with:
    pytest -q                  # fast tests only
    pytest -q -m slow          # includes heavy sampling / init checks
"""

import math

import numpy as np
import pytest
from hypothesis import given, settings, note, HealthCheck
from hypothesis import strategies as st
from scipy.stats import kendalltau as sp_kendalltau

from CopulaFurtif.core.copulas.domain.models.archimedean.BB9 import BB9Copula


# ─────────────────────────────────────────────────────────────────────────────
# Strategies & helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.composite
def valid_theta(draw):
    """ϑ ∈ (1, 10) — strictly greater than 1."""
    return draw(st.floats(min_value=1.0, max_value=10.0,
                          exclude_min=True, allow_nan=False, allow_infinity=False))


@st.composite
def valid_delta(draw):
    """δ ∈ (1e-5, 10) — strictly positive."""
    return draw(st.floats(min_value=1e-5, max_value=10.0,
                          exclude_min=True, allow_nan=False, allow_infinity=False))


@st.composite
def valid_theta_stable(draw):
    """ϑ ∈ (1.1, 7) — avoids near-boundary numerics."""
    return draw(st.floats(min_value=1.1, max_value=7.0,
                          allow_nan=False, allow_infinity=False))


@st.composite
def valid_delta_stable(draw):
    """δ ∈ (0.1, 6)."""
    return draw(st.floats(min_value=0.1, max_value=6.0,
                          allow_nan=False, allow_infinity=False))


unit = st.floats(min_value=1e-3, max_value=0.999, allow_nan=False)


def _cdf_fd(c, u, v, hu=1e-5, hv=1e-5):
    return (c.get_cdf(u+hu, v+hv) - c.get_cdf(u+hu, v-hv)
            - c.get_cdf(u-hu, v+hv) + c.get_cdf(u-hu, v-hv)) / (4*hu*hv)


def _partial_u_fd(c, u, v, h=1e-6):
    return (c.get_cdf(u+h, v) - c.get_cdf(u-h, v)) / (2*h)


def _partial_v_fd(c, u, v, h=1e-6):
    return (c.get_cdf(u, v+h) - c.get_cdf(u, v-h)) / (2*h)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Parameter validation
# ─────────────────────────────────────────────────────────────────────────────

@given(theta=valid_theta(), delta=valid_delta())
def test_parameter_roundtrip(theta, delta):
    c = BB9Copula(); c.set_parameters([theta, delta])
    assert math.isclose(c.get_parameters()[0], theta, rel_tol=1e-12)
    assert math.isclose(c.get_parameters()[1], delta, rel_tol=1e-12)


@pytest.mark.parametrize("theta,delta", [
    (1.0, 0.5), (0.5, 0.5), (0.0, 0.5), (-1.0, 0.5),
])
def test_theta_out_of_bounds(theta, delta):
    """ϑ ≤ 1 must raise ValueError."""
    c = BB9Copula()
    with pytest.raises(ValueError):
        c.set_parameters([theta, delta])


@pytest.mark.parametrize("theta,delta", [
    (2.0, 0.0), (2.0, -0.5),
])
def test_delta_out_of_bounds(theta, delta):
    """δ ≤ 0 must raise ValueError."""
    c = BB9Copula()
    with pytest.raises(ValueError):
        c.set_parameters([theta, delta])


# ─────────────────────────────────────────────────────────────────────────────
# 2. CDF invariants
# ─────────────────────────────────────────────────────────────────────────────

@given(theta=valid_theta(), delta=valid_delta(), u=unit, v=unit)
def test_cdf_in_unit_interval(theta, delta, u, v):
    c = BB9Copula(); c.set_parameters([theta, delta])
    assert 0.0 <= float(c.get_cdf(u, v)) <= 1.0


@given(theta=valid_theta(), delta=valid_delta(), u=unit, v=unit)
def test_cdf_symmetry(theta, delta, u, v):
    """C(u,v) = C(v,u)."""
    c = BB9Copula(); c.set_parameters([theta, delta])
    assert math.isclose(float(c.get_cdf(u, v)), float(c.get_cdf(v, u)), rel_tol=1e-10)


@given(theta=valid_theta(), delta=valid_delta(), u1=unit, u2=unit, v=unit)
def test_cdf_monotone_in_u(theta, delta, u1, u2, v):
    if u1 > u2: u1, u2 = u2, u1
    c = BB9Copula(); c.set_parameters([theta, delta])
    assert float(c.get_cdf(u1, v)) <= float(c.get_cdf(u2, v)) + 1e-12


@pytest.mark.parametrize("theta,delta,eps", [(2.0,0.5,1e-3),(3.0,1.0,1e-3)])
def test_cdf_boundary_v_zero(theta, delta, eps):
    """C(u, 0) ≈ 0."""
    c = BB9Copula(); c.set_parameters([theta, delta])
    for u in [0.2, 0.5, 0.9]:
        assert float(c.get_cdf(u, eps)) < 5e-3


@pytest.mark.parametrize("theta,delta,eps", [(2.0,0.5,1e-3),(3.0,1.0,1e-3)])
def test_cdf_boundary_v_one(theta, delta, eps):
    """C(u, 1) ≈ u."""
    c = BB9Copula(); c.set_parameters([theta, delta])
    for u in [0.2, 0.5, 0.9]:
        assert math.isclose(float(c.get_cdf(u, 1-eps)), u, abs_tol=5e-3)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Fréchet bounds
# ─────────────────────────────────────────────────────────────────────────────

@given(theta=valid_theta_stable(), delta=valid_delta_stable(), u=unit, v=unit)
@settings(max_examples=50, deadline=None)
def test_frechet_lower_bound(theta, delta, u, v):
    """C(u,v) ≥ max(u+v−1, 0)."""
    c = BB9Copula(); c.set_parameters([theta, delta])
    assert float(c.get_cdf(u, v)) >= max(u+v-1.0, 0.0) - 1e-10


@given(theta=valid_theta_stable(), delta=valid_delta_stable(), u=unit, v=unit)
@settings(max_examples=50, deadline=None)
def test_frechet_upper_bound(theta, delta, u, v):
    """C(u,v) ≤ min(u,v)."""
    c = BB9Copula(); c.set_parameters([theta, delta])
    assert float(c.get_cdf(u, v)) <= min(u, v) + 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# 4. PDF invariants
# ─────────────────────────────────────────────────────────────────────────────

@given(theta=valid_theta_stable(), delta=valid_delta_stable(), u=unit, v=unit)
def test_pdf_nonnegative(theta, delta, u, v):
    c = BB9Copula(); c.set_parameters([theta, delta])
    assert float(c.get_pdf(u, v)) >= 0.0


@pytest.mark.parametrize("theta,delta,u,v,rtol", [
    (2.0, 0.5, 0.5, 0.5, 1e-3),
    (2.0, 0.5, 0.3, 0.6, 1e-3),
    (3.0, 1.0, 0.4, 0.4, 1e-3),
    (1.5, 2.0, 0.2, 0.8, 1e-3),
])
def test_pdf_matches_finite_diff(theta, delta, u, v, rtol):
    """Analytical PDF matches mixed finite-difference of CDF."""
    c = BB9Copula(); c.set_parameters([theta, delta])
    assert math.isclose(float(c.get_pdf(u, v)), float(_cdf_fd(c, u, v)),
                        rel_tol=rtol, abs_tol=1e-5)


@given(theta=valid_theta_stable(), delta=valid_delta_stable(), u=unit, v=unit)
@settings(max_examples=60, deadline=None)
def test_pdf_symmetry(theta, delta, u, v):
    """c(u,v) = c(v,u)."""
    c = BB9Copula(); c.set_parameters([theta, delta])
    assert math.isclose(float(c.get_pdf(u, v)), float(c.get_pdf(v, u)), rel_tol=1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# 5. H-functions (partial derivatives)
# ─────────────────────────────────────────────────────────────────────────────

@given(theta=valid_theta_stable(), delta=valid_delta_stable(), u=unit, v=unit)
@settings(max_examples=60, deadline=None)
def test_partial_u_matches_fd(theta, delta, u, v):
    c = BB9Copula(); c.set_parameters([theta, delta])
    assert math.isclose(float(c.partial_derivative_C_wrt_u(u, v)),
                        float(_partial_u_fd(c, u, v)), rel_tol=1e-2, abs_tol=1e-3)


@given(theta=valid_theta_stable(), delta=valid_delta_stable(), u=unit, v=unit)
@settings(max_examples=60, deadline=None)
def test_partial_v_matches_fd(theta, delta, u, v):
    c = BB9Copula(); c.set_parameters([theta, delta])
    assert math.isclose(float(c.partial_derivative_C_wrt_v(u, v)),
                        float(_partial_v_fd(c, u, v)), rel_tol=1e-2, abs_tol=1e-3)


@given(theta=valid_theta_stable(), delta=valid_delta_stable(), u=unit, v=unit)
@settings(max_examples=40, deadline=None)
def test_partial_u_in_unit_interval(theta, delta, u, v):
    """∂C/∂u ∈ [0,1] (conditional CDF)."""
    c = BB9Copula(); c.set_parameters([theta, delta])
    val = float(c.partial_derivative_C_wrt_u(u, v))
    assert -1e-6 <= val <= 1.0 + 1e-6


@given(theta=valid_theta_stable(), delta=valid_delta_stable(), u=unit, v=unit)
@settings(max_examples=40, deadline=None)
def test_partial_symmetry(theta, delta, u, v):
    """∂C/∂u(u,v) = ∂C/∂v(v,u)."""
    c = BB9Copula(); c.set_parameters([theta, delta])
    assert math.isclose(float(c.partial_derivative_C_wrt_u(u, v)),
                        float(c.partial_derivative_C_wrt_v(v, u)), rel_tol=1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Kendall's tau
# ─────────────────────────────────────────────────────────────────────────────

@given(theta=valid_theta_stable(), delta=valid_delta_stable())
@settings(max_examples=20, deadline=None)
def test_kendall_tau_in_unit_interval(theta, delta):
    """τ ∈ (0,1) for valid BB9 params."""
    c = BB9Copula(); c.set_parameters([theta, delta])
    tau = float(c.kendall_tau())
    assert 0.0 < tau < 1.0


@pytest.mark.parametrize("thetas,delta", [([1.2, 1.8, 3.0, 5.0, 8.0], 0.5)])
def test_kendall_tau_increasing_in_theta(thetas, delta):
    """τ increases with ϑ (concordance ↑)."""
    c = BB9Copula()
    taus = [c.set_parameters([th, delta]) or c.kendall_tau() for th in thetas]
    print(f"τ(ϑ) = {[f'{t:.4f}' for t in taus]}")
    assert all(taus[i] < taus[i+1] for i in range(len(taus)-1))


@pytest.mark.parametrize("theta,deltas", [(2.0, [0.2, 0.5, 1.0, 2.0, 5.0])])
def test_kendall_tau_increasing_in_delta(theta, deltas):
    """τ increases with δ (concordance ↑)."""
    c = BB9Copula()
    taus = [c.set_parameters([theta, de]) or c.kendall_tau() for de in deltas]
    print(f"τ(δ) = {[f'{t:.4f}' for t in taus]}")
    assert all(taus[i] < taus[i+1] for i in range(len(taus)-1))


# ─────────────────────────────────────────────────────────────────────────────
# 7. Tail dependence
# ─────────────────────────────────────────────────────────────────────────────

@given(theta=valid_theta(), delta=valid_delta())
def test_ltdc_is_zero(theta, delta):
    """λ_L = 0 for ϑ ≥ 1 (lower tail order κ_L = 2^{1/ϑ} > 1)."""
    c = BB9Copula(); c.set_parameters([theta, delta])
    assert c.LTDC() == 0.0


@given(theta=valid_theta(), delta=valid_delta())
def test_utdc_is_zero(theta, delta):
    """λ_U = 0 for ϑ ≥ 1 (upper tail order κ_U = 2 > 1)."""
    c = BB9Copula(); c.set_parameters([theta, delta])
    assert c.UTDC() == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 8. Blomqvist's beta
# ─────────────────────────────────────────────────────────────────────────────

@given(theta=valid_theta_stable(), delta=valid_delta_stable())
@settings(max_examples=30, deadline=None)
def test_blomqvist_beta_formula(theta, delta):
    """β = 4·C(½,½) − 1 matches blomqvist_beta()."""
    c = BB9Copula(); c.set_parameters([theta, delta])
    assert math.isclose(float(c.blomqvist_beta()),
                        float(4.0 * c.get_cdf(0.5, 0.5) - 1.0), rel_tol=1e-10)


@given(theta=valid_theta_stable(), delta=valid_delta_stable())
@settings(max_examples=30, deadline=None)
def test_blomqvist_beta_range(theta, delta):
    """β ∈ (0,1) for BB9."""
    c = BB9Copula(); c.set_parameters([theta, delta])
    beta = float(c.blomqvist_beta())
    assert 0.0 < beta < 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 9. IAD / AD disabled
# ─────────────────────────────────────────────────────────────────────────────

def test_iad_returns_nan():
    c = BB9Copula(); c.set_parameters([2.0, 0.5])
    assert np.isnan(c.IAD(None))


def test_ad_returns_nan():
    c = BB9Copula(); c.set_parameters([2.0, 0.5])
    assert np.isnan(c.AD(None))


# ─────────────────────────────────────────────────────────────────────────────
# 10. init_from_data  (slow)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
@given(theta=valid_theta_stable(), delta=valid_delta_stable())
@settings(max_examples=8, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_init_from_data_reproduces_tau(theta, delta):
    """init_from_data on n=2000 samples reproduces τ within ±0.15."""
    c = BB9Copula(); c.set_parameters([theta, delta])
    data = c.sample(2000, rng=np.random.default_rng(0))

    c2 = BB9Copula()
    p0 = c2.init_from_data(data[:, 0], data[:, 1])
    assert np.all(np.isfinite(p0)) and np.all(p0 > 0)

    c2.set_parameters(p0)
    tau_fit  = c2.kendall_tau()
    tau_true = c.kendall_tau()
    note(f"ϑ={theta:.3f}, δ={delta:.3f}, τ_true={tau_true:.4f}, τ_fit={tau_fit:.4f}")
    assert abs(tau_fit - tau_true) < 0.15


# ─────────────────────────────────────────────────────────────────────────────
# 11. Sampling  (slow)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
@given(theta=valid_theta_stable(), delta=valid_delta_stable())
@settings(max_examples=8, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_sampling_shape(theta, delta):
    """sample(n) returns (n,2) with values in (0,1)."""
    c = BB9Copula(); c.set_parameters([theta, delta])
    data = c.sample(200, rng=np.random.default_rng(42))
    assert data.shape == (200, 2)
    assert np.all(data > 0) and np.all(data < 1)


@pytest.mark.slow
@given(theta=valid_theta_stable(), delta=valid_delta_stable())
@settings(max_examples=6, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_sampling_empirical_tau_close(theta, delta):
    """Empirical τ from n=4000 samples ≈ theoretical τ (tol 0.10)."""
    c = BB9Copula(); c.set_parameters([theta, delta])
    data = c.sample(4000, rng=np.random.default_rng(0))
    tau_emp = float(sp_kendalltau(data[:, 0], data[:, 1]).correlation)
    tau_th  = c.kendall_tau()
    note(f"ϑ={theta:.3f}, δ={delta:.3f}, τ_emp={tau_emp:.4f}, τ_th={tau_th:.4f}")
    assert math.isfinite(tau_emp)
    assert abs(tau_emp - tau_th) < 0.10


# ─────────────────────────────────────────────────────────────────────────────
# 12. Vectorisation
# ─────────────────────────────────────────────────────────────────────────────

def test_vectorised_cdf_shape():
    c = BB9Copula(); c.set_parameters([2.0, 0.5])
    u = np.linspace(0.05, 0.95, 20); v = np.linspace(0.05, 0.95, 20)
    out = c.get_cdf(u, v)
    assert out.shape == (20,) and np.all(np.isfinite(out))


def test_vectorised_pdf_shape():
    c = BB9Copula(); c.set_parameters([2.0, 0.5])
    u = np.linspace(0.05, 0.95, 20); v = np.linspace(0.05, 0.95, 20)
    out = c.get_pdf(u, v)
    assert out.shape == (20,) and np.all(np.isfinite(out)) and np.all(out >= 0.0)


def test_vectorised_partial_shape():
    c = BB9Copula(); c.set_parameters([2.0, 0.5])
    u = np.linspace(0.05, 0.95, 15); v = np.linspace(0.05, 0.95, 15)
    assert c.partial_derivative_C_wrt_u(u, v).shape == (15,)
    assert c.partial_derivative_C_wrt_v(u, v).shape == (15,)


def test_scalar_and_array_agree():
    c = BB9Copula(); c.set_parameters([2.0, 0.5])
    for u0, v0 in [(0.2, 0.7), (0.5, 0.5), (0.8, 0.3)]:
        scalar = float(c.get_cdf(u0, v0))
        array  = float(c.get_cdf(np.array([u0]), np.array([v0]))[0])
        assert math.isclose(scalar, array, rel_tol=1e-12)