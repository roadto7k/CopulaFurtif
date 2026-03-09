"""
Comprehensive unit-test suite for the bivariate **BB10** copula.

BB10 is an Archimedean copula (Joe 2014, §4.26.1, Marshall & Olkin 1988 Example 4.4):

    C(u,v; θ,π) = uv·[1 − π(1−u^θ)(1−v^θ)]^{−1/θ}

where θ > 0, 0 ≤ π ≤ 1.

Properties (Joe 2014, §4.26.1, p.207):
  • Archimedean generator (inverse): φ⁻¹(t;θ,π) = log[(1−π)t^{−θ}+π].
  • C⊥ as θ→0⁺ or π=0; C⁺ as θ→∞ when π=1.
  • No tail dependence (κ_L = κ_U = 2).
  • Concordance always increases with π.
  • For π=1: concordance increases with θ; for 0<π<1: τ is NOT monotone in θ.

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

from CopulaFurtif.core.copulas.domain.models.archimedean.BB10 import BB10Copula


# ─────────────────────────────────────────────────────────────────────────────
# Strategies & helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.composite
def valid_theta(draw):
    """θ ∈ (1e-5, 10) — strictly positive."""
    return draw(st.floats(min_value=1e-5, max_value=10.0,
                          exclude_min=True, allow_nan=False, allow_infinity=False))


@st.composite
def valid_pi(draw):
    """π ∈ (1e-5, 1) — strictly in (0,1)."""
    return draw(st.floats(min_value=1e-5, max_value=1.0,
                          exclude_min=True, exclude_max=True,
                          allow_nan=False, allow_infinity=False))


@st.composite
def valid_theta_stable(draw):
    """θ ∈ (0.1, 8)."""
    return draw(st.floats(min_value=0.1, max_value=8.0,
                          allow_nan=False, allow_infinity=False))


@st.composite
def valid_pi_stable(draw):
    """π ∈ (0.05, 0.95)."""
    return draw(st.floats(min_value=0.05, max_value=0.95,
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

@given(theta=valid_theta(), pi=valid_pi())
def test_parameter_roundtrip(theta, pi):
    c = BB10Copula(); c.set_parameters([theta, pi])
    assert math.isclose(c.get_parameters()[0], theta, rel_tol=1e-12)
    assert math.isclose(c.get_parameters()[1], pi, rel_tol=1e-12)


@pytest.mark.parametrize("theta,pi", [
    (0.0, 0.5), (-1.0, 0.5),
])
def test_theta_out_of_bounds(theta, pi):
    """θ ≤ 0 must raise ValueError."""
    c = BB10Copula()
    with pytest.raises(ValueError):
        c.set_parameters([theta, pi])


@pytest.mark.parametrize("theta,pi", [
    (2.0, 0.0), (2.0, -0.5), (2.0, 1.0),
])
def test_pi_out_of_bounds(theta, pi):
    """π ≤ 0 or π ≥ 1 must raise ValueError."""
    c = BB10Copula()
    with pytest.raises(ValueError):
        c.set_parameters([theta, pi])


# ─────────────────────────────────────────────────────────────────────────────
# 2. CDF invariants
# ─────────────────────────────────────────────────────────────────────────────

@given(theta=valid_theta(), pi=valid_pi(), u=unit, v=unit)
def test_cdf_in_unit_interval(theta, pi, u, v):
    c = BB10Copula(); c.set_parameters([theta, pi])
    assert 0.0 <= float(c.get_cdf(u, v)) <= 1.0


@given(theta=valid_theta(), pi=valid_pi(), u=unit, v=unit)
def test_cdf_symmetry(theta, pi, u, v):
    """C(u,v) = C(v,u)."""
    c = BB10Copula(); c.set_parameters([theta, pi])
    assert math.isclose(float(c.get_cdf(u, v)), float(c.get_cdf(v, u)), rel_tol=1e-10)


@given(theta=valid_theta(), pi=valid_pi(), u1=unit, u2=unit, v=unit)
def test_cdf_monotone_in_u(theta, pi, u1, u2, v):
    if u1 > u2: u1, u2 = u2, u1
    c = BB10Copula(); c.set_parameters([theta, pi])
    assert float(c.get_cdf(u1, v)) <= float(c.get_cdf(u2, v)) + 1e-12


@pytest.mark.parametrize("theta,pi,eps", [(2.0,0.5,1e-3),(3.0,0.7,1e-3)])
def test_cdf_boundary_v_zero(theta, pi, eps):
    """C(u, 0) ≈ 0."""
    c = BB10Copula(); c.set_parameters([theta, pi])
    for u in [0.2, 0.5, 0.9]:
        assert float(c.get_cdf(u, eps)) < 5e-3


@pytest.mark.parametrize("theta,pi,eps", [(2.0,0.5,1e-3),(3.0,0.7,1e-3)])
def test_cdf_boundary_v_one(theta, pi, eps):
    """C(u, 1) ≈ u."""
    c = BB10Copula(); c.set_parameters([theta, pi])
    for u in [0.2, 0.5, 0.9]:
        assert math.isclose(float(c.get_cdf(u, 1-eps)), u, abs_tol=5e-3)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Fréchet bounds
# ─────────────────────────────────────────────────────────────────────────────

@given(theta=valid_theta_stable(), pi=valid_pi_stable(), u=unit, v=unit)
@settings(max_examples=50, deadline=None)
def test_frechet_lower_bound(theta, pi, u, v):
    """C(u,v) ≥ max(u+v−1, 0)."""
    c = BB10Copula(); c.set_parameters([theta, pi])
    assert float(c.get_cdf(u, v)) >= max(u+v-1.0, 0.0) - 1e-10


@given(theta=valid_theta_stable(), pi=valid_pi_stable(), u=unit, v=unit)
@settings(max_examples=50, deadline=None)
def test_frechet_upper_bound(theta, pi, u, v):
    """C(u,v) ≤ min(u,v)."""
    c = BB10Copula(); c.set_parameters([theta, pi])
    assert float(c.get_cdf(u, v)) <= min(u, v) + 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# 4. PDF invariants
# ─────────────────────────────────────────────────────────────────────────────

@given(theta=valid_theta_stable(), pi=valid_pi_stable(), u=unit, v=unit)
def test_pdf_nonnegative(theta, pi, u, v):
    c = BB10Copula(); c.set_parameters([theta, pi])
    assert float(c.get_pdf(u, v)) >= 0.0


@pytest.mark.parametrize("theta,pi,u,v,rtol", [
    (2.0, 0.5, 0.5, 0.5, 1e-3),
    (2.0, 0.5, 0.3, 0.6, 1e-3),
    (2.0, 0.5, 0.2, 0.8, 1e-3),
    (3.0, 0.7, 0.4, 0.4, 1e-3),
    (1.5, 0.3, 0.3, 0.6, 1e-3),
])
def test_pdf_matches_finite_diff(theta, pi, u, v, rtol):
    """Analytical PDF matches mixed finite-difference of CDF."""
    c = BB10Copula(); c.set_parameters([theta, pi])
    assert math.isclose(float(c.get_pdf(u, v)), float(_cdf_fd(c, u, v)),
                        rel_tol=rtol, abs_tol=1e-5)


@given(theta=valid_theta_stable(), pi=valid_pi_stable(), u=unit, v=unit)
@settings(max_examples=60, deadline=None)
def test_pdf_symmetry(theta, pi, u, v):
    """c(u,v) = c(v,u)."""
    c = BB10Copula(); c.set_parameters([theta, pi])
    assert math.isclose(float(c.get_pdf(u, v)), float(c.get_pdf(v, u)), rel_tol=1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# 5. H-functions (partial derivatives)
# ─────────────────────────────────────────────────────────────────────────────

@given(theta=valid_theta_stable(), pi=valid_pi_stable(), u=unit, v=unit)
@settings(max_examples=60, deadline=None)
def test_partial_u_matches_fd(theta, pi, u, v):
    c = BB10Copula(); c.set_parameters([theta, pi])
    assert math.isclose(float(c.partial_derivative_C_wrt_u(u, v)),
                        float(_partial_u_fd(c, u, v)), rel_tol=1e-2, abs_tol=1e-3)


@given(theta=valid_theta_stable(), pi=valid_pi_stable(), u=unit, v=unit)
@settings(max_examples=60, deadline=None)
def test_partial_v_matches_fd(theta, pi, u, v):
    c = BB10Copula(); c.set_parameters([theta, pi])
    assert math.isclose(float(c.partial_derivative_C_wrt_v(u, v)),
                        float(_partial_v_fd(c, u, v)), rel_tol=1e-2, abs_tol=1e-3)


@given(theta=valid_theta_stable(), pi=valid_pi_stable(), u=unit, v=unit)
@settings(max_examples=40, deadline=None)
def test_partial_u_in_unit_interval(theta, pi, u, v):
    """∂C/∂u ∈ [0,1] (conditional CDF)."""
    c = BB10Copula(); c.set_parameters([theta, pi])
    val = float(c.partial_derivative_C_wrt_u(u, v))
    assert -1e-6 <= val <= 1.0 + 1e-6


@given(theta=valid_theta_stable(), pi=valid_pi_stable(), u=unit, v=unit)
@settings(max_examples=40, deadline=None)
def test_partial_symmetry(theta, pi, u, v):
    """∂C/∂u(u,v) = ∂C/∂v(v,u)."""
    c = BB10Copula(); c.set_parameters([theta, pi])
    assert math.isclose(float(c.partial_derivative_C_wrt_u(u, v)),
                        float(c.partial_derivative_C_wrt_v(v, u)), rel_tol=1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Kendall's tau
# ─────────────────────────────────────────────────────────────────────────────

@given(theta=valid_theta_stable(), pi=valid_pi_stable())
@settings(max_examples=20, deadline=None)
def test_kendall_tau_in_unit_interval(theta, pi):
    """τ ∈ [0,1) for valid BB10 params."""
    c = BB10Copula(); c.set_parameters([theta, pi])
    tau = float(c.kendall_tau())
    assert 0.0 <= tau < 1.0


@pytest.mark.parametrize("theta,pis", [(2.0, [0.1, 0.3, 0.5, 0.7, 0.9])])
def test_kendall_tau_increasing_in_pi(theta, pis):
    """τ always increases with π for fixed θ (Joe 2014, p.207)."""
    c = BB10Copula()
    taus = [c.set_parameters([theta, pi]) or c.kendall_tau() for pi in pis]
    print(f"τ(π) = {[f'{t:.4f}' for t in taus]}")
    assert all(taus[i] < taus[i+1] for i in range(len(taus)-1))


@pytest.mark.parametrize("thetas,pi", [([0.5, 1.0, 2.0, 5.0, 10.0], 1.0 - 1e-6)])
def test_kendall_tau_increasing_in_theta_at_pi1(thetas, pi):
    """For π≈1, τ increases with θ (Joe 2014, p.207)."""
    c = BB10Copula()
    taus = [c.set_parameters([th, pi]) or c.kendall_tau() for th in thetas]
    print(f"τ(θ)|π≈1 = {[f'{t:.4f}' for t in taus]}")
    assert all(taus[i] < taus[i+1] for i in range(len(taus)-1))


# ─────────────────────────────────────────────────────────────────────────────
# 7. Tail dependence
# ─────────────────────────────────────────────────────────────────────────────

@given(theta=valid_theta(), pi=valid_pi())
def test_ltdc_is_zero(theta, pi):
    """λ_L = 0 (lower tail order κ_L = 2 > 1)."""
    c = BB10Copula(); c.set_parameters([theta, pi])
    assert c.LTDC() == 0.0


@given(theta=valid_theta(), pi=valid_pi())
def test_utdc_is_zero(theta, pi):
    """λ_U = 0 (upper tail order κ_U = 2 > 1)."""
    c = BB10Copula(); c.set_parameters([theta, pi])
    assert c.UTDC() == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 8. Blomqvist's beta
# ─────────────────────────────────────────────────────────────────────────────

@given(theta=valid_theta_stable(), pi=valid_pi_stable())
@settings(max_examples=30, deadline=None)
def test_blomqvist_beta_formula(theta, pi):
    """β = 4·C(½,½) − 1 matches blomqvist_beta()."""
    c = BB10Copula(); c.set_parameters([theta, pi])
    assert math.isclose(float(c.blomqvist_beta()),
                        float(4.0 * c.get_cdf(0.5, 0.5) - 1.0), rel_tol=1e-10)


@given(theta=valid_theta_stable(), pi=valid_pi_stable())
@settings(max_examples=30, deadline=None)
def test_blomqvist_beta_range(theta, pi):
    """β ∈ (0,1) for BB10 with π ∈ (0,1)."""
    c = BB10Copula(); c.set_parameters([theta, pi])
    beta = float(c.blomqvist_beta())
    assert 0.0 < beta < 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 9. IAD / AD disabled
# ─────────────────────────────────────────────────────────────────────────────

def test_iad_returns_nan():
    c = BB10Copula(); c.set_parameters([2.0, 0.5])
    assert np.isnan(c.IAD(None))


def test_ad_returns_nan():
    c = BB10Copula(); c.set_parameters([2.0, 0.5])
    assert np.isnan(c.AD(None))


# ─────────────────────────────────────────────────────────────────────────────
# 10. init_from_data  (slow)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
@given(theta=valid_theta_stable(), pi=valid_pi_stable())
@settings(max_examples=8, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_init_from_data_reproduces_tau(theta, pi):
    """init_from_data on n=2000 samples reproduces τ within ±0.15."""
    c = BB10Copula(); c.set_parameters([theta, pi])
    data = c.sample(2000, rng=np.random.default_rng(0))

    c2 = BB10Copula()
    p0 = c2.init_from_data(data[:, 0], data[:, 1])
    assert np.all(np.isfinite(p0))
    assert p0[0] > 0 and 0.0 < p0[1] < 1.0

    c2.set_parameters(p0)
    tau_fit  = c2.kendall_tau()
    tau_true = c.kendall_tau()
    note(f"θ={theta:.3f}, π={pi:.3f}, τ_true={tau_true:.4f}, τ_fit={tau_fit:.4f}")
    assert abs(tau_fit - tau_true) < 0.15


# ─────────────────────────────────────────────────────────────────────────────
# 11. Sampling  (slow)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
@given(theta=valid_theta_stable(), pi=valid_pi_stable())
@settings(max_examples=8, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_sampling_shape(theta, pi):
    """sample(n) returns (n,2) with values in (0,1)."""
    c = BB10Copula(); c.set_parameters([theta, pi])
    data = c.sample(200, rng=np.random.default_rng(42))
    assert data.shape == (200, 2)
    assert np.all(data > 0) and np.all(data < 1)


@pytest.mark.slow
@given(theta=valid_theta_stable(), pi=valid_pi_stable())
@settings(max_examples=6, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_sampling_empirical_tau_close(theta, pi):
    """Empirical τ from n=4000 samples ≈ theoretical τ (tol 0.10)."""
    c = BB10Copula(); c.set_parameters([theta, pi])
    data = c.sample(4000, rng=np.random.default_rng(0))
    tau_emp = float(sp_kendalltau(data[:, 0], data[:, 1]).correlation)
    tau_th  = c.kendall_tau()
    note(f"θ={theta:.3f}, π={pi:.3f}, τ_emp={tau_emp:.4f}, τ_th={tau_th:.4f}")
    assert math.isfinite(tau_emp)
    assert abs(tau_emp - tau_th) < 0.10


# ─────────────────────────────────────────────────────────────────────────────
# 12. Vectorisation
# ─────────────────────────────────────────────────────────────────────────────

def test_vectorised_cdf_shape():
    c = BB10Copula(); c.set_parameters([2.0, 0.5])
    u = np.linspace(0.05, 0.95, 20); v = np.linspace(0.05, 0.95, 20)
    out = c.get_cdf(u, v)
    assert out.shape == (20,) and np.all(np.isfinite(out))


def test_vectorised_pdf_shape():
    c = BB10Copula(); c.set_parameters([2.0, 0.5])
    u = np.linspace(0.05, 0.95, 20); v = np.linspace(0.05, 0.95, 20)
    out = c.get_pdf(u, v)
    assert out.shape == (20,) and np.all(np.isfinite(out)) and np.all(out >= 0.0)


def test_vectorised_partial_shape():
    c = BB10Copula(); c.set_parameters([2.0, 0.5])
    u = np.linspace(0.05, 0.95, 15); v = np.linspace(0.05, 0.95, 15)
    assert c.partial_derivative_C_wrt_u(u, v).shape == (15,)
    assert c.partial_derivative_C_wrt_v(u, v).shape == (15,)


def test_scalar_and_array_agree():
    c = BB10Copula(); c.set_parameters([2.0, 0.5])
    for u0, v0 in [(0.2, 0.7), (0.5, 0.5), (0.8, 0.3)]:
        scalar = float(c.get_cdf(u0, v0))
        array  = float(c.get_cdf(np.array([u0]), np.array([v0]))[0])
        assert math.isclose(scalar, array, rel_tol=1e-12)