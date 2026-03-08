"""
Comprehensive unit-test suite for the bivariate **BB7** (Joe-Clayton) copula.

BB7 is an Archimedean copula (Joe & Hu 1996):

    C(u,v; θ,δ) = 1 − (1 − [(1−ū^θ)^{−δ} + (1−v̄^θ)^{−δ} − 1]^{−1/δ})^{1/θ}

where ū = 1−u, v̄ = 1−v, θ > 0, δ > 0.

Properties (Joe 2014, §4.23):
  • MTCJ (Clayton) family when θ=1; Joe/B5 family as δ→0⁺.
  • C⁺ as θ→∞ or δ→∞.
  • λ_L = 2^{-1/δ}   (lower tail dependence, independent of θ).
  • λ_U = 2 − 2^{1/θ} (upper tail dependence, independent of δ).
  • Blomqvist β = 4·C(½,½) − 1.
  • Concordance increases with δ always; with θ only when δ ≤ 1 in general.
  • Archimedean generator φ(t) = [(1−(1−t)^θ)^{−δ} − 1]/δ.

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
from scipy.special import beta as beta_fn

from CopulaFurtif.core.copulas.domain.models.archimedean.BB7 import BB7Copula


# ─────────────────────────────────────────────────────────────────────────────
# Strategies & helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.composite
def valid_theta(draw):
    """θ ∈ (1e-5, 10) — strictly positive per class bounds."""
    return draw(st.floats(min_value=1e-5, max_value=10.0,
                          exclude_min=True, allow_nan=False, allow_infinity=False))


@st.composite
def valid_delta(draw):
    """δ ∈ (1e-5, 10) — strictly positive per class bounds."""
    return draw(st.floats(min_value=1e-5, max_value=10.0,
                          exclude_min=True, allow_nan=False, allow_infinity=False))


@st.composite
def valid_theta_stable(draw):
    """θ ∈ (1.1, 6) — avoids near-zero instability."""
    return draw(st.floats(min_value=1.1, max_value=6.0,
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
    """set_parameters → get_parameters is lossless."""
    c = BB7Copula()
    c.set_parameters([theta, delta])
    assert math.isclose(c.get_parameters()[0], theta, rel_tol=1e-12)
    assert math.isclose(c.get_parameters()[1], delta, rel_tol=1e-12)


@pytest.mark.parametrize("theta,delta", [
    (0.0, 1.5), (-1.0, 1.5), (-1e-3, 1.5),
])
def test_theta_out_of_bounds(theta, delta):
    """θ ≤ 0 must raise ValueError."""
    c = BB7Copula()
    with pytest.raises(ValueError):
        c.set_parameters([theta, delta])


@pytest.mark.parametrize("theta,delta", [
    (2.0, 0.0), (2.0, -0.5),
])
def test_delta_out_of_bounds(theta, delta):
    """δ ≤ 0 must raise ValueError."""
    c = BB7Copula()
    with pytest.raises(ValueError):
        c.set_parameters([theta, delta])


# ─────────────────────────────────────────────────────────────────────────────
# 2. CDF invariants
# ─────────────────────────────────────────────────────────────────────────────

@given(theta=valid_theta(), delta=valid_delta(), u=unit, v=unit)
def test_cdf_in_unit_interval(theta, delta, u, v):
    c = BB7Copula(); c.set_parameters([theta, delta])
    assert 0.0 <= float(c.get_cdf(u, v)) <= 1.0


@given(theta=valid_theta(), delta=valid_delta(), u=unit, v=unit)
def test_cdf_symmetry(theta, delta, u, v):
    """BB7 is exchangeable: C(u,v) = C(v,u)."""
    c = BB7Copula(); c.set_parameters([theta, delta])
    assert math.isclose(float(c.get_cdf(u, v)), float(c.get_cdf(v, u)), rel_tol=1e-10)


@given(theta=valid_theta(), delta=valid_delta(), u1=unit, u2=unit, v=unit)
def test_cdf_monotone_in_u(theta, delta, u1, u2, v):
    if u1 > u2: u1, u2 = u2, u1
    c = BB7Copula(); c.set_parameters([theta, delta])
    assert float(c.get_cdf(u1, v)) <= float(c.get_cdf(u2, v)) + 1e-12


@pytest.mark.parametrize("theta,delta,eps", [(2.0,1.5,1e-3),(3.0,2.0,1e-3)])
def test_cdf_boundary_v_zero(theta, delta, eps):
    """C(u, 0) ≈ 0."""
    c = BB7Copula(); c.set_parameters([theta, delta])
    for u in [0.2, 0.5, 0.9]:
        assert float(c.get_cdf(u, eps)) < 5e-3


@pytest.mark.parametrize("theta,delta,eps", [(2.0,1.5,1e-3),(3.0,2.0,1e-3)])
def test_cdf_boundary_v_one(theta, delta, eps):
    """C(u, 1) ≈ u."""
    c = BB7Copula(); c.set_parameters([theta, delta])
    for u in [0.2, 0.5, 0.9]:
        assert math.isclose(float(c.get_cdf(u, 1-eps)), u, abs_tol=5e-3)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Fréchet bounds
# ─────────────────────────────────────────────────────────────────────────────

@given(theta=valid_theta_stable(), delta=valid_delta_stable(), u=unit, v=unit)
@settings(max_examples=50, deadline=None)
def test_frechet_lower_bound(theta, delta, u, v):
    """C(u,v) ≥ max(u+v−1, 0) — tested on stable param range."""
    c = BB7Copula(); c.set_parameters([theta, delta])
    assert float(c.get_cdf(u, v)) >= max(u+v-1.0, 0.0) - 1e-10


@given(theta=valid_theta_stable(), delta=valid_delta_stable(), u=unit, v=unit)
@settings(max_examples=50, deadline=None)
def test_frechet_upper_bound(theta, delta, u, v):
    """C(u,v) ≤ min(u,v) (Fréchet upper) — tested on stable param range."""
    c = BB7Copula(); c.set_parameters([theta, delta])
    assert float(c.get_cdf(u, v)) <= min(u, v) + 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# 4. PDF invariants
# ─────────────────────────────────────────────────────────────────────────────

@given(theta=valid_theta_stable(), delta=valid_delta_stable(), u=unit, v=unit)
def test_pdf_nonnegative(theta, delta, u, v):
    c = BB7Copula(); c.set_parameters([theta, delta])
    assert float(c.get_pdf(u, v)) >= 0.0


@pytest.mark.parametrize("theta,delta,u,v,rtol", [
    (2.0, 1.5, 0.5, 0.5, 1e-3),
    (2.0, 1.5, 0.3, 0.6, 1e-3),
    (3.0, 2.0, 0.4, 0.4, 1e-3),
])
def test_pdf_matches_finite_diff(theta, delta, u, v, rtol):
    """Analytical PDF matches mixed finite-difference of CDF."""
    c = BB7Copula(); c.set_parameters([theta, delta])
    ana = float(c.get_pdf(u, v))
    fd  = float(_cdf_fd(c, u, v))
    assert math.isclose(ana, fd, rel_tol=rtol, abs_tol=1e-5)


@given(theta=valid_theta_stable(), delta=valid_delta_stable(), u=unit, v=unit)
@settings(max_examples=60, deadline=None)
def test_pdf_symmetry(theta, delta, u, v):
    """c(u,v) = c(v,u)."""
    c = BB7Copula(); c.set_parameters([theta, delta])
    assert math.isclose(float(c.get_pdf(u, v)), float(c.get_pdf(v, u)), rel_tol=1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# 5. H-functions (partial derivatives)
# ─────────────────────────────────────────────────────────────────────────────

@given(theta=valid_theta_stable(), delta=valid_delta_stable(), u=unit, v=unit)
@settings(max_examples=60, deadline=None)
def test_partial_u_matches_fd(theta, delta, u, v):
    c = BB7Copula(); c.set_parameters([theta, delta])
    assert math.isclose(float(c.partial_derivative_C_wrt_u(u, v)),
                        float(_partial_u_fd(c, u, v)), rel_tol=2e-2, abs_tol=2e-3)


@given(theta=valid_theta_stable(), delta=valid_delta_stable(), u=unit, v=unit)
@settings(max_examples=60, deadline=None)
def test_partial_v_matches_fd(theta, delta, u, v):
    c = BB7Copula(); c.set_parameters([theta, delta])
    assert math.isclose(float(c.partial_derivative_C_wrt_v(u, v)),
                        float(_partial_v_fd(c, u, v)), rel_tol=2e-2, abs_tol=2e-3)


@given(theta=valid_theta_stable(), delta=valid_delta_stable(), u=unit, v=unit)
@settings(max_examples=40, deadline=None)
def test_partial_u_in_unit_interval(theta, delta, u, v):
    """∂C/∂u ∈ [0,1] (conditional CDF)."""
    c = BB7Copula(); c.set_parameters([theta, delta])
    val = float(c.partial_derivative_C_wrt_u(u, v))
    assert -1e-6 <= val <= 1.0 + 1e-6


@given(theta=valid_theta_stable(), delta=valid_delta_stable(), u=unit, v=unit)
@settings(max_examples=40, deadline=None)
def test_partial_symmetry(theta, delta, u, v):
    """∂C/∂u(u,v) = ∂C/∂v(v,u) by exchangeability."""
    c = BB7Copula(); c.set_parameters([theta, delta])
    assert math.isclose(float(c.partial_derivative_C_wrt_u(u, v)),
                        float(c.partial_derivative_C_wrt_v(v, u)), rel_tol=1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Kendall's tau
# ─────────────────────────────────────────────────────────────────────────────

@given(theta=valid_theta_stable(), delta=valid_delta_stable())
@settings(max_examples=20, deadline=None)
def test_kendall_tau_in_unit_interval(theta, delta):
    """τ ∈ (0,1) for valid BB7 params."""
    c = BB7Copula(); c.set_parameters([theta, delta])
    tau = float(c.kendall_tau())
    assert 0.0 < tau < 1.0


@pytest.mark.parametrize("theta,delta,expected,tol", [
    # For 1 < θ < 2: τ = 1 − 2/(δ(2−θ)) + 4/(δθ²)·B(δ+2, 2/θ−1)
    (1.5, 1.5, None, 1e-4),   # closed form checked internally
    (1.8, 2.0, None, 1e-4),
])
def test_kendall_tau_closed_form(theta, delta, expected, tol):
    """For 1 < θ < 2, τ matches the closed-form expression."""
    c = BB7Copula(); c.set_parameters([theta, delta])
    tau_impl = c.kendall_tau()
    tau_cf = 1 - 2/(delta*(2-theta)) + 4/(delta*theta**2) * beta_fn(delta+2, 2/theta-1)
    print(f"τ_impl={tau_impl:.6f}  τ_cf={tau_cf:.6f}")
    assert math.isclose(tau_impl, tau_cf, rel_tol=tol, abs_tol=tol)


@pytest.mark.parametrize("thetas,delta", [
    ([1.2, 1.8, 2.5, 4.0, 7.0], 0.5),  # δ ≤ 1: concordance guaranteed ↑ in θ
])
def test_kendall_tau_increasing_in_theta(thetas, delta):
    """τ increases as θ increases when δ ≤ 1 (Joe 2014, p.203)."""
    c = BB7Copula()
    taus = []
    for th in thetas:
        c.set_parameters([th, delta]); taus.append(c.kendall_tau())
    print(f"τ(θ) = {[f'{t:.4f}' for t in taus]}")
    assert all(taus[i] < taus[i+1] for i in range(len(taus)-1))


@pytest.mark.parametrize("theta,deltas", [
    (2.0, [0.2, 0.5, 1.0, 2.0, 5.0]),
])
def test_kendall_tau_increasing_in_delta(theta, deltas):
    """τ always increases as δ increases (Joe 2014, p.203)."""
    c = BB7Copula()
    taus = []
    for de in deltas:
        c.set_parameters([theta, de]); taus.append(c.kendall_tau())
    print(f"τ(δ) = {[f'{t:.4f}' for t in taus]}")
    assert all(taus[i] < taus[i+1] for i in range(len(taus)-1))


# ─────────────────────────────────────────────────────────────────────────────
# 7. Tail dependence
# ─────────────────────────────────────────────────────────────────────────────

@given(theta=valid_theta_stable(), delta=valid_delta_stable())
@settings(max_examples=50, deadline=None)
def test_ltdc_formula(theta, delta):
    """λ_L = 2^{-1/δ} — independent of θ (tested on stable range)."""
    c = BB7Copula(); c.set_parameters([theta, delta])
    expected = float(np.exp(-np.log(2) / delta))
    assert math.isclose(float(c.LTDC()), expected, rel_tol=1e-10, abs_tol=1e-300)


@given(theta=valid_theta_stable(), delta=valid_delta_stable())
@settings(max_examples=50, deadline=None)
def test_utdc_formula(theta, delta):
    """λ_U = 2 − 2^{1/θ} — independent of δ (tested on stable range)."""
    c = BB7Copula(); c.set_parameters([theta, delta])
    expected = 2.0 - float(np.exp(np.log(2) / theta))
    assert math.isclose(float(c.UTDC()), expected, rel_tol=1e-10)


@given(theta=valid_theta_stable(), delta=valid_delta_stable())
@settings(max_examples=30, deadline=None)
def test_ltdc_in_unit_interval(theta, delta):
    """λ_L ∈ (0,1)."""
    c = BB7Copula(); c.set_parameters([theta, delta])
    assert 0.0 < float(c.LTDC()) < 1.0


@given(theta=valid_theta_stable(), delta=valid_delta_stable())
@settings(max_examples=30, deadline=None)
def test_utdc_negative_for_large_theta(theta, delta):
    """λ_U = 2 − 2^{1/θ} < 0 when θ < 1; ≥ 0 for θ ≥ 1."""
    # For θ ∈ (1.1, 6): λ_U = 2 − 2^{1/θ} ∈ (0, 1) since 1 < 2^{1/θ} < 2
    c = BB7Copula(); c.set_parameters([theta, delta])
    lam_U = float(c.UTDC())
    # For our stable range θ ∈ (1.1, 6), λ_U ∈ (0, 1)
    if theta > 1.0:
        assert lam_U >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 8. Blomqvist's beta
# ─────────────────────────────────────────────────────────────────────────────

@given(theta=valid_theta_stable(), delta=valid_delta_stable())
@settings(max_examples=30, deadline=None)
def test_blomqvist_beta_formula(theta, delta):
    """β = 4·C(½,½) − 1 matches blomqvist_beta()."""
    c = BB7Copula(); c.set_parameters([theta, delta])
    beta_method = float(c.blomqvist_beta())
    beta_direct = float(4.0 * c.get_cdf(0.5, 0.5) - 1.0)
    assert math.isclose(beta_method, beta_direct, rel_tol=1e-10)


@given(theta=valid_theta_stable(), delta=valid_delta_stable())
@settings(max_examples=30, deadline=None)
def test_blomqvist_beta_range(theta, delta):
    """β ∈ (0,1) for BB7 (positively dependent)."""
    c = BB7Copula(); c.set_parameters([theta, delta])
    beta = float(c.blomqvist_beta())
    assert 0.0 < beta < 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 9. IAD / AD disabled
# ─────────────────────────────────────────────────────────────────────────────

def test_iad_returns_nan():
    c = BB7Copula(); c.set_parameters([2.0, 1.5])
    assert np.isnan(c.IAD(None))


def test_ad_returns_nan():
    c = BB7Copula(); c.set_parameters([2.0, 1.5])
    assert np.isnan(c.AD(None))


# ─────────────────────────────────────────────────────────────────────────────
# 10. init_from_data  (slow)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
@given(theta=valid_theta_stable(), delta=valid_delta_stable())
@settings(max_examples=8, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_init_from_data_reproduces_tau(theta, delta):
    """
    init_from_data on n=2000 samples should reproduce τ within ±0.15.
    """
    c = BB7Copula(); c.set_parameters([theta, delta])
    data = c.sample(2000, rng=np.random.default_rng(0))

    c2 = BB7Copula()
    p0 = c2.init_from_data(data[:, 0], data[:, 1])
    assert np.all(np.isfinite(p0)) and np.all(p0 > 0)

    c2.set_parameters(p0)
    tau_fit  = c2.kendall_tau()
    tau_true = c.kendall_tau()
    note(f"θ={theta:.3f}, δ={delta:.3f}, τ_true={tau_true:.4f}, τ_fit={tau_fit:.4f}")
    assert abs(tau_fit - tau_true) < 0.15


# ─────────────────────────────────────────────────────────────────────────────
# 11. Sampling  (slow)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
@given(theta=valid_theta_stable(), delta=valid_delta_stable())
@settings(max_examples=8, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_sampling_shape(theta, delta):
    """sample(n) returns (n,2) with values in (0,1)."""
    c = BB7Copula(); c.set_parameters([theta, delta])
    data = c.sample(200, rng=np.random.default_rng(42))
    assert data.shape == (200, 2)
    assert np.all(data > 0) and np.all(data < 1)


@pytest.mark.slow
@given(theta=valid_theta_stable(), delta=valid_delta_stable())
@settings(max_examples=6, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_sampling_empirical_tau_close(theta, delta):
    """Empirical τ from n=4000 samples ≈ theoretical τ (tol 0.10)."""
    c = BB7Copula(); c.set_parameters([theta, delta])
    data = c.sample(4000, rng=np.random.default_rng(0))
    tau_emp = float(sp_kendalltau(data[:, 0], data[:, 1]).correlation)
    tau_th  = c.kendall_tau()
    note(f"θ={theta:.3f}, δ={delta:.3f}, τ_emp={tau_emp:.4f}, τ_th={tau_th:.4f}")
    assert math.isfinite(tau_emp)
    assert abs(tau_emp - tau_th) < 0.10


# ─────────────────────────────────────────────────────────────────────────────
# 12. Vectorisation
# ─────────────────────────────────────────────────────────────────────────────

def test_vectorised_cdf_shape():
    c = BB7Copula(); c.set_parameters([2.0, 1.5])
    u = np.linspace(0.05, 0.95, 20); v = np.linspace(0.05, 0.95, 20)
    out = c.get_cdf(u, v)
    assert out.shape == (20,) and np.all(np.isfinite(out))


def test_vectorised_pdf_shape():
    c = BB7Copula(); c.set_parameters([2.0, 1.5])
    u = np.linspace(0.05, 0.95, 20); v = np.linspace(0.05, 0.95, 20)
    out = c.get_pdf(u, v)
    assert out.shape == (20,) and np.all(np.isfinite(out)) and np.all(out >= 0.0)


def test_vectorised_partial_shape():
    c = BB7Copula(); c.set_parameters([2.0, 1.5])
    u = np.linspace(0.05, 0.95, 15); v = np.linspace(0.05, 0.95, 15)
    assert c.partial_derivative_C_wrt_u(u, v).shape == (15,)
    assert c.partial_derivative_C_wrt_v(u, v).shape == (15,)


def test_scalar_and_array_agree():
    c = BB7Copula(); c.set_parameters([2.0, 1.5])
    for u0, v0 in [(0.2, 0.7), (0.5, 0.5), (0.8, 0.3)]:
        scalar = float(c.get_cdf(u0, v0))
        array  = float(c.get_cdf(np.array([u0]), np.array([v0]))[0])
        assert math.isclose(scalar, array, rel_tol=1e-12)