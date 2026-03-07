"""
Comprehensive unit-test suite for the bivariate **BB5** copula.

BB5 is a two-parameter extreme-value copula (Joe & Hu 1996):

    C(u,v; θ,δ) = exp{−[xθ + yθ − (x^{−δθ} + y^{−δθ})^{−1/δ}]^{1/θ}}

where x = −log u, y = −log v, θ ≥ 1, δ > 0.

Properties (Joe 2014, §4.21):
  • Gumbel family when δ→0⁺; Galambos family when θ=1.
  • C⁺ as θ→∞ or δ→∞.
  • λ_U = 2 − (2 − 2^{−1/δ})^{1/θ}  > 0 (upper tail dependence).
  • λ_L = 0  (no lower tail dependence).
  • β = 2^{λ_U} − 1  (Blomqvist).
  • Concordance increases with both θ and δ.

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

from CopulaFurtif.core.copulas.domain.models.archimedean.BB5 import BB5Copula


# ─────────────────────────────────────────────────────────────────────────────
# Strategies & helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.composite
def valid_theta(draw):
    """θ ∈ (1, 10) — BB5 requires θ ≥ 1."""
    return draw(st.floats(min_value=1.0, max_value=10.0,
                          exclude_min=True, allow_nan=False, allow_infinity=False))


@st.composite
def valid_delta(draw):
    """δ ∈ (1e-4, 10) — strictly positive."""
    return draw(st.floats(min_value=1e-4, max_value=10.0,
                          exclude_min=True, allow_nan=False, allow_infinity=False))


@st.composite
def valid_theta_stable(draw):
    """θ ∈ (1.1, 6)."""
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
    c = BB5Copula()
    c.set_parameters([theta, delta])
    assert math.isclose(c.get_parameters()[0], theta, rel_tol=1e-12)
    assert math.isclose(c.get_parameters()[1], delta, rel_tol=1e-12)


@pytest.mark.parametrize("theta,delta", [
    (0.5, 1.5), (1.0, 1.5), (0.0, 1.5), (-1.0, 1.5),
])
def test_theta_out_of_bounds(theta, delta):
    """θ ≤ 1 must raise ValueError."""
    c = BB5Copula()
    with pytest.raises(ValueError):
        c.set_parameters([theta, delta])


@pytest.mark.parametrize("theta,delta", [
    (2.0, 0.0), (2.0, -1.0),
])
def test_delta_out_of_bounds(theta, delta):
    """δ ≤ 0 must raise ValueError."""
    c = BB5Copula()
    with pytest.raises(ValueError):
        c.set_parameters([theta, delta])


# ─────────────────────────────────────────────────────────────────────────────
# 2. CDF invariants
# ─────────────────────────────────────────────────────────────────────────────

@given(theta=valid_theta(), delta=valid_delta(), u=unit, v=unit)
def test_cdf_in_unit_interval(theta, delta, u, v):
    c = BB5Copula(); c.set_parameters([theta, delta])
    val = float(c.get_cdf(u, v))
    assert 0.0 <= val <= 1.0


@given(theta=valid_theta(), delta=valid_delta(), u=unit, v=unit)
def test_cdf_symmetry(theta, delta, u, v):
    """BB5 is exchangeable: C(u,v) = C(v,u)."""
    c = BB5Copula(); c.set_parameters([theta, delta])
    assert math.isclose(float(c.get_cdf(u, v)), float(c.get_cdf(v, u)), rel_tol=1e-10)


@given(theta=valid_theta(), delta=valid_delta(), u1=unit, u2=unit, v=unit)
def test_cdf_monotone_in_u(theta, delta, u1, u2, v):
    if u1 > u2: u1, u2 = u2, u1
    c = BB5Copula(); c.set_parameters([theta, delta])
    assert float(c.get_cdf(u1, v)) <= float(c.get_cdf(u2, v)) + 1e-12


@pytest.mark.parametrize("theta,delta,eps", [(2.0,1.5,1e-3),(3.0,2.0,1e-3)])
def test_cdf_boundary_v_zero(theta, delta, eps):
    """C(u, 0) ≈ 0."""
    c = BB5Copula(); c.set_parameters([theta, delta])
    for u in [0.2, 0.5, 0.9]:
        assert float(c.get_cdf(u, eps)) < 5e-3


@pytest.mark.parametrize("theta,delta,eps", [(2.0,1.5,1e-3),(3.0,2.0,1e-3)])
def test_cdf_boundary_v_one(theta, delta, eps):
    """C(u, 1) ≈ u."""
    c = BB5Copula(); c.set_parameters([theta, delta])
    for u in [0.2, 0.5, 0.9]:
        assert math.isclose(float(c.get_cdf(u, 1-eps)), u, abs_tol=5e-3)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Fréchet bounds
# ─────────────────────────────────────────────────────────────────────────────

@given(theta=valid_theta(), delta=valid_delta(), u=unit, v=unit)
def test_frechet_lower_bound(theta, delta, u, v):
    c = BB5Copula(); c.set_parameters([theta, delta])
    assert float(c.get_cdf(u, v)) >= max(u+v-1.0, 0.0) - 1e-12


@given(theta=valid_theta(), delta=valid_delta(), u=unit, v=unit)
def test_frechet_upper_bound(theta, delta, u, v):
    c = BB5Copula(); c.set_parameters([theta, delta])
    assert float(c.get_cdf(u, v)) <= min(u, v) + 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# 4. PDF invariants
# ─────────────────────────────────────────────────────────────────────────────

@given(theta=valid_theta(), delta=valid_delta(), u=unit, v=unit)
def test_pdf_nonnegative(theta, delta, u, v):
    c = BB5Copula(); c.set_parameters([theta, delta])
    assert float(c.get_pdf(u, v)) >= 0.0


@pytest.mark.parametrize("theta,delta,u,v,rtol", [
    (2.0, 1.5, 0.5, 0.5, 1e-3),
    (2.0, 1.5, 0.3, 0.6, 1e-3),
    (2.0, 1.5, 0.2, 0.8, 1e-3),
    (3.0, 2.0, 0.4, 0.4, 1e-3),
])
def test_pdf_matches_finite_diff(theta, delta, u, v, rtol):
    """Analytical PDF matches central finite-difference of CDF."""
    c = BB5Copula(); c.set_parameters([theta, delta])
    ana = float(c.get_pdf(u, v))
    fd  = float(_cdf_fd(c, u, v))
    assert math.isclose(ana, fd, rel_tol=rtol, abs_tol=1e-5)


@given(theta=valid_theta_stable(), delta=valid_delta_stable(), u=unit, v=unit)
@settings(max_examples=60, deadline=None)
def test_pdf_symmetry(theta, delta, u, v):
    """c(u,v) = c(v,u)."""
    c = BB5Copula(); c.set_parameters([theta, delta])
    assert math.isclose(float(c.get_pdf(u, v)), float(c.get_pdf(v, u)), rel_tol=1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# 5. H-functions (partial derivatives)
# ─────────────────────────────────────────────────────────────────────────────

@given(theta=valid_theta_stable(), delta=valid_delta_stable(), u=unit, v=unit)
@settings(max_examples=60, deadline=None)
def test_partial_u_matches_fd(theta, delta, u, v):
    c = BB5Copula(); c.set_parameters([theta, delta])
    assert math.isclose(float(c.partial_derivative_C_wrt_u(u, v)),
                        float(_partial_u_fd(c, u, v)), rel_tol=1e-3, abs_tol=1e-4)


@given(theta=valid_theta_stable(), delta=valid_delta_stable(), u=unit, v=unit)
@settings(max_examples=60, deadline=None)
def test_partial_v_matches_fd(theta, delta, u, v):
    c = BB5Copula(); c.set_parameters([theta, delta])
    assert math.isclose(float(c.partial_derivative_C_wrt_v(u, v)),
                        float(_partial_v_fd(c, u, v)), rel_tol=1e-3, abs_tol=1e-4)


@given(theta=valid_theta_stable(), delta=valid_delta_stable(), u=unit, v=unit)
@settings(max_examples=40, deadline=None)
def test_partial_u_in_unit_interval(theta, delta, u, v):
    """∂C/∂u ∈ [0,1] (it is a conditional CDF)."""
    c = BB5Copula(); c.set_parameters([theta, delta])
    val = float(c.partial_derivative_C_wrt_u(u, v))
    assert -1e-6 <= val <= 1.0 + 1e-6


@given(theta=valid_theta_stable(), delta=valid_delta_stable(), u=unit, v=unit)
@settings(max_examples=40, deadline=None)
def test_partial_symmetry(theta, delta, u, v):
    """∂C/∂u(u,v) = ∂C/∂v(v,u) by exchangeability."""
    c = BB5Copula(); c.set_parameters([theta, delta])
    assert math.isclose(float(c.partial_derivative_C_wrt_u(u, v)),
                        float(c.partial_derivative_C_wrt_v(v, u)), rel_tol=1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Kendall's tau
# ─────────────────────────────────────────────────────────────────────────────

@given(theta=valid_theta_stable(), delta=valid_delta_stable())
@settings(max_examples=20, deadline=None)
def test_kendall_tau_in_unit_interval(theta, delta):
    """τ ∈ (0,1] for valid BB5 params."""
    c = BB5Copula(); c.set_parameters([theta, delta])
    tau = float(c.kendall_tau())
    assert 0.0 < tau <= 1.0


@pytest.mark.parametrize("thetas,delta", [([1.2, 1.8, 2.5, 4.0, 7.0], 1.5)])
def test_kendall_tau_increasing_in_theta(thetas, delta):
    """τ increases as θ increases (concordance ↑ with θ)."""
    c = BB5Copula()
    taus = []
    for th in thetas:
        c.set_parameters([th, delta]); taus.append(c.kendall_tau())
    print(f"τ(θ) = {[f'{t:.4f}' for t in taus]}")
    assert all(taus[i] < taus[i+1] for i in range(len(taus)-1))


@pytest.mark.parametrize("theta,deltas", [(2.0, [0.2, 0.5, 1.0, 2.0, 5.0])])
def test_kendall_tau_increasing_in_delta(theta, deltas):
    """τ increases as δ increases (concordance ↑ with δ)."""
    c = BB5Copula()
    taus = []
    for de in deltas:
        c.set_parameters([theta, de]); taus.append(c.kendall_tau())
    print(f"τ(δ) = {[f'{t:.4f}' for t in taus]}")
    assert all(taus[i] < taus[i+1] for i in range(len(taus)-1))


# ─────────────────────────────────────────────────────────────────────────────
# 7. Tail dependence
# ─────────────────────────────────────────────────────────────────────────────

@given(theta=valid_theta(), delta=valid_delta())
def test_ltdc_is_zero(theta, delta):
    """BB5 has no lower tail dependence."""
    c = BB5Copula(); c.set_parameters([theta, delta])
    assert c.LTDC() == 0.0


@given(theta=valid_theta(), delta=valid_delta())
def test_utdc_formula(theta, delta):
    """λ_U = 2 − (2 − 2^{−1/δ})^{1/θ} ∈ (0,1]."""
    c = BB5Copula(); c.set_parameters([theta, delta])
    lam = float(c.UTDC())
    expected = 2.0 - (2.0 - 2.0**(-1.0/delta))**(1.0/theta)
    assert math.isclose(lam, expected, rel_tol=1e-12)
    assert 0.0 < lam <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 8. Blomqvist's beta
# ─────────────────────────────────────────────────────────────────────────────

@given(theta=valid_theta_stable(), delta=valid_delta_stable())
@settings(max_examples=30, deadline=None)
def test_blomqvist_beta_formula(theta, delta):
    """β = 2^{λ_U} − 1  matches  4·C(½,½) − 1."""
    c = BB5Copula(); c.set_parameters([theta, delta])
    beta_formula = float(c.blomqvist_beta())
    beta_direct  = float(4.0 * c.get_cdf(0.5, 0.5) - 1.0)
    assert math.isclose(beta_formula, beta_direct, rel_tol=1e-6, abs_tol=1e-8)


@given(theta=valid_theta_stable(), delta=valid_delta_stable())
@settings(max_examples=30, deadline=None)
def test_blomqvist_beta_range(theta, delta):
    """β ∈ (0,1) for BB5."""
    c = BB5Copula(); c.set_parameters([theta, delta])
    beta = float(c.blomqvist_beta())
    assert 0.0 < beta < 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 9. IAD / AD disabled
# ─────────────────────────────────────────────────────────────────────────────

def test_iad_returns_nan():
    c = BB5Copula(); c.set_parameters([2.0, 1.5])
    assert np.isnan(c.IAD(None))


def test_ad_returns_nan():
    c = BB5Copula(); c.set_parameters([2.0, 1.5])
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
    BB5 has 2 params, but τ alone under-identifies the system; we only
    check that the fitted model's τ is close to the true τ.
    """
    c = BB5Copula(); c.set_parameters([theta, delta])
    data = c.sample(2000, rng=np.random.default_rng(0))

    c2 = BB5Copula()
    p0 = c2.init_from_data(data[:, 0], data[:, 1])
    assert np.all(np.isfinite(p0))
    assert p0[0] > 1.0 and p0[1] > 0.0

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
    """sample(n) returns (n, 2) with values in (0, 1)."""
    c = BB5Copula(); c.set_parameters([theta, delta])
    data = c.sample(200, rng=np.random.default_rng(42))
    assert data.shape == (200, 2)
    assert np.all(data > 0) and np.all(data < 1)


@pytest.mark.slow
@given(theta=valid_theta_stable(), delta=valid_delta_stable())
@settings(max_examples=6, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_sampling_empirical_tau_close(theta, delta):
    """Empirical τ from n=4000 samples ≈ theoretical τ (tol 0.10)."""
    c = BB5Copula(); c.set_parameters([theta, delta])
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
    c = BB5Copula(); c.set_parameters([2.0, 1.5])
    u = np.linspace(0.05, 0.95, 20); v = np.linspace(0.05, 0.95, 20)
    out = c.get_cdf(u, v)
    assert out.shape == (20,) and np.all(np.isfinite(out))


def test_vectorised_pdf_shape():
    c = BB5Copula(); c.set_parameters([2.0, 1.5])
    u = np.linspace(0.05, 0.95, 20); v = np.linspace(0.05, 0.95, 20)
    out = c.get_pdf(u, v)
    assert out.shape == (20,) and np.all(np.isfinite(out)) and np.all(out >= 0.0)


def test_vectorised_partial_shape():
    c = BB5Copula(); c.set_parameters([2.0, 1.5])
    u = np.linspace(0.05, 0.95, 15); v = np.linspace(0.05, 0.95, 15)
    assert c.partial_derivative_C_wrt_u(u, v).shape == (15,)
    assert c.partial_derivative_C_wrt_v(u, v).shape == (15,)


def test_scalar_and_array_agree():
    """Scalar and length-1 array calls return same values."""
    c = BB5Copula(); c.set_parameters([2.0, 1.5])
    for u0, v0 in [(0.2, 0.7), (0.5, 0.5), (0.8, 0.3)]:
        scalar = float(c.get_cdf(u0, v0))
        array  = float(c.get_cdf(np.array([u0]), np.array([v0]))[0])
        assert math.isclose(scalar, array, rel_tol=1e-12)