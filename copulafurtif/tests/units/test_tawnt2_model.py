"""
Comprehensive unit-test suite for the bivariate **Tawn Type-2** copula.

Tawn Type-2 is an asymmetric extreme-value copula (Joe 2014, §4.8.1;
Tadi & Witzany 2025, Table 2) with Pickands dependence function
(using t = x/s, x = −ln u, y = −ln v, s = x+y):

    A(t) = (1 − β)(1 − t) + [t^θ + (β(1 − t))^θ]^{1/θ}

Parameters:  θ ∈ (1, 500),  β ∈ (0, 1)   — strict open bounds.

Properties:
  • **Asymmetric**: C(u,v) ≠ C(v,u) for generic β ∈ (0, 1).
  • β → 0  ⇒  independence copula  C⊥.
  • β → 1  ⇒  Gumbel copula with parameter θ.
  • θ → 1  ⇒  independence copula (for any β).
  • λ_L = 0 always;  λ_U = 2 − 2A(½) > 0 when β > 0 and θ > 1.
  • τ ∈ [0, 1), computed by Gauss–Legendre quadrature.
  • C_T2(u,v; θ,β) = C_T1(v,u; θ,β)  (mirror of Tawn Type-1).

Run with:
    pytest -q                  # fast tests
    pytest -q -m slow          # include heavy sampling / init checks

IMPORTANT: The real CopulaParameters._validate uses STRICT inequalities
(lo < val < hi), so boundary values 0.0, 1.0 for β and 1.0 for θ are
rejected.  Tests use ε-offset values (e.g. 1e-8, 1-1e-8) for limit cases.
"""

import math

import numpy as np
import pytest
from hypothesis import given, settings, note, HealthCheck
from hypothesis import strategies as st
from scipy.stats import kendalltau as sp_kendalltau

from CopulaFurtif.core.copulas.domain.models.archimedean.TawnT2 import TawnT2Copula


# ── Near-boundary constants (strict open bounds) ─────────────────────────────
_BETA_LO = 1e-8          # β → 0 (independence limit)
_BETA_HI = 1.0 - 1e-8    # β → 1 (Gumbel limit)
_THETA_LO = 1.0 + 1e-8   # θ → 1 (independence limit)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures & Hypothesis strategies
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def copula_default():
    """Tawn-T2 copula with θ = 2.0, β = 0.5 for deterministic tests."""
    c = TawnT2Copula()
    c.set_parameters([2.0, 0.5])
    return c


@st.composite
def valid_theta(draw):
    """θ ∈ (1.001, 499) — strictly inside bounds."""
    return draw(st.floats(
        min_value=1.001, max_value=499.0,
        allow_nan=False, allow_infinity=False,
    ))


@st.composite
def valid_beta(draw):
    """β ∈ (1e-6, 1-1e-6) — strictly inside (0, 1)."""
    return draw(st.floats(
        min_value=1e-6, max_value=1.0 - 1e-6,
        allow_nan=False, allow_infinity=False,
    ))


@st.composite
def valid_theta_stable(draw):
    """θ ∈ (1.01, 15) — numerically comfortable range."""
    return draw(st.floats(
        min_value=1.01, max_value=15.0,
        allow_nan=False, allow_infinity=False,
    ))


@st.composite
def valid_beta_stable(draw):
    """β ∈ (0.05, 0.95) — away from degenerate boundaries."""
    return draw(st.floats(
        min_value=0.05, max_value=0.95,
        allow_nan=False, allow_infinity=False,
    ))


unit = st.floats(min_value=1e-3, max_value=0.999, allow_nan=False)


# ── Numerical helpers ────────────────────────────────────────────────────────

def _clip01(x, eps=1e-12):
    return min(max(float(x), eps), 1.0 - eps)


def _clipped_C(cdf_fn, u, v, eps=1e-12):
    return float(cdf_fn(_clip01(u, eps), _clip01(v, eps)))


def _partial_u_fd(cdf_fn, u, v, h=1e-6):
    return (_clipped_C(cdf_fn, u + h, v) - _clipped_C(cdf_fn, u - h, v)) / (2 * h)


def _partial_v_fd(cdf_fn, u, v, h=1e-6):
    return (_clipped_C(cdf_fn, u, v + h) - _clipped_C(cdf_fn, u, v - h)) / (2 * h)


def _mixed_fd(cdf_fn, u, v, h=1e-5):
    """Central 2-D finite difference ∂²C/∂u∂v."""
    return (
        _clipped_C(cdf_fn, u + h, v + h)
        - _clipped_C(cdf_fn, u + h, v - h)
        - _clipped_C(cdf_fn, u - h, v + h)
        + _clipped_C(cdf_fn, u - h, v - h)
    ) / (4 * h * h)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Parameter validation
# ═══════════════════════════════════════════════════════════════════════════════

@given(theta=valid_theta(), beta=valid_beta())
def test_parameter_roundtrip(theta, beta):
    """set_parameters → get_parameters round-trip."""
    c = TawnT2Copula()
    c.set_parameters([theta, beta])
    p = c.get_parameters()
    assert math.isclose(p[0], theta, rel_tol=1e-12)
    assert math.isclose(p[1], beta, rel_tol=1e-12)


@pytest.mark.parametrize("theta,beta", [
    (0.5, 0.5),     # θ < 1
    (1.0, 0.5),     # θ = 1 (exact boundary, rejected by strict <)
    (501.0, 0.5),   # θ > 500
])
def test_parameter_theta_out_of_bounds(theta, beta):
    """θ ≤ 1 or θ ≥ 500 must be rejected (strict open bounds)."""
    c = TawnT2Copula()
    with pytest.raises(ValueError):
        c.set_parameters([theta, beta])


@pytest.mark.parametrize("theta,beta", [
    (2.0, 0.0),     # β = 0 (exact boundary, rejected by strict <)
    (2.0, -0.1),    # β < 0
    (2.0, 1.0),     # β = 1 (exact boundary, rejected by strict <)
    (2.0, 1.1),     # β > 1
])
def test_parameter_beta_out_of_bounds(theta, beta):
    """β ≤ 0 or β ≥ 1 must be rejected (strict open bounds)."""
    c = TawnT2Copula()
    with pytest.raises(ValueError):
        c.set_parameters([theta, beta])


def test_parameter_wrong_size():
    """Wrong number of parameters must raise ValueError."""
    c = TawnT2Copula()
    with pytest.raises(ValueError):
        c.set_parameters([2.0])
    with pytest.raises(ValueError):
        c.set_parameters([2.0, 0.5, 0.3])
    with pytest.raises(ValueError):
        c.set_parameters([])


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CDF invariants  (no symmetry — asymmetric copula)
# ═══════════════════════════════════════════════════════════════════════════════

@given(theta=valid_theta_stable(), beta=valid_beta_stable(), u=unit, v=unit)
def test_cdf_bounds(theta, beta, u, v):
    """CDF must lie in [0, 1]."""
    c = TawnT2Copula(); c.set_parameters([theta, beta])
    val = float(c.get_cdf(u, v))
    assert 0.0 <= val <= 1.0


@given(theta=valid_theta_stable(), beta=valid_beta_stable(),
       u1=unit, u2=unit, v=unit)
def test_cdf_monotone_in_u(theta, beta, u1, u2, v):
    """C(u1, v) ≤ C(u2, v) when u1 ≤ u2."""
    if u1 > u2:
        u1, u2 = u2, u1
    c = TawnT2Copula(); c.set_parameters([theta, beta])
    assert float(c.get_cdf(u1, v)) <= float(c.get_cdf(u2, v)) + 1e-12


def test_cdf_is_asymmetric():
    """C(u,v) ≠ C(v,u) for generic parameters — confirms asymmetry."""
    c = TawnT2Copula(); c.set_parameters([2.5, 0.4])
    u, v = 0.23, 0.81
    assert not math.isclose(
        float(c.get_cdf(u, v)), float(c.get_cdf(v, u)), abs_tol=1e-4
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Fréchet–Hoeffding boundaries
# ═══════════════════════════════════════════════════════════════════════════════

@given(theta=valid_theta_stable(), beta=valid_beta_stable(), u=unit)
def test_cdf_boundary_u_zero(theta, beta, u):
    """C(u, ε) ≈ 0."""
    c = TawnT2Copula(); c.set_parameters([theta, beta])
    assert math.isclose(float(c.get_cdf(u, 1e-12)), 0.0, abs_tol=1e-6)


@given(theta=valid_theta_stable(), beta=valid_beta_stable(), v=unit)
def test_cdf_boundary_v_zero(theta, beta, v):
    """C(ε, v) ≈ 0."""
    c = TawnT2Copula(); c.set_parameters([theta, beta])
    assert math.isclose(float(c.get_cdf(1e-12, v)), 0.0, abs_tol=1e-6)


@given(theta=valid_theta_stable(), beta=valid_beta_stable(), u=unit)
def test_cdf_boundary_v_one(theta, beta, u):
    """C(u, 1) ≈ u."""
    c = TawnT2Copula(); c.set_parameters([theta, beta])
    assert math.isclose(float(c.get_cdf(u, 1 - 1e-8)), u,
                        rel_tol=1e-2, abs_tol=1e-2)


@given(theta=valid_theta_stable(), beta=valid_beta_stable(), v=unit)
def test_cdf_boundary_u_one(theta, beta, v):
    """C(1, v) ≈ v."""
    c = TawnT2Copula(); c.set_parameters([theta, beta])
    assert math.isclose(float(c.get_cdf(1 - 1e-8, v)), v,
                        rel_tol=1e-2, abs_tol=1e-2)


@given(theta=valid_theta_stable(), beta=valid_beta_stable(), u=unit, v=unit)
@settings(max_examples=50, deadline=None)
def test_frechet_lower_bound(theta, beta, u, v):
    """C(u,v) ≥ max(u+v−1, 0)."""
    c = TawnT2Copula(); c.set_parameters([theta, beta])
    assert float(c.get_cdf(u, v)) >= max(u + v - 1.0, 0.0) - 1e-10


@given(theta=valid_theta_stable(), beta=valid_beta_stable(), u=unit, v=unit)
@settings(max_examples=50, deadline=None)
def test_frechet_upper_bound(theta, beta, u, v):
    """C(u,v) ≤ min(u,v)."""
    c = TawnT2Copula(); c.set_parameters([theta, beta])
    assert float(c.get_cdf(u, v)) <= min(u, v) + 1e-10


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PDF invariants
# ═══════════════════════════════════════════════════════════════════════════════

@given(theta=valid_theta_stable(), beta=valid_beta_stable(), u=unit, v=unit)
def test_pdf_nonnegative(theta, beta, u, v):
    c = TawnT2Copula(); c.set_parameters([theta, beta])
    assert float(c.get_pdf(u, v)) >= -1e-12


@given(theta=valid_theta_stable(), beta=valid_beta_stable(), u=unit, v=unit)
@settings(max_examples=100, deadline=None)
def test_pdf_matches_mixed_derivative(theta, beta, u, v):
    c = TawnT2Copula(); c.set_parameters([theta, beta])
    pdf_ana = float(c.get_pdf(u, v))
    pdf_num = _mixed_fd(c.get_cdf, u, v)
    assert math.isclose(pdf_ana, pdf_num, rel_tol=3e-2, abs_tol=1e-3), \
        f"θ={theta:.3f}, β={beta:.3f}, u={u:.4f}, v={v:.4f}: " \
        f"ana={pdf_ana:.6f}, num={pdf_num:.6f}"


@pytest.mark.parametrize("theta,beta", [
    (1.5, 0.2), (2.0, 0.5), (3.0, 0.7), (5.0, 0.9), (8.0, 0.4),
])
def test_pdf_integrates_to_one(theta, beta):
    c = TawnT2Copula(); c.set_parameters([theta, beta])
    rng = np.random.default_rng(42)
    u, v = rng.random(100_000), rng.random(100_000)
    integral = float(np.asarray(c.get_pdf(u, v)).mean())
    assert math.isclose(integral, 1.0, rel_tol=3e-2)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. h-functions
# ═══════════════════════════════════════════════════════════════════════════════

@given(theta=valid_theta_stable(), beta=valid_beta_stable(), u=unit, v=unit)
def test_h_functions_are_probabilities(theta, beta, u, v):
    c = TawnT2Copula(); c.set_parameters([theta, beta])
    h1 = float(c.partial_derivative_C_wrt_u(u, v))
    h2 = float(c.partial_derivative_C_wrt_v(u, v))
    eps = 1e-6
    assert -eps <= h1 <= 1.0 + eps
    assert -eps <= h2 <= 1.0 + eps


@given(theta=valid_theta_stable(), beta=valid_beta_stable(), u=unit)
@settings(max_examples=50)
def test_h_u_boundary_in_v(theta, beta, u):
    c = TawnT2Copula(); c.set_parameters([theta, beta])
    h_low = float(c.partial_derivative_C_wrt_u(u, 1e-10))
    h_high = float(c.partial_derivative_C_wrt_u(u, 1 - 1e-8))
    assert math.isclose(h_low, 0.0, abs_tol=1e-2)
    assert math.isclose(h_high, 1.0, abs_tol=1e-2)


@given(theta=valid_theta_stable(), beta=valid_beta_stable(), v=unit)
@settings(max_examples=50)
def test_h_v_boundary_in_u(theta, beta, v):
    c = TawnT2Copula(); c.set_parameters([theta, beta])
    h_low = float(c.partial_derivative_C_wrt_v(1e-10, v))
    h_high = float(c.partial_derivative_C_wrt_v(1 - 1e-8, v))
    assert math.isclose(h_low, 0.0, abs_tol=1e-2)
    assert math.isclose(h_high, 1.0, abs_tol=1e-2)


@given(theta=valid_theta_stable(), beta=valid_beta_stable(),
       u=unit, v1=unit, v2=unit)
@settings(max_examples=50)
def test_h_function_monotone_in_v(theta, beta, u, v1, v2):
    if v1 > v2:
        v1, v2 = v2, v1
    c = TawnT2Copula(); c.set_parameters([theta, beta])
    assert float(c.partial_derivative_C_wrt_u(u, v1)) <= \
           float(c.partial_derivative_C_wrt_u(u, v2)) + 1e-9


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Derivatives
# ═══════════════════════════════════════════════════════════════════════════════

@given(theta=valid_theta_stable(), beta=valid_beta_stable(), u=unit, v=unit)
@settings(max_examples=100, deadline=None)
def test_partial_derivative_matches_finite_diff(theta, beta, u, v):
    c = TawnT2Copula(); c.set_parameters([theta, beta])
    ana_du = float(c.partial_derivative_C_wrt_u(u, v))
    ana_dv = float(c.partial_derivative_C_wrt_v(u, v))
    num_du = _partial_u_fd(c.get_cdf, u, v)
    num_dv = _partial_v_fd(c.get_cdf, u, v)
    assert math.isfinite(ana_du) and math.isfinite(ana_dv)
    assert math.isclose(ana_du, num_du, rel_tol=1e-3, abs_tol=1e-4)
    assert math.isclose(ana_dv, num_dv, rel_tol=1e-3, abs_tol=1e-4)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Kendall's tau
# ═══════════════════════════════════════════════════════════════════════════════

@given(theta=valid_theta_stable(), beta=valid_beta_stable())
@settings(max_examples=30, deadline=None)
def test_kendall_tau_range(theta, beta):
    c = TawnT2Copula(); c.set_parameters([theta, beta])
    tau = c.kendall_tau()
    assert 0.0 <= tau < 1.0


@given(theta=valid_theta_stable(), beta=valid_beta_stable())
@settings(max_examples=30, deadline=None)
def test_kendall_tau_positive(theta, beta):
    c = TawnT2Copula(); c.set_parameters([theta, beta])
    assert c.kendall_tau() >= 0.0


def test_kendall_tau_near_zero_at_independence():
    """β → 0 ⇒ τ ≈ 0."""
    c = TawnT2Copula(); c.set_parameters([3.0, _BETA_LO])
    assert math.isclose(c.kendall_tau(), 0.0, abs_tol=1e-6)


def test_kendall_tau_monotone_in_beta():
    theta = 3.0
    betas = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    c = TawnT2Copula()
    taus = []
    for b in betas:
        c.set_parameters([theta, b])
        taus.append(c.kendall_tau())
    for i in range(len(taus) - 1):
        assert taus[i] <= taus[i + 1] + 1e-12


def test_kendall_tau_monotone_in_theta():
    """For β ≈ 1 (near-Gumbel), τ increases with θ."""
    thetas = [1.01, 1.5, 2.0, 3.0, 5.0, 10.0]
    c = TawnT2Copula()
    taus = []
    for th in thetas:
        c.set_parameters([th, _BETA_HI])
        taus.append(c.kendall_tau())
    for i in range(len(taus) - 1):
        assert taus[i] <= taus[i + 1] + 1e-12


def test_kendall_tau_matches_t1_mirror():
    """τ_T2(θ,β) = τ_T1(θ,α) — same Pickands function."""
    from CopulaFurtif.core.copulas.domain.models.archimedean.TawnT1 import TawnT1Copula
    c1 = TawnT1Copula()
    c2 = TawnT2Copula()
    for theta, param2 in [(2.0, 0.5), (3.0, 0.8), (5.0, 0.3)]:
        c1.set_parameters([theta, param2])
        c2.set_parameters([theta, param2])
        assert math.isclose(c1.kendall_tau(), c2.kendall_tau(), rel_tol=1e-10)


@pytest.mark.slow
@pytest.mark.parametrize("theta,beta", [
    (1.5, 0.3), (2.0, 0.5), (3.0, 0.8), (5.0, 0.99),
])
def test_kendall_tau_vs_empirical(theta, beta):
    c = TawnT2Copula(); c.set_parameters([theta, beta])
    data = c.sample(8_000, rng=np.random.default_rng(42))
    tau_emp, _ = sp_kendalltau(data[:, 0], data[:, 1])
    tau_theo = c.kendall_tau()
    n = len(data)
    sigma = math.sqrt(2 * (2 * n + 5) / (9 * n * (n - 1)))
    assert math.isclose(tau_emp, tau_theo, abs_tol=4 * sigma)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Tail dependence
# ═══════════════════════════════════════════════════════════════════════════════

@given(theta=valid_theta_stable(), beta=valid_beta_stable())
def test_ltdc_is_zero(theta, beta):
    c = TawnT2Copula(); c.set_parameters([theta, beta])
    assert c.LTDC() == 0.0


@given(theta=valid_theta_stable(), beta=valid_beta_stable())
@settings(max_examples=30, deadline=None)
def test_utdc_formula(theta, beta):
    c = TawnT2Copula(); c.set_parameters([theta, beta])
    expected = max(2.0 - 2.0 * float(c._A(0.5)), 0.0)
    assert math.isclose(c.UTDC(), expected, rel_tol=1e-12)


def test_utdc_near_zero_at_independence():
    c = TawnT2Copula(); c.set_parameters([3.0, _BETA_LO])
    assert math.isclose(c.UTDC(), 0.0, abs_tol=1e-6)


def test_utdc_positive_for_dependence():
    c = TawnT2Copula(); c.set_parameters([4.0, 0.8])
    assert c.UTDC() > 0.0


def test_utdc_increases_with_beta():
    theta = 4.0
    betas = [0.01, 0.2, 0.5, 0.8, 0.99]
    c = TawnT2Copula()
    utdcs = []
    for b in betas:
        c.set_parameters([theta, b])
        utdcs.append(c.UTDC())
    for i in range(len(utdcs) - 1):
        assert utdcs[i] <= utdcs[i + 1] + 1e-12


def test_utdc_increases_with_theta():
    thetas = [1.01, 1.5, 2.0, 3.0, 5.0, 10.0]
    c = TawnT2Copula()
    utdcs = []
    for th in thetas:
        c.set_parameters([th, 0.8])
        utdcs.append(c.UTDC())
    for i in range(len(utdcs) - 1):
        assert utdcs[i] <= utdcs[i + 1] + 1e-12


def test_utdc_near_gumbel_case():
    """β ≈ 1 ⇒ λ_U ≈ 2 − 2^{1/θ}."""
    c = TawnT2Copula()
    for theta in [1.5, 2.0, 4.0, 8.0]:
        c.set_parameters([theta, _BETA_HI])
        expected = 2.0 - 2.0 ** (1.0 / theta)
        assert math.isclose(c.UTDC(), expected, rel_tol=1e-6)


def test_utdc_empirical_approximation():
    c = TawnT2Copula(); c.set_parameters([4.0, 0.8])
    u = 0.9995
    approx = (1.0 - 2.0 * u + float(c.get_cdf(u, u))) / (1.0 - u)
    assert math.isclose(c.UTDC(), approx, abs_tol=5e-3)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Blomqvist β
# ═══════════════════════════════════════════════════════════════════════════════

@given(theta=valid_theta_stable(), beta=valid_beta_stable())
@settings(max_examples=30, deadline=None)
def test_blomqvist_beta_matches_definition(theta, beta):
    c = TawnT2Copula(); c.set_parameters([theta, beta])
    beta_direct = 4.0 * float(c.get_cdf(0.5, 0.5)) - 1.0
    assert math.isclose(float(c.blomqvist_beta()), beta_direct, rel_tol=1e-10)


def test_blomqvist_beta_near_zero_at_independence():
    c = TawnT2Copula(); c.set_parameters([3.0, _BETA_LO])
    assert math.isclose(c.blomqvist_beta(), 0.0, abs_tol=1e-6)


def test_blomqvist_beta_positive_for_dependence():
    c = TawnT2Copula(); c.set_parameters([3.0, 0.7])
    assert c.blomqvist_beta() > 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Independence case (β → 0)
# ═══════════════════════════════════════════════════════════════════════════════

@given(u=unit, v=unit)
def test_independence_cdf_near_product(u, v):
    c = TawnT2Copula(); c.set_parameters([3.5, _BETA_LO])
    assert math.isclose(float(c.get_cdf(u, v)), u * v,
                        rel_tol=1e-6, abs_tol=1e-6)


@given(u=unit, v=unit)
def test_independence_pdf_near_one(u, v):
    c = TawnT2Copula(); c.set_parameters([3.5, _BETA_LO])
    assert math.isclose(float(c.get_pdf(u, v)), 1.0,
                        rel_tol=1e-4, abs_tol=1e-4)


@given(u=unit, v=unit)
def test_independence_h_functions_identity(u, v):
    c = TawnT2Copula(); c.set_parameters([3.5, _BETA_LO])
    assert math.isclose(float(c.partial_derivative_C_wrt_u(u, v)), v,
                        rel_tol=1e-4, abs_tol=1e-4)
    assert math.isclose(float(c.partial_derivative_C_wrt_v(u, v)), u,
                        rel_tol=1e-4, abs_tol=1e-4)


# ═══════════════════════════════════════════════════════════════════════════════
# 10b. Pickands function invariants
# ═══════════════════════════════════════════════════════════════════════════════

def test_pickands_boundaries():
    c = TawnT2Copula(); c.set_parameters([2.7, 0.35])
    assert math.isclose(float(c._A(0.0)), 1.0, abs_tol=1e-12)
    assert math.isclose(float(c._A(1.0)), 1.0, abs_tol=1e-12)


@pytest.mark.parametrize("theta,beta", [
    (1.5, 0.2), (2.7, 0.35), (5.0, 0.9),
])
def test_pickands_convexity(theta, beta):
    c = TawnT2Copula(); c.set_parameters([theta, beta])
    grid = np.linspace(1e-4, 1 - 1e-4, 200)
    App = c._A_double(grid)
    assert np.all(App >= -1e-8)


@pytest.mark.parametrize("theta,beta", [(2.0, 0.5), (4.0, 0.8)])
def test_pickands_lower_bound(theta, beta):
    c = TawnT2Copula(); c.set_parameters([theta, beta])
    grid = np.linspace(0.0, 1.0, 500)
    A = c._A(grid)
    lower = np.maximum(grid, 1.0 - grid)
    assert np.all(A >= lower - 1e-10)
    assert np.all(A <= 1.0 + 1e-10)


def test_cdf_comonotone_limit():
    """β ≈ 1, θ large ⇒ C(u,v) → min(u,v) for off-diagonal points."""
    c = TawnT2Copula(); c.set_parameters([200.0, _BETA_HI])
    for u, v in [(0.3, 0.7), (0.8, 0.2)]:
        assert math.isclose(float(c.get_cdf(u, v)), min(u, v), abs_tol=5e-2)


def test_t2_is_t1_mirror():
    """C_T2(u,v; θ,β) = C_T1(v,u; θ,α)."""
    from CopulaFurtif.core.copulas.domain.models.archimedean.TawnT1 import TawnT1Copula
    c1 = TawnT1Copula()
    c2 = TawnT2Copula()
    for theta, param2 in [(2.0, 0.5), (3.0, 0.8)]:
        c1.set_parameters([theta, param2])
        c2.set_parameters([theta, param2])
        for u, v in [(0.2, 0.7), (0.6, 0.3), (0.5, 0.5)]:
            val_t2 = float(c2.get_cdf(u, v))
            val_t1_swapped = float(c1.get_cdf(v, u))
            assert math.isclose(val_t2, val_t1_swapped, rel_tol=1e-10)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. init_from_data  (@slow)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
@pytest.mark.parametrize("theta,beta", [
    (1.5, 0.3), (2.4, 0.55), (3.0, 0.8), (5.0, 0.99),
])
def test_init_from_data_roundtrip(theta, beta):
    c = TawnT2Copula(); c.set_parameters([theta, beta])
    data = c.sample(1500, rng=np.random.default_rng(321))
    init_param = c.init_from_data(data[:, 0], data[:, 1])
    assert init_param.shape == (2,)
    assert init_param[0] > 1.0
    assert 0.0 < init_param[1] < 1.0
    tau_emp, _ = sp_kendalltau(data[:, 0], data[:, 1])
    beta_emp = 2.0 * np.mean((data[:, 0] > 0.5) == (data[:, 1] > 0.5)) - 1.0
    c2 = TawnT2Copula(); c2.set_parameters(init_param)
    assert abs(c2.kendall_tau() - tau_emp) < 0.12
    assert abs(c2.blomqvist_beta() - beta_emp) < 0.12


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Sampling  (@slow)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
@pytest.mark.parametrize("theta,beta", [(2.0, 0.5), (2.7, 0.65), (4.0, 0.9)])
def test_sampling_shape_and_bounds(theta, beta):
    c = TawnT2Copula(); c.set_parameters([theta, beta])
    data = c.sample(1200, rng=np.random.default_rng(123))
    assert data.shape == (1200, 2)
    assert np.all(data > 0.0) and np.all(data < 1.0)


@pytest.mark.slow
@pytest.mark.parametrize("theta,beta", [(2.0, 0.5), (3.0, 0.8), (5.0, 0.99)])
def test_empirical_kendall_tau_close(theta, beta):
    c = TawnT2Copula(); c.set_parameters([theta, beta])
    data = c.sample(5000, rng=np.random.default_rng(0))
    tau_emp = float(sp_kendalltau(data[:, 0], data[:, 1]).correlation)
    tau_theo = c.kendall_tau()
    assert abs(tau_emp - tau_theo) < 0.08


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Vectorisation
# ═══════════════════════════════════════════════════════════════════════════════

def test_vectorised_shapes(copula_default):
    u = np.linspace(0.05, 0.95, 15)
    v = np.linspace(0.05, 0.95, 15)
    assert copula_default.get_cdf(u, v).shape == (15,)
    assert copula_default.get_pdf(u, v).shape == (15,)
    assert copula_default.partial_derivative_C_wrt_u(u, v).shape == (15,)
    assert copula_default.partial_derivative_C_wrt_v(u, v).shape == (15,)
    samples = copula_default.sample(256, rng=np.random.default_rng(99))
    assert samples.shape == (256, 2)


def test_vectorised_inputs_are_pairwise_not_grid(copula_default):
    u = np.array([0.2, 0.8])
    v = np.array([0.3, 0.7])
    cdf_vec = copula_default.get_cdf(u, v)
    cdf_0 = float(copula_default.get_cdf(u[0], v[0]))
    cdf_1 = float(copula_default.get_cdf(u[1], v[1]))
    assert cdf_vec.shape == (2,)
    assert np.allclose(cdf_vec, [cdf_0, cdf_1])


def test_scalar_and_array_agree(copula_default):
    for u0, v0 in [(0.2, 0.7), (0.5, 0.5), (0.8, 0.3)]:
        scalar = float(copula_default.get_cdf(u0, v0))
        array = float(copula_default.get_cdf(np.array([u0]), np.array([v0]))[0])
        assert math.isclose(scalar, array, rel_tol=1e-12)


# ═══════════════════════════════════════════════════════════════════════════════
# 14. IAD / AD disabled
# ═══════════════════════════════════════════════════════════════════════════════

def test_iad_returns_nan():
    c = TawnT2Copula(); c.set_parameters([2.0, 0.5])
    assert np.isnan(c.IAD(None))


def test_ad_returns_nan():
    c = TawnT2Copula(); c.set_parameters([2.0, 0.5])
    assert np.isnan(c.AD(None))