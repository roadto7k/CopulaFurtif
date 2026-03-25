"""
Comprehensive unit-test suite for the bivariate **Tawn Type-1** copula.

Tawn Type-1 is an asymmetric extreme-value copula (Joe 2014, §4.8.1;
Tadi & Witzany 2025, Table 2) with Pickands dependence function
(using t = x/s, x = −ln u, y = −ln v, s = x+y):

    A(t) = (1 − α)(1 − t) + [t^θ + (α(1 − t))^θ]^{1/θ}

Parameters:  θ ∈ (1, 500),  α ∈ (0, 1)   — strict open bounds.

Properties:
  • **Asymmetric**: C(u,v) ≠ C(v,u) for generic α ∈ (0, 1).
  • α → 0  ⇒  independence copula  C⊥.
  • α → 1  ⇒  Gumbel copula with parameter θ.
  • θ → 1  ⇒  independence copula (for any α).
  • λ_L = 0 always;  λ_U = 2 − 2A(½) > 0 when α > 0 and θ > 1.
  • τ ∈ [0, 1), computed by Gauss–Legendre quadrature.
  • C_T2(u,v; θ,α) = C_T1(v,u; θ,α)  (mirror of Tawn Type-1).

Run with:
    pytest -q                  # fast tests
    pytest -q -m slow          # include heavy sampling / init checks

IMPORTANT: The real CopulaParameters._validate uses STRICT inequalities
(lo < val < hi), so boundary values 0.0, 1.0 for α and 1.0 for θ are
rejected.  Tests use ε-offset values (e.g. 1e-8, 1-1e-8) for limit cases.
"""

import math

import numpy as np
import pytest
from hypothesis import given, settings, note, HealthCheck
from hypothesis import strategies as st
from scipy.stats import kendalltau as sp_kendalltau

from CopulaFurtif.core.copulas.domain.models.archimedean.TawnT1 import TawnT1Copula
from CopulaFurtif.core.copulas.domain.models.archimedean.TawnT2 import TawnT2Copula as TawnT2Copula_


# ── Near-boundary constants (strict open bounds) ─────────────────────────────
_ALPHA_LO = 1e-8          # α → 0 (independence limit)
_ALPHA_HI = 1.0 - 1e-8    # α → 1 (Gumbel limit)
_THETA_LO = 1.0 + 1e-8   # θ → 1 (independence limit)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures & Hypothesis strategies
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def copula_default():
    """Tawn-T1 copula with θ = 2.0, α = 0.5 for deterministic tests."""
    c = TawnT1Copula()
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
def valid_alpha(draw):
    """α ∈ (1e-6, 1-1e-6) — strictly inside (0, 1)."""
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
def valid_alpha_stable(draw):
    """α ∈ (0.05, 0.95) — away from degenerate boundaries."""
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

@given(theta=valid_theta(), alpha=valid_alpha())
def test_parameter_roundtrip(theta, alpha):
    """set_parameters → get_parameters round-trip."""
    c = TawnT1Copula()
    c.set_parameters([theta, alpha])
    p = c.get_parameters()
    assert math.isclose(p[0], theta, rel_tol=1e-12)
    assert math.isclose(p[1], alpha, rel_tol=1e-12)


@pytest.mark.parametrize("theta,alpha", [
    (0.5, 0.5),     # θ < 1
    (1.0, 0.5),     # θ = 1 (exact boundary, rejected by strict <)
    (501.0, 0.5),   # θ > 500
])
def test_parameter_theta_out_of_bounds(theta, alpha):
    """θ ≤ 1 or θ ≥ 500 must be rejected (strict open bounds)."""
    c = TawnT1Copula()
    with pytest.raises(ValueError):
        c.set_parameters([theta, alpha])


@pytest.mark.parametrize("theta,alpha", [
    (2.0, 0.0),     # α = 0 (exact boundary, rejected by strict <)
    (2.0, -0.1),    # α < 0
    (2.0, 1.0),     # α = 1 (exact boundary, rejected by strict <)
    (2.0, 1.1),     # α > 1
])
def test_parameter_alpha_out_of_bounds(theta, alpha):
    """α ≤ 0 or α ≥ 1 must be rejected (strict open bounds)."""
    c = TawnT1Copula()
    with pytest.raises(ValueError):
        c.set_parameters([theta, alpha])


def test_parameter_wrong_size():
    """Wrong number of parameters must raise ValueError."""
    c = TawnT1Copula()
    with pytest.raises(ValueError):
        c.set_parameters([2.0])
    with pytest.raises(ValueError):
        c.set_parameters([2.0, 0.5, 0.3])
    with pytest.raises(ValueError):
        c.set_parameters([])


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CDF invariants  (no symmetry — asymmetric copula)
# ═══════════════════════════════════════════════════════════════════════════════

@given(theta=valid_theta_stable(), alpha=valid_alpha_stable(), u=unit, v=unit)
def test_cdf_bounds(theta, alpha, u, v):
    """CDF must lie in [0, 1]."""
    c = TawnT1Copula(); c.set_parameters([theta, alpha])
    val = float(c.get_cdf(u, v))
    assert 0.0 <= val <= 1.0


@given(theta=valid_theta_stable(), alpha=valid_alpha_stable(),
       u1=unit, u2=unit, v=unit)
def test_cdf_monotone_in_u(theta, alpha, u1, u2, v):
    """C(u1, v) ≤ C(u2, v) when u1 ≤ u2."""
    if u1 > u2:
        u1, u2 = u2, u1
    c = TawnT1Copula(); c.set_parameters([theta, alpha])
    assert float(c.get_cdf(u1, v)) <= float(c.get_cdf(u2, v)) + 1e-12


def test_cdf_is_asymmetric():
    """C(u,v) ≠ C(v,u) for generic parameters — confirms asymmetry."""
    c = TawnT1Copula(); c.set_parameters([2.5, 0.4])
    u, v = 0.23, 0.81
    assert not math.isclose(
        float(c.get_cdf(u, v)), float(c.get_cdf(v, u)), abs_tol=1e-4
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Fréchet–Hoeffding boundaries
# ═══════════════════════════════════════════════════════════════════════════════

@given(theta=valid_theta_stable(), alpha=valid_alpha_stable(), u=unit)
def test_cdf_boundary_u_zero(theta, alpha, u):
    """C(u, ε) ≈ 0."""
    c = TawnT1Copula(); c.set_parameters([theta, alpha])
    assert math.isclose(float(c.get_cdf(u, 1e-12)), 0.0, abs_tol=1e-6)


@given(theta=valid_theta_stable(), alpha=valid_alpha_stable(), v=unit)
def test_cdf_boundary_v_zero(theta, alpha, v):
    """C(ε, v) ≈ 0."""
    c = TawnT1Copula(); c.set_parameters([theta, alpha])
    assert math.isclose(float(c.get_cdf(1e-12, v)), 0.0, abs_tol=1e-6)


@given(theta=valid_theta_stable(), alpha=valid_alpha_stable(), u=unit)
def test_cdf_boundary_v_one(theta, alpha, u):
    """C(u, 1) ≈ u."""
    c = TawnT1Copula(); c.set_parameters([theta, alpha])
    assert math.isclose(float(c.get_cdf(u, 1 - 1e-8)), u,
                        rel_tol=1e-2, abs_tol=1e-2)


@given(theta=valid_theta_stable(), alpha=valid_alpha_stable(), v=unit)
def test_cdf_boundary_u_one(theta, alpha, v):
    """C(1, v) ≈ v."""
    c = TawnT1Copula(); c.set_parameters([theta, alpha])
    assert math.isclose(float(c.get_cdf(1 - 1e-8, v)), v,
                        rel_tol=1e-2, abs_tol=1e-2)


@given(theta=valid_theta_stable(), alpha=valid_alpha_stable(), u=unit, v=unit)
@settings(max_examples=50, deadline=None)
def test_frechet_lower_bound(theta, alpha, u, v):
    """C(u,v) ≥ max(u+v−1, 0)."""
    c = TawnT1Copula(); c.set_parameters([theta, alpha])
    assert float(c.get_cdf(u, v)) >= max(u + v - 1.0, 0.0) - 1e-10


@given(theta=valid_theta_stable(), alpha=valid_alpha_stable(), u=unit, v=unit)
@settings(max_examples=50, deadline=None)
def test_frechet_upper_bound(theta, alpha, u, v):
    """C(u,v) ≤ min(u,v)."""
    c = TawnT1Copula(); c.set_parameters([theta, alpha])
    assert float(c.get_cdf(u, v)) <= min(u, v) + 1e-10


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PDF invariants
# ═══════════════════════════════════════════════════════════════════════════════

@given(theta=valid_theta_stable(), alpha=valid_alpha_stable(), u=unit, v=unit)
def test_pdf_nonnegative(theta, alpha, u, v):
    c = TawnT1Copula(); c.set_parameters([theta, alpha])
    assert float(c.get_pdf(u, v)) >= -1e-12


@given(theta=valid_theta_stable(), alpha=valid_alpha_stable(), u=unit, v=unit)
@settings(max_examples=100, deadline=None)
def test_pdf_matches_mixed_derivative(theta, alpha, u, v):
    c = TawnT1Copula(); c.set_parameters([theta, alpha])
    pdf_ana = float(c.get_pdf(u, v))
    pdf_num = _mixed_fd(c.get_cdf, u, v)
    assert math.isclose(pdf_ana, pdf_num, rel_tol=3e-2, abs_tol=1e-3), \
        f"θ={theta:.3f}, α={alpha:.3f}, u={u:.4f}, v={v:.4f}: " \
        f"ana={pdf_ana:.6f}, num={pdf_num:.6f}"


@pytest.mark.parametrize("theta,alpha", [
    (1.5, 0.2), (2.0, 0.5), (3.0, 0.7), (5.0, 0.9), (8.0, 0.4),
])
def test_pdf_integrates_to_one(theta, alpha):
    c = TawnT1Copula(); c.set_parameters([theta, alpha])
    rng = np.random.default_rng(42)
    u, v = rng.random(100_000), rng.random(100_000)
    integral = float(np.asarray(c.get_pdf(u, v)).mean())
    assert math.isclose(integral, 1.0, rel_tol=3e-2)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. h-functions
# ═══════════════════════════════════════════════════════════════════════════════

@given(theta=valid_theta_stable(), alpha=valid_alpha_stable(), u=unit, v=unit)
def test_h_functions_are_probabilities(theta, alpha, u, v):
    c = TawnT1Copula(); c.set_parameters([theta, alpha])
    h1 = float(c.partial_derivative_C_wrt_u(u, v))
    h2 = float(c.partial_derivative_C_wrt_v(u, v))
    eps = 1e-6
    assert -eps <= h1 <= 1.0 + eps
    assert -eps <= h2 <= 1.0 + eps


@given(theta=valid_theta_stable(), alpha=valid_alpha_stable(), u=unit)
@settings(max_examples=50)
def test_h_u_boundary_in_v(theta, alpha, u):
    c = TawnT1Copula(); c.set_parameters([theta, alpha])
    h_low = float(c.partial_derivative_C_wrt_u(u, 1e-10))
    h_high = float(c.partial_derivative_C_wrt_u(u, 1 - 1e-8))
    assert math.isclose(h_low, 0.0, abs_tol=1e-2)
    assert math.isclose(h_high, 1.0, abs_tol=1e-2)


@given(theta=valid_theta_stable(), alpha=valid_alpha_stable(), v=unit)
@settings(max_examples=50)
def test_h_v_boundary_in_u(theta, alpha, v):
    c = TawnT1Copula(); c.set_parameters([theta, alpha])
    h_low = float(c.partial_derivative_C_wrt_v(1e-10, v))
    h_high = float(c.partial_derivative_C_wrt_v(1 - 1e-8, v))
    assert math.isclose(h_low, 0.0, abs_tol=1e-2)
    assert math.isclose(h_high, 1.0, abs_tol=1e-2)


@given(theta=valid_theta_stable(), alpha=valid_alpha_stable(),
       u=unit, v1=unit, v2=unit)
@settings(max_examples=50)
def test_h_function_monotone_in_v(theta, alpha, u, v1, v2):
    if v1 > v2:
        v1, v2 = v2, v1
    c = TawnT1Copula(); c.set_parameters([theta, alpha])
    assert float(c.partial_derivative_C_wrt_u(u, v1)) <= \
           float(c.partial_derivative_C_wrt_u(u, v2)) + 1e-9


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Derivatives
# ═══════════════════════════════════════════════════════════════════════════════

@given(theta=valid_theta_stable(), alpha=valid_alpha_stable(), u=unit, v=unit)
@settings(max_examples=100, deadline=None)
def test_partial_derivative_matches_finite_diff(theta, alpha, u, v):
    c = TawnT1Copula(); c.set_parameters([theta, alpha])
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

@given(theta=valid_theta_stable(), alpha=valid_alpha_stable())
@settings(max_examples=30, deadline=None)
def test_kendall_tau_range(theta, alpha):
    c = TawnT1Copula(); c.set_parameters([theta, alpha])
    tau = c.kendall_tau()
    assert 0.0 <= tau < 1.0


@given(theta=valid_theta_stable(), alpha=valid_alpha_stable())
@settings(max_examples=30, deadline=None)
def test_kendall_tau_positive(theta, alpha):
    c = TawnT1Copula(); c.set_parameters([theta, alpha])
    assert c.kendall_tau() >= 0.0


def test_kendall_tau_near_zero_at_independence():
    """α → 0 ⇒ τ ≈ 0."""
    c = TawnT1Copula(); c.set_parameters([3.0, _ALPHA_LO])
    assert math.isclose(c.kendall_tau(), 0.0, abs_tol=1e-6)


def test_kendall_tau_monotone_in_alpha():
    theta = 3.0
    alphas = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    c = TawnT1Copula()
    taus = []
    for b in alphas:
        c.set_parameters([theta, b])
        taus.append(c.kendall_tau())
    for i in range(len(taus) - 1):
        assert taus[i] <= taus[i + 1] + 1e-12


def test_kendall_tau_monotone_in_theta():
    """For α ≈ 1 (near-Gumbel), τ increases with θ."""
    thetas = [1.01, 1.5, 2.0, 3.0, 5.0, 10.0]
    c = TawnT1Copula()
    taus = []
    for th in thetas:
        c.set_parameters([th, _ALPHA_HI])
        taus.append(c.kendall_tau())
    for i in range(len(taus) - 1):
        assert taus[i] <= taus[i + 1] + 1e-12


def test_kendall_tau_matches_t1_mirror():
    """τ_T2(θ,α) = τ_T1(θ,α) — same Pickands function."""
    from CopulaFurtif.core.copulas.domain.models.archimedean.TawnT2 import TawnT2Copula as TawnT2Copula_
    c1 = TawnT1Copula()
    c2 = TawnT2Copula_()
    for theta, param2 in [(2.0, 0.5), (3.0, 0.8), (5.0, 0.3)]:
        c1.set_parameters([theta, param2])
        c2.set_parameters([theta, param2])
        assert math.isclose(c1.kendall_tau(), c2.kendall_tau(), rel_tol=1e-10)


@pytest.mark.slow
@pytest.mark.parametrize("theta,alpha", [
    (1.5, 0.3), (2.0, 0.5), (3.0, 0.8), (5.0, 0.99),
])
def test_kendall_tau_vs_empirical(theta, alpha):
    c = TawnT1Copula(); c.set_parameters([theta, alpha])
    data = c.sample(8_000, rng=np.random.default_rng(42))
    tau_emp, _ = sp_kendalltau(data[:, 0], data[:, 1])
    tau_theo = c.kendall_tau()
    n = len(data)
    sigma = math.sqrt(2 * (2 * n + 5) / (9 * n * (n - 1)))
    assert math.isclose(tau_emp, tau_theo, abs_tol=4 * sigma)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Tail dependence
# ═══════════════════════════════════════════════════════════════════════════════

@given(theta=valid_theta_stable(), alpha=valid_alpha_stable())
def test_ltdc_is_zero(theta, alpha):
    c = TawnT1Copula(); c.set_parameters([theta, alpha])
    assert c.LTDC() == 0.0


@given(theta=valid_theta_stable(), alpha=valid_alpha_stable())
@settings(max_examples=30, deadline=None)
def test_utdc_formula(theta, alpha):
    c = TawnT1Copula(); c.set_parameters([theta, alpha])
    expected = max(2.0 - 2.0 * float(c._A(0.5)), 0.0)
    assert math.isclose(c.UTDC(), expected, rel_tol=1e-12)


def test_utdc_near_zero_at_independence():
    c = TawnT1Copula(); c.set_parameters([3.0, _ALPHA_LO])
    assert math.isclose(c.UTDC(), 0.0, abs_tol=1e-6)


def test_utdc_positive_for_dependence():
    c = TawnT1Copula(); c.set_parameters([4.0, 0.8])
    assert c.UTDC() > 0.0


def test_utdc_increases_with_alpha():
    theta = 4.0
    alphas = [0.01, 0.2, 0.5, 0.8, 0.99]
    c = TawnT1Copula()
    utdcs = []
    for b in alphas:
        c.set_parameters([theta, b])
        utdcs.append(c.UTDC())
    for i in range(len(utdcs) - 1):
        assert utdcs[i] <= utdcs[i + 1] + 1e-12


def test_utdc_increases_with_theta():
    thetas = [1.01, 1.5, 2.0, 3.0, 5.0, 10.0]
    c = TawnT1Copula()
    utdcs = []
    for th in thetas:
        c.set_parameters([th, 0.8])
        utdcs.append(c.UTDC())
    for i in range(len(utdcs) - 1):
        assert utdcs[i] <= utdcs[i + 1] + 1e-12


def test_utdc_near_gumbel_case():
    """α ≈ 1 ⇒ λ_U ≈ 2 − 2^{1/θ}."""
    c = TawnT1Copula()
    for theta in [1.5, 2.0, 4.0, 8.0]:
        c.set_parameters([theta, _ALPHA_HI])
        expected = 2.0 - 2.0 ** (1.0 / theta)
        assert math.isclose(c.UTDC(), expected, rel_tol=1e-6)


def test_utdc_empirical_approximation():
    c = TawnT1Copula(); c.set_parameters([4.0, 0.8])
    u = 0.9995
    approx = (1.0 - 2.0 * u + float(c.get_cdf(u, u))) / (1.0 - u)
    assert math.isclose(c.UTDC(), approx, abs_tol=5e-3)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Blomqvist α
# ═══════════════════════════════════════════════════════════════════════════════

@given(theta=valid_theta_stable(), alpha=valid_alpha_stable())
@settings(max_examples=30, deadline=None)
def test_blomqvist_beta_matches_definition(theta, alpha):
    c = TawnT1Copula(); c.set_parameters([theta, alpha])
    beta_direct = 4.0 * float(c.get_cdf(0.5, 0.5)) - 1.0
    assert math.isclose(float(c.blomqvist_beta()), beta_direct, rel_tol=1e-10)


def test_blomqvist_beta_near_zero_at_independence():
    c = TawnT1Copula(); c.set_parameters([3.0, _ALPHA_LO])
    assert math.isclose(c.blomqvist_beta(), 0.0, abs_tol=1e-6)


def test_blomqvist_beta_positive_for_dependence():
    c = TawnT1Copula(); c.set_parameters([3.0, 0.7])
    assert c.blomqvist_beta() > 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Independence case (α → 0)
# ═══════════════════════════════════════════════════════════════════════════════

@given(u=unit, v=unit)
def test_independence_cdf_near_product(u, v):
    c = TawnT1Copula(); c.set_parameters([3.5, _ALPHA_LO])
    assert math.isclose(float(c.get_cdf(u, v)), u * v,
                        rel_tol=1e-6, abs_tol=1e-6)


@given(u=unit, v=unit)
def test_independence_pdf_near_one(u, v):
    c = TawnT1Copula(); c.set_parameters([3.5, _ALPHA_LO])
    assert math.isclose(float(c.get_pdf(u, v)), 1.0,
                        rel_tol=1e-4, abs_tol=1e-4)


@given(u=unit, v=unit)
def test_independence_h_functions_identity(u, v):
    c = TawnT1Copula(); c.set_parameters([3.5, _ALPHA_LO])
    assert math.isclose(float(c.partial_derivative_C_wrt_u(u, v)), v,
                        rel_tol=1e-4, abs_tol=1e-4)
    assert math.isclose(float(c.partial_derivative_C_wrt_v(u, v)), u,
                        rel_tol=1e-4, abs_tol=1e-4)


# ═══════════════════════════════════════════════════════════════════════════════
# 10b. Pickands function invariants
# ═══════════════════════════════════════════════════════════════════════════════

def test_pickands_boundaries():
    c = TawnT1Copula(); c.set_parameters([2.7, 0.35])
    assert math.isclose(float(c._A(0.0)), 1.0, abs_tol=1e-12)
    assert math.isclose(float(c._A(1.0)), 1.0, abs_tol=1e-12)


@pytest.mark.parametrize("theta,alpha", [
    (1.5, 0.2), (2.7, 0.35), (5.0, 0.9),
])
def test_pickands_convexity(theta, alpha):
    c = TawnT1Copula(); c.set_parameters([theta, alpha])
    grid = np.linspace(1e-4, 1 - 1e-4, 200)
    App = c._A_double(grid)
    assert np.all(App >= -1e-8)


@pytest.mark.parametrize("theta,alpha", [(2.0, 0.5), (4.0, 0.8)])
def test_pickands_lower_bound(theta, alpha):
    c = TawnT1Copula(); c.set_parameters([theta, alpha])
    grid = np.linspace(0.0, 1.0, 500)
    A = c._A(grid)
    lower = np.maximum(grid, 1.0 - grid)
    assert np.all(A >= lower - 1e-10)
    assert np.all(A <= 1.0 + 1e-10)


def test_cdf_comonotone_limit():
    """α ≈ 1, θ large ⇒ C(u,v) → min(u,v) for off-diagonal points."""
    c = TawnT1Copula(); c.set_parameters([200.0, _ALPHA_HI])
    for u, v in [(0.3, 0.7), (0.8, 0.2)]:
        assert math.isclose(float(c.get_cdf(u, v)), min(u, v), abs_tol=5e-2)


def test_t2_is_t1_mirror():
    """C_T2(u,v; θ,α) = C_T1(v,u; θ,α)."""
    from CopulaFurtif.core.copulas.domain.models.archimedean.TawnT2 import TawnT2Copula as TawnT2Copula_
    c1 = TawnT1Copula()
    c2 = TawnT2Copula_()
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
@pytest.mark.parametrize("theta,alpha", [
    (1.5, 0.3), (2.4, 0.55), (3.0, 0.8), (5.0, 0.99),
])
def test_init_from_data_roundtrip(theta, alpha):
    c = TawnT1Copula(); c.set_parameters([theta, alpha])
    data = c.sample(1500, rng=np.random.default_rng(321))
    init_param = c.init_from_data(data[:, 0], data[:, 1])
    assert init_param.shape == (2,)
    assert init_param[0] > 1.0
    assert 0.0 < init_param[1] < 1.0
    tau_emp, _ = sp_kendalltau(data[:, 0], data[:, 1])
    beta_emp = 2.0 * np.mean((data[:, 0] > 0.5) == (data[:, 1] > 0.5)) - 1.0
    c2 = TawnT2Copula_(); c2.set_parameters(init_param)
    assert abs(c2.kendall_tau() - tau_emp) < 0.12
    assert abs(c2.blomqvist_beta() - beta_emp) < 0.12


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Sampling  (@slow)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
@pytest.mark.parametrize("theta,alpha", [(2.0, 0.5), (2.7, 0.65), (4.0, 0.9)])
def test_sampling_shape_and_bounds(theta, alpha):
    c = TawnT1Copula(); c.set_parameters([theta, alpha])
    data = c.sample(1200, rng=np.random.default_rng(123))
    assert data.shape == (1200, 2)
    assert np.all(data > 0.0) and np.all(data < 1.0)


@pytest.mark.slow
@pytest.mark.parametrize("theta,alpha", [(2.0, 0.5), (3.0, 0.8), (5.0, 0.99)])
def test_empirical_kendall_tau_close(theta, alpha):
    c = TawnT1Copula(); c.set_parameters([theta, alpha])
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
    c = TawnT1Copula(); c.set_parameters([2.0, 0.5])
    assert np.isnan(c.IAD(None))


def test_ad_returns_nan():
    c = TawnT1Copula(); c.set_parameters([2.0, 0.5])
    assert np.isnan(c.AD(None))