"""Unit-test suite for the Husler-Reiss Extreme-Value Copula.

Run with:  pytest -q  (add -m 'not slow' on CI to skip heavy tests)

Dependencies (add to requirements-dev.txt):
    pytest
    hypothesis
    scipy

The tests focus on:
    * Parameter validation (inside/outside the admissible interval).
    * Core invariants: symmetry, monotonicity, bounds of CDF/PDF.
    * Fréchet–Hoeffding boundary conditions.
    * PDF integrates to 1 (Monte-Carlo, multiple δ values).
    * Analytical vs numerical partial derivatives (spot-check).
    * PDF matches mixed second-order finite difference of the CDF.
    * h-functions (conditional CDFs): probability range, boundaries, symmetry, monotonicity.
    * Upper tail dependence: λ_U = 2·(1 − Φ(1/δ)) > 0; λ_L = 0 (EV copula).
    * Blomqvist's beta: β = 4·C(0.5, 0.5) - 1.
    * Kendall's tau: range, monotonicity, empirical check.
    * Limit cases: δ → 0+ → independence, δ → ∞ → comonotonicity.
    * init_from_data round-trip.

Husler-Reiss copula properties:
    - δ ∈ (0, 50) in this implementation (strict).
    - Extreme-value copula: upper tail dependence only (λ_L = 0).
    - λ_U = 2·(1 - Φ(1/δ)), where Φ is the standard normal CDF.
    - As δ → 0+, approaches independence.
    - As δ → ∞, approaches comonotonicity.

Slow / stochastic tests are marked with @pytest.mark.slow so they can be
optionally skipped (-m "not slow").
"""

import math
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, assume
from scipy.stats import norm, kendalltau

from CopulaFurtif.core.copulas.domain.models.exotic.husle_reiss import HuslerReissCopula


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

@st.composite
def valid_delta(draw):
    """δ strictly inside (0.02, 30) — avoids extremes that cause numerical issues."""
    return draw(st.floats(
        min_value=0.02, max_value=30.0,
        exclude_min=True, exclude_max=True,
        allow_nan=False, allow_infinity=False,
        allow_subnormal=False,
        width=64,
    ))


@st.composite
def unit_interval(draw, eps=1e-3):
    """Floats strictly inside (eps, 1-eps)."""
    return draw(st.floats(
        min_value=eps, max_value=1.0 - eps,
        exclude_min=True, exclude_max=True,
        allow_nan=False, allow_infinity=False,
        allow_subnormal=False,
        width=64,
    ))


def _clip01(x, eps=1e-12):
    return min(max(float(x), eps), 1.0 - eps)


def _clipped_f(f, x, y, eps=1e-12):
    return f(_clip01(x, eps), _clip01(y, eps))


def _finite_diff(f, x, y, h=1e-5, eps=1e-12):
    """Central FD ∂f/∂x with clipping to (0, 1)."""
    return (_clipped_f(f, x + h, y, eps) - _clipped_f(f, x - h, y, eps)) / (2.0 * h)


def _mixed_finite_diff(C, u, v, h=1e-5, eps=1e-12):
    """Central FD ∂²C/∂u∂v with clipping to (0, 1)."""
    return (
        _clipped_f(C, u + h, v + h, eps)
        - _clipped_f(C, u + h, v - h, eps)
        - _clipped_f(C, u - h, v + h, eps)
        + _clipped_f(C, u - h, v - h, eps)
    ) / (4.0 * h * h)


def _make(delta: float) -> HuslerReissCopula:
    c = HuslerReissCopula()
    c.set_parameters([float(delta)])
    return c


# ---------------------------------------------------------------------------
# Parameter tests
# ---------------------------------------------------------------------------

@given(delta=valid_delta())
def test_parameter_roundtrip(delta):
    """set_parameters then get_parameters should return the same value."""
    c = HuslerReissCopula()
    c.set_parameters([delta])
    assert math.isclose(float(c.get_parameters()[0]), float(delta), rel_tol=0, abs_tol=0)


@pytest.mark.parametrize("bad", [0.0, -1.0, 50.0, 100.0])
def test_parameter_out_of_bounds(bad):
    """Values at or outside bounds must be rejected."""
    c = HuslerReissCopula()
    with pytest.raises(ValueError):
        c.set_parameters([bad])


@pytest.mark.parametrize("bad", [0.0, -0.5])
def test_parameter_at_lower_boundary_rejected(bad):
    """δ ≤ 0 must be rejected (Husler-Reiss requires δ > 0 strict)."""
    c = HuslerReissCopula()
    with pytest.raises(ValueError):
        c.set_parameters([bad])


def test_parameter_wrong_size():
    """Passing wrong number of parameters must raise ValueError."""
    c = HuslerReissCopula()
    with pytest.raises(ValueError):
        c.set_parameters([])
    with pytest.raises(ValueError):
        c.set_parameters([1.0, 2.0])


# ---------------------------------------------------------------------------
# CDF invariants
# ---------------------------------------------------------------------------

@given(delta=valid_delta(), u=unit_interval(), v=unit_interval())
def test_cdf_bounds(delta, u, v):
    """CDF must lie in [0, 1]."""
    c = _make(delta)
    assert 0.0 <= float(c.get_cdf(u, v)) <= 1.0


@given(delta=valid_delta(), u=unit_interval(), v=unit_interval())
def test_cdf_monotone_in_u(delta, u, v):
    """C(u, v) ≤ C(u2, v) for u2 > u."""
    c = _make(delta)
    u2 = min(u + 1e-3, 0.999)
    assert c.get_cdf(u, v) <= c.get_cdf(u2, v) + 1e-12


@given(delta=valid_delta(), u=unit_interval(), v=unit_interval())
def test_cdf_symmetry(delta, u, v):
    """Husler-Reiss copula is symmetric: C(u, v) = C(v, u)."""
    c = _make(delta)
    assert math.isclose(float(c.get_cdf(u, v)), float(c.get_cdf(v, u)), rel_tol=1e-12, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# Fréchet–Hoeffding boundary conditions
# ---------------------------------------------------------------------------

@given(delta=valid_delta(), u=unit_interval())
def test_cdf_boundary_u_zero(delta, u):
    """C(u, 0) = 0 for any copula."""
    c = _make(delta)
    assert math.isclose(float(c.get_cdf(u, 0.0)), 0.0, abs_tol=2e-12)


@given(delta=valid_delta(), v=unit_interval())
def test_cdf_boundary_v_zero(delta, v):
    """C(0, v) = 0 for any copula."""
    c = _make(delta)
    assert math.isclose(float(c.get_cdf(0.0, v)), 0.0, abs_tol=2e-12)


@given(delta=valid_delta(), u=unit_interval())
def test_cdf_boundary_v_one(delta, u):
    """C(u, 1) = u for any copula."""
    c = _make(delta)
    assert math.isclose(float(c.get_cdf(u, 1.0)), u, abs_tol=2e-3)


@given(delta=valid_delta(), v=unit_interval())
def test_cdf_boundary_u_one(delta, v):
    """C(1, v) = v for any copula."""
    c = _make(delta)
    assert math.isclose(float(c.get_cdf(1.0, v)), v, abs_tol=2e-3)


# ---------------------------------------------------------------------------
# PDF invariants
# ---------------------------------------------------------------------------

@given(delta=valid_delta(), u=unit_interval(), v=unit_interval())
def test_pdf_nonnegative(delta, u, v):
    """PDF must be non-negative and finite."""
    c = _make(delta)
    val = float(c.get_pdf(u, v))
    assert val >= 0.0
    assert math.isfinite(val)


@given(delta=valid_delta(), u=unit_interval(), v=unit_interval())
def test_pdf_symmetry(delta, u, v):
    """Husler-Reiss copula density is symmetric: c(u, v) = c(v, u)."""
    c = _make(delta)
    assert math.isclose(float(c.get_pdf(u, v)), float(c.get_pdf(v, u)), rel_tol=1e-10, abs_tol=1e-10)


@given(delta=st.floats(min_value=0.1, max_value=10.0,
                       exclude_min=True, exclude_max=True,
                       allow_nan=False, allow_infinity=False,
                       allow_subnormal=False, width=64),
       u=unit_interval(eps=0.05), v=unit_interval(eps=0.05))
@settings(max_examples=70, deadline=None)
def test_pdf_matches_mixed_derivative(delta, u, v):
    """c(u, v) ≈ ∂²C/∂u∂v via 2D central finite difference."""
    c = _make(delta)
    C = lambda a, b: float(c.get_cdf(a, b))
    pdf_num = _mixed_finite_diff(C, u, v, h=2e-4)
    pdf_ana = float(c.get_pdf(u, v))
    assert math.isclose(pdf_ana, pdf_num, rel_tol=3e-2, abs_tol=3e-2)


@pytest.mark.parametrize("delta", [0.3, 1.0, 3.0])
def test_pdf_integrates_to_one(delta):
    """Monte-Carlo check that ∫₀¹∫₀¹ c(u,v) du dv ≈ 1."""
    c = _make(delta)
    rng = np.random.default_rng(0)
    u = rng.uniform(1e-3, 1.0 - 1e-3, size=80_000)
    v = rng.uniform(1e-3, 1.0 - 1e-3, size=80_000)
    assert math.isclose(float(np.mean(c.get_pdf(u, v))), 1.0, rel_tol=2e-2, abs_tol=2e-2)


# ---------------------------------------------------------------------------
# h-functions (conditional CDFs)
# ---------------------------------------------------------------------------

@given(delta=valid_delta(), u=unit_interval(), v=unit_interval())
def test_h_functions_are_probabilities(delta, u, v):
    """h-functions are conditional CDFs and must lie in [0, 1]."""
    c = _make(delta)
    h1 = float(c.partial_derivative_C_wrt_u(u, v))
    h2 = float(c.partial_derivative_C_wrt_v(u, v))
    eps = 1e-12
    assert -eps <= h1 <= 1.0 + eps
    assert -eps <= h2 <= 1.0 + eps


@given(delta=st.floats(min_value=0.1, max_value=10.0,
                       exclude_min=True, exclude_max=True,
                       allow_nan=False, allow_infinity=False,
                       allow_subnormal=False, width=64),
       u=unit_interval(eps=0.05))
@settings(max_examples=50, deadline=None)
def test_h_u_boundary_in_v(delta, u):
    """∂C/∂u: at v ≈ 0 → 0, at v ≈ 1 → 1."""
    c = _make(delta)
    assert math.isclose(float(c.partial_derivative_C_wrt_u(u, 1e-9)), 0.0, abs_tol=1e-3)
    assert math.isclose(float(c.partial_derivative_C_wrt_u(u, 1 - 1e-9)), 1.0, abs_tol=1e-3)


@given(delta=st.floats(min_value=0.1, max_value=10.0,
                       exclude_min=True, exclude_max=True,
                       allow_nan=False, allow_infinity=False,
                       allow_subnormal=False, width=64),
       v=unit_interval(eps=0.05))
@settings(max_examples=50, deadline=None)
def test_h_v_boundary_in_u(delta, v):
    """∂C/∂v: at u ≈ 0 → 0, at u ≈ 1 → 1."""
    c = _make(delta)
    assert math.isclose(float(c.partial_derivative_C_wrt_v(1e-9, v)), 0.0, abs_tol=1e-3)
    assert math.isclose(float(c.partial_derivative_C_wrt_v(1 - 1e-9, v)), 1.0, abs_tol=1e-3)


@given(delta=valid_delta(), u=unit_interval(), v=unit_interval())
def test_h_functions_cross_symmetry(delta, u, v):
    """For symmetric copulas: ∂C/∂u(u,v) = ∂C/∂v(v,u)."""
    c = _make(delta)
    assert math.isclose(
        float(c.partial_derivative_C_wrt_u(u, v)),
        float(c.partial_derivative_C_wrt_v(v, u)),
        rel_tol=1e-8, abs_tol=1e-8,
    )


@given(delta=valid_delta(), u=unit_interval(), v1=unit_interval(), v2=unit_interval())
@settings(max_examples=50)
def test_h_function_monotone_in_v(delta, u, v1, v2):
    """∂C/∂u is monotone increasing in v (it's a CDF in v)."""
    if v1 > v2:
        v1, v2 = v2, v1
    c = _make(delta)
    assert c.partial_derivative_C_wrt_u(u, v1) <= c.partial_derivative_C_wrt_u(u, v2) + 1e-10


# ---------------------------------------------------------------------------
# Derivative cross-check
# ---------------------------------------------------------------------------

@given(delta=st.floats(min_value=0.1, max_value=10.0,
                       exclude_min=True, exclude_max=True,
                       allow_nan=False, allow_infinity=False,
                       allow_subnormal=False, width=64),
       u=unit_interval(eps=0.05), v=unit_interval(eps=0.05))
@settings(max_examples=70, deadline=None)
def test_partial_derivative_matches_finite_diff(delta, u, v):
    """Analytical ∂C/∂u vs numerical central finite difference."""
    c = _make(delta)
    C = lambda a, b: float(c.get_cdf(a, b))
    h_num = _finite_diff(C, u, v, h=2e-4)
    h_ana = float(c.partial_derivative_C_wrt_u(u, v))
    assert math.isclose(h_ana, h_num, rel_tol=3e-2, abs_tol=3e-2)


# ---------------------------------------------------------------------------
# Kendall's tau
# ---------------------------------------------------------------------------

@given(delta=valid_delta())
def test_kendall_tau_range(delta):
    """Kendall's τ must lie in [0, 1) for Husler-Reiss (EV copula, positive dep.)."""
    c = _make(delta)
    tau = float(c.kendall_tau())
    assert math.isfinite(tau)
    assert 0.0 <= tau <= 1.0


def test_kendall_tau_monotone_in_delta():
    """τ increases with δ (stronger dependence at larger δ)."""
    deltas = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    taus = [float(_make(d).kendall_tau()) for d in deltas]
    for i in range(len(taus) - 1):
        assert taus[i] <= taus[i + 1] + 1e-6


def test_kendall_tau_near_zero_at_independence():
    """As δ → 0+, τ → 0 (independence)."""
    assert float(_make(0.03).kendall_tau()) < 0.05


# ---------------------------------------------------------------------------
# Tail dependence
# ---------------------------------------------------------------------------

@given(delta=valid_delta())
def test_lower_tail_dependence_zero(delta):
    """Husler-Reiss is an EV copula: λ_L = 0 for any δ."""
    c = _make(delta)
    assert float(c.LTDC()) == 0.0


@given(delta=valid_delta())
def test_upper_tail_dependence_formula(delta):
    """λ_U = 2·(1 - Φ(1/δ)), where Φ is the standard normal CDF."""
    c = _make(delta)
    lam_th = float(2.0 * (1.0 - norm.cdf(1.0 / delta)))
    assert math.isclose(float(c.UTDC()), lam_th, rel_tol=0, abs_tol=0)


@given(delta=valid_delta())
def test_upper_tail_dependence_positive(delta):
    c = HuslerReissCopula()
    c.set_parameters([delta])

    # avoid the regime where 1/delta is huge -> 1 - Phi(1/delta) underflows to 0.0
    assume(delta > 0.2)

    assert c.UTDC() > 0.0


def test_upper_tail_dependence_increases_with_delta():
    """λ_U = 2·(1 - Φ(1/δ)) increases with δ."""
    deltas = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    utdcs = [float(_make(d).UTDC()) for d in deltas]
    for i in range(len(utdcs) - 1):
        assert utdcs[i] < utdcs[i + 1]


def test_upper_tail_dependence_limits():
    """As δ → 0+, λ_U → 0. As δ → ∞, λ_U → 1."""
    assert _make(0.03).UTDC() < 0.05
    assert _make(25.0).UTDC() > 0.80


# ---------------------------------------------------------------------------
# Blomqvist beta
# ---------------------------------------------------------------------------

@given(delta=valid_delta())
def test_blomqvist_beta_matches_closed_form(delta):
    """HR closed-form: β(δ) = 2^{2-2Φ(1/δ)} - 1."""
    c = _make(delta)

    beta = float(c.blomqvist_beta())
    beta_cf = 2.0 ** (2.0 - 2.0 * norm.cdf(1.0 / float(delta))) - 1.0

    assert math.isfinite(beta)
    assert 0.0 <= beta <= 1.0   # HR is positive-dependence only
    assert math.isclose(beta, beta_cf, rel_tol=1e-12, abs_tol=1e-12)

@given(delta=valid_delta())
@settings(max_examples=30, deadline=None)
def test_blomqvist_beta_consistent_with_definition(delta):
    c = _make(delta)
    beta = float(c.blomqvist_beta())
    beta_def = 4.0 * float(c.get_cdf(0.5, 0.5)) - 1.0
    assert math.isclose(beta, beta_def, rel_tol=1e-10, abs_tol=1e-10)


# ---------------------------------------------------------------------------
# Independence case (δ → 0+)
# ---------------------------------------------------------------------------

def test_cdf_independence_limit_bulk():
    """As δ → 0+, HR approaches independence: C(u,v) ≈ u·v in the bulk."""
    c = _make(0.03)
    grid = np.linspace(0.1, 0.9, 9)
    for u in grid:
        for v in grid:
            assert math.isclose(float(c.get_cdf(u, v)), u * v, rel_tol=0.08, abs_tol=0.08)


@given(u=unit_interval(eps=0.1), v=unit_interval(eps=0.1))
def test_independence_pdf_near_one(u, v):
    """As δ → 0+, copula density → 1 on the bulk of (0,1)²."""
    c = _make(0.03)
    assert math.isclose(float(c.get_pdf(u, v)), 1.0, rel_tol=0.15, abs_tol=0.15)


@given(u=unit_interval(eps=0.1), v=unit_interval(eps=0.1))
def test_independence_h_functions_identity(u, v):
    """As δ → 0+: ∂C/∂u ≈ v and ∂C/∂v ≈ u."""
    c = _make(0.03)
    assert math.isclose(float(c.partial_derivative_C_wrt_u(u, v)), v, rel_tol=0.15, abs_tol=0.15)
    assert math.isclose(float(c.partial_derivative_C_wrt_v(u, v)), u, rel_tol=0.15, abs_tol=0.15)


def test_cdf_comonotone_limit():
    """As δ → ∞, HR approaches comonotonicity: C(u,v) → min(u,v)."""
    c = _make(25.0)
    for u, v in [(0.2, 0.8), (0.6, 0.4), (0.3, 0.3)]:
        assert math.isclose(float(c.get_cdf(u, v)), min(u, v), rel_tol=0.08, abs_tol=0.08)


# ---------------------------------------------------------------------------
# init_from_data round-trip (slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("delta", [0.2, 1.0, 3.0, 10.0])
def test_init_from_data_roundtrip(delta):
    """Generate samples with known δ, verify init_from_data returns valid parameters."""
    c = _make(delta)
    data = c.sample(12_000, rng=np.random.default_rng(0))

    c2 = HuslerReissCopula()
    res = c2.init_from_data(data[:, 0], data[:, 1])

    # Support both patterns: returned params OR in-place
    if isinstance(res, (list, tuple, np.ndarray)) and np.asarray(res).size == 1:
        c2.set_parameters([float(np.asarray(res).ravel()[0])])

    d_hat = float(c2.get_parameters()[0])
    assert d_hat > 0.0
    assert math.isfinite(d_hat)


# ---------------------------------------------------------------------------
# Sampling sanity check (slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@given(delta=st.floats(min_value=0.1, max_value=10.0,
                       exclude_min=True, exclude_max=True,
                       allow_nan=False, allow_infinity=False,
                       allow_subnormal=False, width=64))
@settings(max_examples=10, deadline=None)
def test_sampling_empirical_tau_close(delta):
    """Empirical Kendall τ from samples should be close to theoretical."""
    c = _make(delta)
    data = c.sample(8_000, rng=np.random.default_rng(0))
    tau_emp = float(kendalltau(data[:, 0], data[:, 1]).correlation)
    tau_th = float(c.kendall_tau())

    assert math.isfinite(tau_emp) and math.isfinite(tau_th)
    assert abs(tau_emp - tau_th) < 0.06


# ---------------------------------------------------------------------------
# Shape checks (vectorised input)
# ---------------------------------------------------------------------------

def test_vectorised_shapes():
    """CDF, PDF, and sample must return arrays with correct shapes."""
    c = _make(1.5)
    u = np.linspace(0.05, 0.95, 13)
    v = np.linspace(0.05, 0.95, 13)
    assert c.get_cdf(u, v).shape == (13,)
    assert c.get_pdf(u, v).shape == (13,)
    assert c.sample(256).shape == (256, 2)


def test_vectorised_inputs_are_pairwise_not_grid():
    """Vectorized get_cdf/get_pdf operate pairwise on (u[i], v[i]), not on the Cartesian grid."""
    c = _make(1.5)
    u = np.array([0.2, 0.8])
    v = np.array([0.3, 0.7])

    cdf_vec = c.get_cdf(u, v)
    cdf_pair = np.array([
        c.get_cdf(float(u[0]), float(v[0])),
        c.get_cdf(float(u[1]), float(v[1])),
    ])

    assert cdf_vec.shape == (2,)
    assert np.allclose(cdf_vec, cdf_pair, rtol=1e-10, atol=1e-12)