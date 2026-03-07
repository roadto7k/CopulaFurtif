"""
Comprehensive unit-test suite for the bivariate **BB6** Copula (Joe & Hu 1996).

Conforms to the canonical test structure defined in test_coverage_analysis.md.

Run with:
    pytest -q                    # fast tests only
    pytest -q -m slow            # includes heavy sampling / init_from_data tests

Reference: Joe H. (1997) §4.22, images p.200-201.
"""

import math
import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from CopulaFurtif.core.copulas.domain.models.archimedean.BB6 import BB6Copula


# =============================================================================
# Helpers & strategies
# =============================================================================

EPS_U = 1e-3   # stay away from 0/1 for unit-interval draws

# θ ∈ (1, ∞),  δ ∈ (1, ∞) — both bounds excluded by parent class
# We cap at 10 to keep quadrature tractable.
theta_st = st.floats(min_value=1.05, max_value=10.0,
                     allow_nan=False, allow_infinity=False)
delta_st = st.floats(min_value=1.05, max_value=10.0,
                     allow_nan=False, allow_infinity=False)
unit_st  = st.floats(min_value=EPS_U, max_value=1.0 - EPS_U,
                     allow_nan=False, allow_infinity=False)


def _fd(f, x, y, h=1e-6):
    """Central finite difference ∂f/∂x."""
    return (f(x + h, y) - f(x - h, y)) / (2.0 * h)


# =============================================================================
# 1. PARAMETERS
# =============================================================================

@given(theta=theta_st, delta=delta_st)
def test_parameter_roundtrip(theta, delta):
    c = BB6Copula()
    c.set_parameters([theta, delta])
    p = c.get_parameters()
    assert math.isclose(p[0], theta, rel_tol=1e-12)
    assert math.isclose(p[1], delta, rel_tol=1e-12)


@given(
    theta=st.floats(max_value=1.0, allow_nan=False, allow_infinity=False),
    delta=delta_st,
)
def test_parameter_out_of_bounds_theta(theta, delta):
    """θ ≤ 1 must be rejected."""
    c = BB6Copula()
    with pytest.raises((ValueError, Exception)):
        c.set_parameters([theta, delta])


@given(
    theta=theta_st,
    delta=st.floats(max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_parameter_out_of_bounds_delta(theta, delta):
    """δ ≤ 1 must be rejected."""
    c = BB6Copula()
    with pytest.raises((ValueError, Exception)):
        c.set_parameters([theta, delta])


@pytest.mark.parametrize("params", [
    [1.0, 2.0],   # theta exactly at lower bound
    [2.0, 1.0],   # delta exactly at lower bound
    [1.0, 1.0],   # both at boundary
])
def test_parameter_at_boundary_rejected(params):
    c = BB6Copula()
    with pytest.raises((ValueError, Exception)):
        c.set_parameters(params)


@pytest.mark.parametrize("bad_params", [
    [],
    [2.0],
    [2.0, 2.0, 2.0],
])
def test_parameter_wrong_size(bad_params):
    c = BB6Copula()
    with pytest.raises(Exception):
        c.set_parameters(bad_params)


# =============================================================================
# 2. CDF INVARIANTS
# =============================================================================

@given(theta=theta_st, delta=delta_st, u=unit_st, v=unit_st)
def test_cdf_bounds(theta, delta, u, v):
    c = BB6Copula()
    c.set_parameters([theta, delta])
    val = c.get_cdf(u, v)
    assert 0.0 <= val <= 1.0


@given(theta=theta_st, delta=delta_st, u1=unit_st, u2=unit_st, v=unit_st)
def test_cdf_monotone_in_u(theta, delta, u1, u2, v):
    if u1 > u2:
        u1, u2 = u2, u1
    c = BB6Copula()
    c.set_parameters([theta, delta])
    assert c.get_cdf(u1, v) <= c.get_cdf(u2, v) + 1e-12


@given(theta=theta_st, delta=delta_st, u=unit_st, v=unit_st)
def test_cdf_symmetry(theta, delta, u, v):
    """BB6 is symmetric: C(u,v) = C(v,u)."""
    c = BB6Copula()
    c.set_parameters([theta, delta])
    assert math.isclose(c.get_cdf(u, v), c.get_cdf(v, u), rel_tol=1e-10, abs_tol=1e-12)


# =============================================================================
# 3. FRÉCHET–HOEFFDING BOUNDARIES
# =============================================================================

@given(theta=theta_st, delta=delta_st, u=unit_st)
def test_cdf_boundary_v_zero(theta, delta, u):
    """C(u, 0) = 0."""
    c = BB6Copula()
    c.set_parameters([theta, delta])
    assert abs(c.get_cdf(u, 1e-10)) < 1e-6


@given(theta=theta_st, delta=delta_st, v=unit_st)
def test_cdf_boundary_u_zero(theta, delta, v):
    """C(0, v) = 0."""
    c = BB6Copula()
    c.set_parameters([theta, delta])
    assert abs(c.get_cdf(1e-10, v)) < 1e-6


@given(theta=theta_st, delta=delta_st, u=unit_st)
def test_cdf_boundary_v_one(theta, delta, u):
    """C(u, 1−ε) → u as ε → 0.  Use ε=1e-3 (small enough for all θ∈[1,10])."""
    # (1−ε)^θ ≤ (0.001)^1.05 ≈ 6e-4  → y ≈ 6e-4, negligible vs x.
    c = BB6Copula()
    c.set_parameters([theta, delta])
    assert math.isclose(c.get_cdf(u, 1.0 - 1e-3), u, rel_tol=5e-3, abs_tol=5e-3)


@given(theta=theta_st, delta=delta_st, v=unit_st)
def test_cdf_boundary_u_one(theta, delta, v):
    """C(1−ε, v) → v as ε → 0."""
    c = BB6Copula()
    c.set_parameters([theta, delta])
    assert math.isclose(c.get_cdf(1.0 - 1e-3, v), v, rel_tol=5e-3, abs_tol=5e-3)


# =============================================================================
# 4. PDF INVARIANTS
# =============================================================================

@given(theta=theta_st, delta=delta_st, u=unit_st, v=unit_st)
def test_pdf_nonnegative(theta, delta, u, v):
    c = BB6Copula()
    c.set_parameters([theta, delta])
    assert c.get_pdf(u, v) >= 0.0


@given(theta=theta_st, delta=delta_st, u=unit_st, v=unit_st)
def test_pdf_symmetry(theta, delta, u, v):
    """BB6 is symmetric: c(u,v) = c(v,u)."""
    c = BB6Copula()
    c.set_parameters([theta, delta])
    assert math.isclose(c.get_pdf(u, v), c.get_pdf(v, u), rel_tol=1e-8, abs_tol=1e-10)


@given(theta=theta_st, delta=delta_st, u=unit_st, v=unit_st)
@settings(max_examples=100)
def test_pdf_matches_mixed_derivative(theta, delta, u, v):
    """c(u,v) ≈ ∂²C/∂u∂v via 2D central finite differences."""
    assume(0.02 < u < 0.98 and 0.02 < v < 0.98)
    c = BB6Copula()
    c.set_parameters([theta, delta])
    h = 1e-4
    num = (
        c.get_cdf(u + h, v + h)
        - c.get_cdf(u + h, v - h)
        - c.get_cdf(u - h, v + h)
        + c.get_cdf(u - h, v - h)
    ) / (4.0 * h * h)
    ana = c.get_pdf(u, v)
    assert math.isclose(ana, num, rel_tol=1e-2, abs_tol=1e-4), \
        f"pdf={ana:.6f}, fd={num:.6f}"


@pytest.mark.parametrize("theta,delta", [
    (1.5, 1.5), (2.0, 2.0), (3.0, 1.5), (5.0, 3.0),
])
def test_pdf_integrates_to_one(theta, delta):
    """Monte-Carlo integration: E[pdf(U,V)] ≈ 1."""
    rng = np.random.default_rng(42)
    n = 200_000
    u = rng.uniform(0.001, 0.999, n)
    v = rng.uniform(0.001, 0.999, n)
    c = BB6Copula()
    c.set_parameters([theta, delta])
    integral = c.get_pdf(u, v).mean()   # E over uniform = ∫∫ c du dv
    assert math.isclose(integral, 1.0, rel_tol=5e-3), \
        f"θ={theta}, δ={delta}: ∫∫c={integral:.5f}"


# =============================================================================
# 5. H-FUNCTIONS  (partial derivatives / conditional CDF)
# =============================================================================

@given(theta=theta_st, delta=delta_st, u=unit_st, v=unit_st)
def test_h_functions_are_probabilities(theta, delta, u, v):
    c = BB6Copula()
    c.set_parameters([theta, delta])
    h_u = c.partial_derivative_C_wrt_u(u, v)
    h_v = c.partial_derivative_C_wrt_v(u, v)
    assert -1e-8 <= h_u <= 1.0 + 1e-8
    assert -1e-8 <= h_v <= 1.0 + 1e-8


@given(theta=theta_st, delta=delta_st, u=unit_st)
def test_h_u_boundary_in_v(theta, delta, u):
    """∂C/∂u(u, v→0) → 0  and  ∂C/∂u(u, v→1) → 1."""
    c = BB6Copula()
    c.set_parameters([theta, delta])
    assert c.partial_derivative_C_wrt_u(u, 1e-9) < 0.05
    assert c.partial_derivative_C_wrt_u(u, 1.0 - 1e-9) > 0.95


@given(theta=theta_st, delta=delta_st, v=unit_st)
def test_h_v_boundary_in_u(theta, delta, v):
    """∂C/∂v(u→0, v) → 0  and  ∂C/∂v(u→1, v) → 1."""
    c = BB6Copula()
    c.set_parameters([theta, delta])
    assert c.partial_derivative_C_wrt_v(1e-9, v) < 0.05
    assert c.partial_derivative_C_wrt_v(1.0 - 1e-9, v) > 0.95


@given(theta=theta_st, delta=delta_st, u=unit_st, v=unit_st)
def test_h_functions_cross_symmetry(theta, delta, u, v):
    """BB6 symmetric ⟹ ∂C/∂u(u,v) = ∂C/∂v(v,u)."""
    c = BB6Copula()
    c.set_parameters([theta, delta])
    assert math.isclose(
        c.partial_derivative_C_wrt_u(u, v),
        c.partial_derivative_C_wrt_v(v, u),
        rel_tol=1e-8, abs_tol=1e-10,
    )


@given(theta=theta_st, delta=delta_st, u=unit_st, v1=unit_st, v2=unit_st)
def test_h_function_monotone_in_v(theta, delta, u, v1, v2):
    """∂C/∂u is monotone increasing in v."""
    if v1 > v2:
        v1, v2 = v2, v1
    c = BB6Copula()
    c.set_parameters([theta, delta])
    assert c.partial_derivative_C_wrt_u(u, v1) <= c.partial_derivative_C_wrt_u(u, v2) + 1e-10


# =============================================================================
# 6. DERIVATIVES — analytical vs. finite difference
# =============================================================================

@given(theta=theta_st, delta=delta_st, u=unit_st, v=unit_st)
@settings(max_examples=150)
def test_partial_derivative_matches_finite_diff(theta, delta, u, v):
    assume(0.02 < u < 0.98 and 0.02 < v < 0.98)
    c = BB6Copula()
    c.set_parameters([theta, delta])

    num_u = _fd(lambda x, y: c.get_cdf(x, y), u, v)
    num_v = _fd(lambda x, y: c.get_cdf(y, x), v, u)

    ana_u = c.partial_derivative_C_wrt_u(u, v)
    ana_v = c.partial_derivative_C_wrt_v(u, v)

    assert math.isclose(ana_u, num_u, rel_tol=1e-3, abs_tol=1e-4), \
        f"∂C/∂u: ana={ana_u:.6f}, fd={num_u:.6f}"
    assert math.isclose(ana_v, num_v, rel_tol=1e-3, abs_tol=1e-4), \
        f"∂C/∂v: ana={ana_v:.6f}, fd={num_v:.6f}"


# =============================================================================
# 7. KENDALL'S TAU
# =============================================================================

@given(theta=theta_st, delta=delta_st)
@settings(max_examples=50)
def test_kendall_tau_range(theta, delta):
    """τ ∈ (0, 1) for BB6 (positive dependence copula, θ,δ > 1)."""
    c = BB6Copula()
    c.set_parameters([theta, delta])
    tau = c.kendall_tau()
    assert 0.0 < tau < 1.0, f"τ={tau} out of range for θ={theta}, δ={delta}"


@pytest.mark.parametrize("thetas,deltas", [
    ([1.5, 3.0, 6.0], [2.0, 2.0, 2.0]),   # increasing theta → increasing tau
    ([2.0, 2.0, 2.0], [1.5, 3.0, 6.0]),   # increasing delta → increasing tau
])
def test_kendall_tau_monotone_in_params(thetas, deltas):
    """τ increases with both θ and δ."""
    c = BB6Copula()
    taus = []
    for th, de in zip(thetas, deltas):
        c.set_parameters([th, de])
        taus.append(c.kendall_tau())
    assert all(taus[i] <= taus[i + 1] + 1e-6 for i in range(len(taus) - 1)), \
        f"τ not monotone: {taus}"


@pytest.mark.slow
@pytest.mark.parametrize("theta,delta", [
    (1.5, 1.5), (2.0, 2.0), (3.0, 3.0), (5.0, 2.0),
])
def test_kendall_tau_vs_empirical(theta, delta):
    """Empirical τ̂ ≈ theoretical τ within 4σ (n=5000)."""
    from scipy.stats import kendalltau as _kt
    rng = np.random.default_rng(1234)
    c = BB6Copula()
    c.set_parameters([theta, delta])
    data = c.sample(5000, rng=rng)
    tau_emp, _ = _kt(data[:, 0], data[:, 1])
    tau_th = c.kendall_tau()
    n = 5000
    sigma = math.sqrt(2.0 * (2 * n + 5) / (9 * n * (n - 1)))
    print(f"θ={theta}, δ={delta}: τ_emp={tau_emp:.4f}, τ_th={tau_th:.4f}, tol={4*sigma:.4f}")
    assert abs(tau_emp - tau_th) <= 4 * sigma


# =============================================================================
# 8. TAIL DEPENDENCE
# =============================================================================

@given(theta=theta_st, delta=delta_st)
def test_tail_dependence_zero_lower(theta, delta):
    """BB6 has no lower tail dependence (λ_L = 0)."""
    c = BB6Copula()
    c.set_parameters([theta, delta])
    assert c.LTDC() == 0.0


@given(theta=theta_st, delta=delta_st)
def test_tail_dependence_upper_formula(theta, delta):
    """λ_U = 2 − 2^{1/(θδ)} and is in (0, 1)."""
    c = BB6Copula()
    c.set_parameters([theta, delta])
    expected = 2.0 - 2.0 ** (1.0 / (theta * delta))
    got = c.UTDC()
    assert math.isclose(got, expected, rel_tol=1e-12)
    assert 0.0 < got < 1.0


@pytest.mark.parametrize("thetas,delta", [
    ([1.5, 2.0, 3.0, 5.0], 2.0),
])
def test_utdc_monotone_in_theta(thetas, delta):
    """λ_U increases with θ."""
    c = BB6Copula()
    utdcs = []
    for th in thetas:
        c.set_parameters([th, delta])
        utdcs.append(c.UTDC())
    assert all(utdcs[i] <= utdcs[i + 1] + 1e-12 for i in range(len(utdcs) - 1))


@pytest.mark.parametrize("theta,deltas", [
    (2.0, [1.5, 2.0, 3.0, 5.0]),
])
def test_utdc_monotone_in_delta(theta, deltas):
    """λ_U increases with δ."""
    c = BB6Copula()
    utdcs = []
    for de in deltas:
        c.set_parameters([theta, de])
        utdcs.append(c.UTDC())
    assert all(utdcs[i] <= utdcs[i + 1] + 1e-12 for i in range(len(utdcs) - 1))


# =============================================================================
# 9. BLOMQVIST β
# =============================================================================

@given(theta=theta_st, delta=delta_st)
@settings(max_examples=80)
def test_blomqvist_beta_matches_definition(theta, delta):
    """β = 4·C(½, ½) − 1."""
    c = BB6Copula()
    c.set_parameters([theta, delta])
    expected = 4.0 * c.get_cdf(0.5, 0.5) - 1.0
    got = c.blomqvist_beta()
    assert math.isclose(got, expected, rel_tol=1e-10, abs_tol=1e-12)


@given(theta=theta_st, delta=delta_st)
def test_blomqvist_beta_range(theta, delta):
    """β ∈ (−1, 1)."""
    c = BB6Copula()
    c.set_parameters([theta, delta])
    beta = c.blomqvist_beta()
    assert -1.0 < beta < 1.0


# =============================================================================
# 10. INDEPENDENCE CASE  — θ→1, δ→1  (boundary, not reachable exactly)
#     BB6 nests Gumbel (θ=1) and Joe (δ=1); true independence is θ=δ=1 limit.
#     We just check near-independence gives sensible results.
# =============================================================================

@pytest.mark.parametrize("theta,delta", [(1.01, 1.01)])
def test_near_independence_cdf(theta, delta):
    """Near θ=δ=1, C(u,v) should be close to u·v."""
    c = BB6Copula()
    c.set_parameters([theta, delta])
    rng = np.random.default_rng(0)
    u = rng.uniform(0.1, 0.9, 200)
    v = rng.uniform(0.1, 0.9, 200)
    cdf_vals = c.get_cdf(u, v)
    prod_vals = u * v
    assert np.mean(np.abs(cdf_vals - prod_vals)) < 0.05


# =============================================================================
# 11. INIT FROM DATA  (@slow)
# =============================================================================

@pytest.mark.slow
@pytest.mark.parametrize("theta,delta", [
    (2.0, 2.0), (3.0, 1.5), (1.5, 3.0), (5.0, 2.0),
])
def test_init_from_data_roundtrip(theta, delta):
    """
    init_from_data should return valid parameters that reproduce
    Kendall's tau within 15% of the true value.

    Note: BB6 has 2 parameters but only 1 dependence summary (τ) is matched,
    so individual (θ, δ) recovery is not guaranteed — only τ reproduction is.
    """
    rng = np.random.default_rng(99)
    c = BB6Copula()
    c.set_parameters([theta, delta])
    data = c.sample(2000, rng=rng)
    params_init = c.init_from_data(data[:, 0], data[:, 1])

    # Params must be in-bounds
    assert params_init[0] > 1.0, f"θ_init={params_init[0]:.3f} not > 1"
    assert params_init[1] > 1.0, f"δ_init={params_init[1]:.3f} not > 1"

    # τ from init params should be close to true τ (within 15%)
    tau_true = c.kendall_tau([theta, delta])
    tau_init = c.kendall_tau(params_init)
    assert abs(tau_init - tau_true) < 0.15, \
        f"τ_init={tau_init:.4f} vs τ_true={tau_true:.4f} (θ={theta}, δ={delta})"


# =============================================================================
# 12. SAMPLING  (@slow)
# =============================================================================

@pytest.mark.slow
@pytest.mark.parametrize("theta,delta", [
    (2.0, 2.0), (3.0, 1.5), (5.0, 3.0),
])
def test_empirical_kendall_tau_close(theta, delta):
    """Empirical τ from sample within 4σ of theoretical τ (n=3000)."""
    from scipy.stats import kendalltau as _kt
    rng = np.random.default_rng(777)
    c = BB6Copula()
    c.set_parameters([theta, delta])
    data = c.sample(3000, rng=rng)
    assert data.shape == (3000, 2)
    tau_emp, _ = _kt(data[:, 0], data[:, 1])
    tau_th = c.kendall_tau()
    n = 3000
    sigma = math.sqrt(2.0 * (2 * n + 5) / (9 * n * (n - 1)))
    print(f"θ={theta}, δ={delta}: τ_emp={tau_emp:.4f}, τ_th={tau_th:.4f}")
    assert abs(tau_emp - tau_th) <= 4 * sigma, \
        f"θ={theta}, δ={delta}: τ_emp={tau_emp:.4f}, τ_th={tau_th:.4f}"


# =============================================================================
# 13. VECTORIZATION
# =============================================================================

def test_vectorised_shapes():
    c = BB6Copula()
    c.set_parameters([2.0, 2.0])
    n = 50
    u = np.linspace(0.05, 0.95, n)
    v = np.linspace(0.05, 0.95, n)
    assert c.get_cdf(u, v).shape == (n,)
    assert c.get_pdf(u, v).shape == (n,)
    assert c.partial_derivative_C_wrt_u(u, v).shape == (n,)
    assert c.partial_derivative_C_wrt_v(u, v).shape == (n,)


def test_vectorised_inputs_are_pairwise_not_grid():
    """(u[i], v[i]) evaluated pairwise, not on a (n×n) grid."""
    c = BB6Copula()
    c.set_parameters([2.0, 2.0])
    u = np.array([0.2, 0.5, 0.8])
    v = np.array([0.3, 0.5, 0.7])
    result = c.get_cdf(u, v)
    expected = np.array([c.get_cdf(u[i], v[i]) for i in range(3)])
    np.testing.assert_allclose(result, expected, rtol=1e-12)