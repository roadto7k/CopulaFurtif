"""Unit-test suite for the Marshall-Olkin Copula.

Run with:  pytest -q  (add -m 'not slow' on CI to skip heavy tests)

Marshall-Olkin copula properties:
    - pi1, pi2 in (0, 1) strictly.
    - Asymmetric copula: C(u,v;pi1,pi2) = C(v,u;pi2,pi1).
    - Singular component along the curve u^pi1 == v^pi2.
    - No lower tail dependence; lambda_U = min(pi1, pi2).
    - tau = pi1*pi2 / (pi1 + pi2 - pi1*pi2).
"""

import math
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, assume
from scipy.stats import kendalltau

from CopulaFurtif.core.copulas.domain.models.exotic.marshall_olkin import MarshallOlkinCopula


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

@st.composite
def valid_pi(draw):
    return draw(st.floats(min_value=0.02, max_value=0.98,
                          exclude_min=True, exclude_max=True,
                          allow_nan=False, allow_infinity=False,
                          allow_subnormal=False, width=64))


@st.composite
def unit_interval(draw, eps=1e-3):
    return draw(st.floats(min_value=eps, max_value=1.0 - eps,
                          exclude_min=True, exclude_max=True,
                          allow_nan=False, allow_infinity=False,
                          allow_subnormal=False, width=64))


def _clip01(x, eps=1e-12):
    return min(max(float(x), eps), 1.0 - eps)


def _clipped_f(f, x, y, eps=1e-12):
    return f(_clip01(x, eps), _clip01(y, eps))


def _finite_diff(f, x, y, h=1e-5, eps=1e-12):
    return (_clipped_f(f, x + h, y, eps) - _clipped_f(f, x - h, y, eps)) / (2.0 * h)


def _mixed_finite_diff(C, u, v, h=1e-5, eps=1e-12):
    return (
        _clipped_f(C, u + h, v + h, eps)
        - _clipped_f(C, u + h, v - h, eps)
        - _clipped_f(C, u - h, v + h, eps)
        + _clipped_f(C, u - h, v - h, eps)
    ) / (4.0 * h * h)


def _away_from_singular(u, v, pi1, pi2, tol=5e-3):
    return abs((u ** pi1) - (v ** pi2)) > tol


def _tau_theoretical(pi1, pi2):
    return (pi1 * pi2) / (pi1 + pi2 - pi1 * pi2)


def _ac_mass(pi1, pi2):
    return 1.0 - _tau_theoretical(pi1, pi2)


# ---------------------------------------------------------------------------
# Parameter tests
# ---------------------------------------------------------------------------

@given(pi1=valid_pi(), pi2=valid_pi())
def test_parameter_roundtrip(pi1, pi2):
    """set_parameters then get_parameters should return the same values."""
    c = MarshallOlkinCopula()
    c.set_parameters([pi1, pi2])
    p = c.get_parameters()
    assert math.isclose(float(p[0]), pi1, rel_tol=0, abs_tol=0)
    assert math.isclose(float(p[1]), pi2, rel_tol=0, abs_tol=0)


@given(
    bad=st.one_of(
        st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1.0, allow_nan=False, allow_infinity=False),
    ),
    ok=valid_pi(),
)
def test_parameter_out_of_bounds_extreme(bad, ok):
    """Values <= 0 or >= 1 must be rejected (bounds are (0,1) strict)."""
    c = MarshallOlkinCopula()
    with pytest.raises(ValueError):
        c.set_parameters([bad, ok])
    with pytest.raises(ValueError):
        c.set_parameters([ok, bad])


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.1])
def test_parameter_at_boundary_rejected(bad):
    """Exact boundary values must be rejected."""
    c = MarshallOlkinCopula()
    with pytest.raises(ValueError):
        c.set_parameters([bad, 0.5])
    with pytest.raises(ValueError):
        c.set_parameters([0.5, bad])


def test_parameter_wrong_size():
    """Passing wrong number of parameters must raise ValueError."""
    c = MarshallOlkinCopula()
    with pytest.raises(ValueError):
        c.set_parameters([0.5])
    with pytest.raises(ValueError):
        c.set_parameters([0.3, 0.4, 0.5])
    with pytest.raises(ValueError):
        c.set_parameters([])


# ---------------------------------------------------------------------------
# CDF invariants
# ---------------------------------------------------------------------------

@given(pi1=valid_pi(), pi2=valid_pi(), u=unit_interval(), v=unit_interval())
def test_cdf_bounds(pi1, pi2, u, v):
    """CDF must lie in [0, 1]."""
    c = MarshallOlkinCopula()
    c.set_parameters([pi1, pi2])
    assert 0.0 <= float(c.get_cdf(u, v)) <= 1.0


@given(pi1=valid_pi(), pi2=valid_pi(), u=unit_interval(), v=unit_interval())
def test_cdf_monotone_in_u(pi1, pi2, u, v):
    """C(u, v) <= C(u2, v) for u2 > u."""
    c = MarshallOlkinCopula()
    c.set_parameters([pi1, pi2])
    u2 = min(u + 1e-3, 0.999)
    assert c.get_cdf(u, v) <= c.get_cdf(u2, v) + 1e-12


@given(pi1=valid_pi(), pi2=valid_pi(), u=unit_interval(), v=unit_interval())
def test_cdf_swap_parameter_symmetry(pi1, pi2, u, v):
    """Marshall-Olkin satisfies C(u,v;pi1,pi2) = C(v,u;pi2,pi1)."""
    c1 = MarshallOlkinCopula()
    c2 = MarshallOlkinCopula()
    c1.set_parameters([pi1, pi2])
    c2.set_parameters([pi2, pi1])
    assert math.isclose(float(c1.get_cdf(u, v)), float(c2.get_cdf(v, u)), rel_tol=1e-12, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# Frechet-Hoeffding boundary conditions
# ---------------------------------------------------------------------------

@given(pi1=valid_pi(), pi2=valid_pi(), u=unit_interval())
def test_cdf_boundary_u_zero(pi1, pi2, u):
    """C(u, 0) = 0 for any copula."""
    c = MarshallOlkinCopula()
    c.set_parameters([pi1, pi2])
    assert math.isclose(float(c.get_cdf(u, 0.0)), 0.0, abs_tol=1e-12)


@given(pi1=valid_pi(), pi2=valid_pi(), v=unit_interval())
def test_cdf_boundary_v_zero(pi1, pi2, v):
    """C(0, v) = 0 for any copula."""
    c = MarshallOlkinCopula()
    c.set_parameters([pi1, pi2])
    assert math.isclose(float(c.get_cdf(0.0, v)), 0.0, abs_tol=1e-12)


@given(pi1=valid_pi(), pi2=valid_pi(), u=unit_interval())
def test_cdf_boundary_v_one(pi1, pi2, u):
    """C(u, 1) = u for any copula."""
    c = MarshallOlkinCopula()
    c.set_parameters([pi1, pi2])
    assert math.isclose(float(c.get_cdf(u, 1.0)), u, abs_tol=1e-8)


@given(pi1=valid_pi(), pi2=valid_pi(), v=unit_interval())
def test_cdf_boundary_u_one(pi1, pi2, v):
    """C(1, v) = v for any copula."""
    c = MarshallOlkinCopula()
    c.set_parameters([pi1, pi2])
    assert math.isclose(float(c.get_cdf(1.0, v)), v, abs_tol=1e-8)


# ---------------------------------------------------------------------------
# PDF invariants
# ---------------------------------------------------------------------------

@given(pi1=valid_pi(), pi2=valid_pi(), u=unit_interval(), v=unit_interval())
def test_pdf_nonnegative(pi1, pi2, u, v):
    """AC density must be non-negative."""
    c = MarshallOlkinCopula()
    c.set_parameters([pi1, pi2])
    assert float(c.get_pdf(u, v)) >= 0.0


@given(pi1=valid_pi(), pi2=valid_pi(), u=unit_interval(eps=0.05), v=unit_interval(eps=0.05))
@settings(max_examples=60, deadline=None)
def test_pdf_matches_mixed_derivative_off_singular(pi1, pi2, u, v):
    """Off the singular curve, the AC density matches the mixed finite difference."""
    assume(_away_from_singular(u, v, pi1, pi2))
    c = MarshallOlkinCopula()
    c.set_parameters([pi1, pi2])
    C = lambda a, b: float(c.get_cdf(a, b))
    pdf_num = _mixed_finite_diff(C, u, v, h=2e-4)
    pdf_ana = float(c.get_pdf(u, v))
    assert math.isfinite(pdf_ana)
    assert math.isclose(pdf_ana, pdf_num, rel_tol=3e-2, abs_tol=3e-2)


@pytest.mark.slow
def test_pdf_integrates_to_ac_mass():
    """
    Marshall-Olkin has a singular component, so the Lebesgue density integrates
    to the AC mass only: 1 - tau = 1 - pi1*pi2/(pi1+pi2-pi1*pi2).
    """
    pi1, pi2 = 0.35, 0.7
    c = MarshallOlkinCopula()
    c.set_parameters([pi1, pi2])
    n = 300
    xs = (np.arange(n) + 0.5) / n
    U, V = np.meshgrid(xs, xs, indexing="ij")
    est = float(np.mean(c.get_pdf(U.ravel(), V.ravel())))
    target = _ac_mass(pi1, pi2)
    assert math.isclose(est, target, rel_tol=3e-2, abs_tol=3e-2), (est, target)


# ---------------------------------------------------------------------------
# h-functions (conditional CDFs)
# ---------------------------------------------------------------------------

@given(pi1=valid_pi(), pi2=valid_pi(), u=unit_interval(), v=unit_interval())
def test_h_functions_are_probabilities(pi1, pi2, u, v):
    """h-functions are conditional CDFs and must lie in [0, 1]."""
    c = MarshallOlkinCopula()
    c.set_parameters([pi1, pi2])
    eps = 1e-12
    assert -eps <= float(c.partial_derivative_C_wrt_u(u, v)) <= 1.0 + eps
    assert -eps <= float(c.partial_derivative_C_wrt_v(u, v)) <= 1.0 + eps


@given(pi1=valid_pi(), pi2=valid_pi(), u=unit_interval(eps=0.05))
@settings(max_examples=50, deadline=None)
def test_h_u_boundary_in_v(pi1, pi2, u):
    """dC/du: at v~0 -> 0, at v~1 -> 1."""
    c = MarshallOlkinCopula()
    c.set_parameters([pi1, pi2])
    assert math.isclose(float(c.partial_derivative_C_wrt_u(u, 1e-10)), 0.0, abs_tol=1e-4)
    assert math.isclose(float(c.partial_derivative_C_wrt_u(u, 1 - 1e-10)), 1.0, abs_tol=1e-4)


@given(pi1=valid_pi(), pi2=valid_pi(), v=unit_interval(eps=0.05))
@settings(max_examples=50, deadline=None)
def test_h_v_boundary_in_u(pi1, pi2, v):
    """dC/dv: at u~0 -> 0, at u~1 -> 1."""
    c = MarshallOlkinCopula()
    c.set_parameters([pi1, pi2])
    assert math.isclose(float(c.partial_derivative_C_wrt_v(1e-10, v)), 0.0, abs_tol=1e-4)
    assert math.isclose(float(c.partial_derivative_C_wrt_v(1 - 1e-10, v)), 1.0, abs_tol=1e-4)


@given(pi1=valid_pi(), pi2=valid_pi(), u=unit_interval(), v1=unit_interval(), v2=unit_interval())
@settings(max_examples=50)
def test_h_function_monotone_in_v(pi1, pi2, u, v1, v2):
    """dC/du is monotone increasing in v (it is a CDF in v)."""
    if v1 > v2:
        v1, v2 = v2, v1
    c = MarshallOlkinCopula()
    c.set_parameters([pi1, pi2])
    assert c.partial_derivative_C_wrt_u(u, v1) <= c.partial_derivative_C_wrt_u(u, v2) + 1e-10


# ---------------------------------------------------------------------------
# Derivative cross-check
# ---------------------------------------------------------------------------

@given(pi1=valid_pi(), pi2=valid_pi(), u=unit_interval(eps=0.05), v=unit_interval(eps=0.05))
@settings(max_examples=80, deadline=None)
def test_partial_derivative_matches_finite_diff(pi1, pi2, u, v):
    """Away from the singular curve, dC/du should match finite differences."""
    assume(_away_from_singular(u, v, pi1, pi2))
    c = MarshallOlkinCopula()
    c.set_parameters([pi1, pi2])
    C = lambda a, b: float(c.get_cdf(a, b))
    h_num = _finite_diff(C, u, v, h=1e-5)
    h_ana = float(c.partial_derivative_C_wrt_u(u, v))
    assert math.isclose(h_ana, h_num, rel_tol=2e-2, abs_tol=2e-2)


# ---------------------------------------------------------------------------
# Kendall's tau
# ---------------------------------------------------------------------------

@given(pi1=valid_pi(), pi2=valid_pi())
def test_kendall_tau_formula(pi1, pi2):
    """tau = pi1*pi2 / (pi1 + pi2 - pi1*pi2) for Marshall-Olkin."""
    c = MarshallOlkinCopula()
    c.set_parameters([pi1, pi2])
    assert math.isclose(float(c.kendall_tau()), _tau_theoretical(pi1, pi2), rel_tol=1e-12, abs_tol=1e-12)


@given(pi1=valid_pi(), pi2=valid_pi())
def test_kendall_tau_range(pi1, pi2):
    """tau must lie in (0, 1) for any valid parameters."""
    c = MarshallOlkinCopula()
    c.set_parameters([pi1, pi2])
    tau = float(c.kendall_tau())
    assert 0.0 < tau < 1.0


def test_kendall_tau_monotone_in_pi():
    """In the symmetric case pi1=pi2=pi, tau = pi/(2-pi) increases with pi."""
    pis = [0.1, 0.3, 0.5, 0.7, 0.9]
    taus = []
    for pi in pis:
        c = MarshallOlkinCopula()
        c.set_parameters([pi, pi])
        taus.append(float(c.kendall_tau()))
    for i in range(len(taus) - 1):
        assert taus[i] < taus[i + 1]


@pytest.mark.slow
@given(pi1=valid_pi(), pi2=valid_pi())
@settings(max_examples=10, deadline=None)
def test_kendall_tau_vs_empirical(pi1, pi2):
    """Empirical Kendall tau from samples should be close to theoretical."""
    c = MarshallOlkinCopula()
    c.set_parameters([pi1, pi2])
    data = c.sample(6_000, rng=np.random.default_rng(0))
    tau_emp = float(kendalltau(data[:, 0], data[:, 1]).correlation)
    assert math.isfinite(tau_emp)
    assert abs(tau_emp - _tau_theoretical(pi1, pi2)) < 0.06


# ---------------------------------------------------------------------------
# Tail dependence
# ---------------------------------------------------------------------------

@given(pi1=valid_pi(), pi2=valid_pi())
def test_tail_dependence_formulas(pi1, pi2):
    """lambda_L = 0, lambda_U = min(pi1, pi2)."""
    c = MarshallOlkinCopula()
    c.set_parameters([pi1, pi2])
    assert float(c.LTDC()) == 0.0
    assert math.isclose(float(c.UTDC()), min(pi1, pi2), rel_tol=0, abs_tol=0)


@given(pi1=valid_pi(), pi2=valid_pi())
def test_upper_tail_dependence_positive(pi1, pi2):
    """lambda_U = min(pi1, pi2) > 0 for any valid parameters."""
    c = MarshallOlkinCopula()
    c.set_parameters([pi1, pi2])
    assert float(c.UTDC()) > 0.0


def test_upper_tail_dependence_increases_with_pi():
    """In the symmetric case, lambda_U = pi increases with pi."""
    pis = [0.1, 0.3, 0.5, 0.7, 0.9]
    utdcs = []
    for pi in pis:
        c = MarshallOlkinCopula()
        c.set_parameters([pi, pi])
        utdcs.append(float(c.UTDC()))
    for i in range(len(utdcs) - 1):
        assert utdcs[i] < utdcs[i + 1]


# ---------------------------------------------------------------------------
# Blomqvist beta
# ---------------------------------------------------------------------------

@given(pi1=valid_pi(), pi2=valid_pi())
def test_blomqvist_beta_matches_closed_form(pi1, pi2):
    c = MarshallOlkinCopula()
    c.set_parameters([pi1, pi2])

    beta = float(c.blomqvist_beta())
    beta_cf = 2.0 ** (min(pi1, pi2)) - 1.0

    assert math.isfinite(beta)
    assert -1.0 <= beta <= 1.0
    assert math.isclose(beta, beta_cf, rel_tol=1e-12, abs_tol=1e-12)

@given(pi1=valid_pi(), pi2=valid_pi())
@settings(max_examples=30, deadline=None)
def test_blomqvist_beta_consistent_with_definition(pi1, pi2):
    c = MarshallOlkinCopula()
    c.set_parameters([pi1, pi2])

    beta = float(c.blomqvist_beta())
    beta_def = 4.0 * float(c.get_cdf(0.5, 0.5)) - 1.0

    assert math.isfinite(beta_def)
    assert math.isclose(beta, beta_def, rel_tol=1e-12, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# init_from_data round-trip (slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@given(pi1=valid_pi(), pi2=valid_pi())
@settings(max_examples=10, deadline=None)
def test_init_from_data_roundtrip(pi1, pi2):
    """Generate samples with known (pi1, pi2), verify init_from_data returns valid params."""
    c = MarshallOlkinCopula()
    c.set_parameters([pi1, pi2])
    data = c.sample(8_000, rng=np.random.default_rng(123))

    c2 = MarshallOlkinCopula()
    res = c2.init_from_data(data[:, 0], data[:, 1])

    if isinstance(res, (list, tuple, np.ndarray)) and np.asarray(res).size == 2:
        c2.set_parameters(np.asarray(res, dtype=float))

    p = c2.get_parameters()
    assert 0.0 < float(p[0]) < 1.0
    assert 0.0 < float(p[1]) < 1.0


# ---------------------------------------------------------------------------
# Sampling sanity check (slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@given(pi1=valid_pi(), pi2=valid_pi())
@settings(max_examples=10, deadline=None)
def test_sampling_empirical_tau_close(pi1, pi2):
    """Empirical Kendall tau from samples should be close to theoretical."""
    c = MarshallOlkinCopula()
    c.set_parameters([pi1, pi2])
    data = c.sample(6_000, rng=np.random.default_rng(0))
    tau_emp = float(kendalltau(data[:, 0], data[:, 1]).correlation)
    assert math.isfinite(tau_emp)
    assert abs(tau_emp - _tau_theoretical(pi1, pi2)) < 0.06


# ---------------------------------------------------------------------------
# Shape checks (vectorised input)
# ---------------------------------------------------------------------------

def test_vectorised_shapes():
    """CDF, PDF, and sample must return arrays with correct shapes."""
    c = MarshallOlkinCopula()
    c.set_parameters([0.4, 0.7])
    u = np.linspace(0.05, 0.95, 13)
    v = np.linspace(0.05, 0.95, 13)
    assert c.get_cdf(u, v).shape == (13,)
    assert c.get_pdf(u, v).shape == (13,)
    assert c.sample(256).shape == (256, 2)


def test_vectorised_inputs_are_pairwise_not_grid():
    """Vectorized get_cdf/get_pdf operate pairwise on (u[i], v[i]), not on the Cartesian grid."""
    c = MarshallOlkinCopula()
    c.set_parameters([0.4, 0.7])
    u = np.array([0.2, 0.8])
    v = np.array([0.3, 0.7])

    cdf_vec = c.get_cdf(u, v)
    cdf_pair = np.array([
        c.get_cdf(float(u[0]), float(v[0])),
        c.get_cdf(float(u[1]), float(v[1])),
    ])

    assert cdf_vec.shape == (2,)
    assert np.allclose(cdf_vec, cdf_pair, rtol=1e-10, atol=1e-12)