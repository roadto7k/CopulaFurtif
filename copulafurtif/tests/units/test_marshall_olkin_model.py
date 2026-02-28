import math
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, assume
from scipy.stats import kendalltau

from CopulaFurtif.core.copulas.domain.models.exotic.marshall_olkin import MarshallOlkinCopula


# ============================================================
# Helpers
# ============================================================

@st.composite
def valid_pi(draw):
    """pi in (0,1) strictly; avoid extremes for numerical stability in tests."""
    return draw(st.floats(
        min_value=0.02, max_value=0.98,
        exclude_min=True, exclude_max=True,
        allow_nan=False, allow_infinity=False,
        allow_subnormal=False,
        width=64,
    ))

@st.composite
def unit_interval(draw, eps=1e-3):
    """u in (eps, 1-eps) strictly."""
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
    """Central FD ∂f/∂x with clipping to (0,1)."""
    return (_clipped_f(f, x + h, y, eps) - _clipped_f(f, x - h, y, eps)) / (2.0 * h)

def _mixed_finite_diff(C, u, v, h=1e-5, eps=1e-12):
    """Central FD ∂²C/∂u∂v with clipping to (0,1)."""
    return (
        _clipped_f(C, u + h, v + h, eps)
        - _clipped_f(C, u + h, v - h, eps)
        - _clipped_f(C, u - h, v + h, eps)
        + _clipped_f(C, u - h, v - h, eps)
    ) / (4.0 * h * h)

def _away_from_singular_curve(u, v, pi1, pi2, tol=2e-3):
    """
    Singular set is u^pi1 == v^pi2 (where the copula has a singular component).
    Derivatives are not well-behaved exactly on that curve.
    """
    return abs((u ** pi1) - (v ** pi2)) > tol

def _tau_theoretical(pi1, pi2):
    # Book: τ = pi1*pi2 / (pi1 + pi2 - pi1*pi2)
    den = pi1 + pi2 - pi1 * pi2
    return (pi1 * pi2) / den

def _beta_theoretical(pi1, pi2):
    # Book: β = 2^{min(pi1,pi2)} - 1
    return 2.0 ** (min(pi1, pi2)) - 1.0

def _mass_ac_theoretical(pi1, pi2):
    # Continuous mass = 1 - singular mass, and singular mass = τ (book gives same fraction)
    return 1.0 - _tau_theoretical(pi1, pi2)


# ============================================================
# Parameter handling
# ============================================================

def test_parameter_wrong_size():
    c = MarshallOlkinCopula()
    with pytest.raises(ValueError):
        c.set_parameters([0.5])
    with pytest.raises(ValueError):
        c.set_parameters([0.3, 0.4, 0.5])

@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.1])
def test_parameter_at_boundary_rejected(bad):
    c = MarshallOlkinCopula()
    with pytest.raises(ValueError):
        c.set_parameters([bad, 0.5])
    with pytest.raises(ValueError):
        c.set_parameters([0.5, bad])

@given(pi1=valid_pi(), pi2=valid_pi())
def test_parameter_roundtrip(pi1, pi2):
    c = MarshallOlkinCopula()
    c.set_parameters([pi1, pi2])
    p = c.get_parameters()
    assert math.isclose(float(p[0]), pi1, rel_tol=0, abs_tol=0)
    assert math.isclose(float(p[1]), pi2, rel_tol=0, abs_tol=0)


# ============================================================
# CDF properties
# ============================================================

@given(pi1=valid_pi(), pi2=valid_pi(), u=unit_interval(), v=unit_interval())
def test_cdf_bounds(pi1, pi2, u, v):
    c = MarshallOlkinCopula()
    c.set_parameters([pi1, pi2])
    val = float(c.get_cdf(u, v))
    assert 0.0 <= val <= 1.0

@given(pi1=valid_pi(), pi2=valid_pi(), u=unit_interval(), v=unit_interval())
def test_cdf_monotone_in_u(pi1, pi2, u, v):
    c = MarshallOlkinCopula()
    c.set_parameters([pi1, pi2])
    u2 = min(u + 1e-3, 0.999)
    assert c.get_cdf(u, v) <= c.get_cdf(u2, v) + 1e-12

def test_cdf_boundaries():
    c = MarshallOlkinCopula()
    c.set_parameters([0.4, 0.7])
    grid = np.linspace(0.0, 1.0, 21)
    for u in grid:
        assert math.isclose(float(c.get_cdf(u, 0.0)), 0.0, abs_tol=1e-12)
        assert math.isclose(float(c.get_cdf(0.0, u)), 0.0, abs_tol=1e-12)
        assert math.isclose(float(c.get_cdf(u, 1.0)), u, abs_tol=1e-8)
        assert math.isclose(float(c.get_cdf(1.0, u)), u, abs_tol=1e-8)

@given(pi1=valid_pi(), pi2=valid_pi(), u=unit_interval(), v=unit_interval())
def test_cdf_swap_parameter_symmetry(pi1, pi2, u, v):
    """
    Book-consistent asymmetry property:
      C(u,v;pi1,pi2) == C(v,u;pi2,pi1)
    """
    c1 = MarshallOlkinCopula()
    c2 = MarshallOlkinCopula()
    c1.set_parameters([pi1, pi2])
    c2.set_parameters([pi2, pi1])
    assert math.isclose(float(c1.get_cdf(u, v)), float(c2.get_cdf(v, u)), rel_tol=1e-12, abs_tol=1e-12)


# ============================================================
# h-functions (partials of CDF)
# ============================================================

@given(pi1=valid_pi(), pi2=valid_pi(), u=unit_interval(), v=unit_interval())
def test_h_functions_are_probabilities(pi1, pi2, u, v):
    c = MarshallOlkinCopula()
    c.set_parameters([pi1, pi2])
    h1 = float(c.partial_derivative_C_wrt_u(u, v))
    h2 = float(c.partial_derivative_C_wrt_v(u, v))
    eps = 1e-12
    assert -eps <= h1 <= 1.0 + eps
    assert -eps <= h2 <= 1.0 + eps

@given(pi1=valid_pi(), pi2=valid_pi(), u=unit_interval(eps=0.05), v=unit_interval(eps=0.05))
@settings(max_examples=80, deadline=None)
def test_partial_derivative_matches_finite_diff(pi1, pi2, u, v):
    """
    Away from the singular curve, ∂C/∂u should match FD.
    """
    assume(_away_from_singular_curve(u, v, pi1, pi2, tol=5e-3))

    c = MarshallOlkinCopula()
    c.set_parameters([pi1, pi2])
    C = lambda a, b: float(c.get_cdf(a, b))
    h_num = _finite_diff(C, u, v, h=1e-5)
    h_ana = float(c.partial_derivative_C_wrt_u(u, v))
    assert math.isclose(h_ana, h_num, rel_tol=2e-2, abs_tol=2e-2)


# ============================================================
# PDF (absolutely-continuous part only)
# ============================================================

@given(pi1=valid_pi(), pi2=valid_pi(), u=unit_interval(), v=unit_interval())
def test_pdf_nonnegative(pi1, pi2, u, v):
    c = MarshallOlkinCopula()
    c.set_parameters([pi1, pi2])
    pdf = float(c.get_pdf(u, v))
    assert pdf >= 0.0

@given(pi1=valid_pi(), pi2=valid_pi(), u=unit_interval(eps=0.05), v=unit_interval(eps=0.05))
@settings(max_examples=60, deadline=None)
def test_pdf_matches_mixed_derivative_off_singular(pi1, pi2, u, v):
    """
    Off the singular curve, the AC density matches ∂²C/∂u∂v.
    (On the curve, there is singular mass, so this identity breaks.)
    """
    assume(_away_from_singular_curve(u, v, pi1, pi2, tol=8e-3))

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
    Because Marshall–Olkin has a singular component, the Lebesgue density integrates
    to the absolutely-continuous mass only, not to 1.

    AC mass = 1 - singular mass, and the book gives singular mass = pi1*pi2/(pi1+pi2-pi1*pi2).
    """
    pi1, pi2 = 0.35, 0.7
    c = MarshallOlkinCopula()
    c.set_parameters([pi1, pi2])

    # deterministic grid midpoints (stable, lower variance than pure MC)
    n = 300  # 90k points
    xs = (np.arange(n) + 0.5) / n
    U, V = np.meshgrid(xs, xs, indexing="ij")
    pdf = c.get_pdf(U.ravel(), V.ravel())
    est = float(np.mean(pdf))  # area = 1

    target = _mass_ac_theoretical(pi1, pi2)
    assert math.isclose(est, target, rel_tol=3e-2, abs_tol=3e-2), (est, target)


# ============================================================
# Closed-form dependence measures
# ============================================================

@given(pi1=valid_pi(), pi2=valid_pi())
def test_kendall_tau_formula(pi1, pi2):
    c = MarshallOlkinCopula()
    c.set_parameters([pi1, pi2])
    tau = float(c.kendall_tau())
    tau_th = _tau_theoretical(pi1, pi2)
    assert math.isclose(tau, tau_th, rel_tol=1e-12, abs_tol=1e-12)

@given(pi1=valid_pi(), pi2=valid_pi())
def test_blomqvist_beta_formula(pi1, pi2):
    c = MarshallOlkinCopula()
    c.set_parameters([pi1, pi2])

    # if method exists, use it; else compare via C(1/2,1/2)
    beta_th = _beta_theoretical(pi1, pi2)
    if hasattr(c, "blomqvist_beta"):
        beta = float(c.blomqvist_beta())
        assert math.isclose(beta, beta_th, rel_tol=1e-12, abs_tol=1e-12)
    else:
        beta = 4.0 * float(c.get_cdf(0.5, 0.5)) - 1.0
        assert math.isclose(beta, beta_th, rel_tol=1e-12, abs_tol=1e-12)

@given(pi1=valid_pi(), pi2=valid_pi())
def test_tail_dependence(pi1, pi2):
    c = MarshallOlkinCopula()
    c.set_parameters([pi1, pi2])
    assert float(c.LTDC()) == 0.0
    assert math.isclose(float(c.UTDC()), min(pi1, pi2), rel_tol=0, abs_tol=0)


# ============================================================
# init_from_data + sampling sanity
# ============================================================

@pytest.mark.slow
@given(pi1=valid_pi(), pi2=valid_pi())
@settings(max_examples=10, deadline=None)
def test_init_from_data_roundtrip(pi1, pi2):
    """
    This init is usually conservative (often symmetric), so we only check it
    returns valid parameters and doesn't explode. If your init is stronger,
    tighten this check.
    """
    c = MarshallOlkinCopula()
    c.set_parameters([pi1, pi2])
    data = c.sample(8000, rng=np.random.default_rng(123))

    c2 = MarshallOlkinCopula()
    res = c2.init_from_data(data[:, 0], data[:, 1])

    # Support both styles:
    # - init_from_data sets parameters in-place and returns None
    # - init_from_data returns an array of params
    if isinstance(res, (list, tuple, np.ndarray)) and np.asarray(res).size == 2:
        c2.set_parameters(np.asarray(res, dtype=float))

    p = c2.get_parameters()
    assert 0.0 < float(p[0]) < 1.0
    assert 0.0 < float(p[1]) < 1.0

@pytest.mark.slow
@given(pi1=valid_pi(), pi2=valid_pi())
@settings(max_examples=10, deadline=None)
def test_sampling_empirical_tau_close(pi1, pi2):
    c = MarshallOlkinCopula()
    c.set_parameters([pi1, pi2])
    data = c.sample(6000, rng=np.random.default_rng(0))
    tau_emp = float(kendalltau(data[:, 0], data[:, 1]).correlation)
    tau_th = _tau_theoretical(pi1, pi2)

    # generous tolerance because ties/singular mass slightly change finite-sample behavior
    assert math.isfinite(tau_emp)
    assert abs(tau_emp - tau_th) < 0.06


# ============================================================
# Vectorization contract
# ============================================================

def test_vectorised_inputs_are_pairwise_not_grid():
    c = MarshallOlkinCopula()
    c.set_parameters([0.4, 0.7])

    u = np.array([0.2, 0.8])
    v = np.array([0.3, 0.7])

    cdf_vec = c.get_cdf(u, v)
    cdf_pair = np.array([c.get_cdf(float(u[0]), float(v[0])),
                         c.get_cdf(float(u[1]), float(v[1]))])

    assert cdf_vec.shape == (2,)
    assert np.allclose(cdf_vec, cdf_pair, rtol=1e-10, atol=1e-12)