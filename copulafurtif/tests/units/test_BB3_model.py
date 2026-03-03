import math

import numpy as np
import pytest
import scipy.stats as stx
from hypothesis import given, settings, strategies as st, assume
from scipy.stats import qmc

from CopulaFurtif.core.copulas.domain.models.archimedean.BB3 import BB3Copula


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def unit_interval(eps: float = 1e-3):
    return st.floats(
        min_value=eps, max_value=1.0 - eps,
        allow_nan=False, allow_infinity=False,
        allow_subnormal=False, width=64
    )

def _clip01(x: float, eps: float = 1e-12) -> float:
    return float(min(max(x, eps), 1.0 - eps))

def _clipped_call(f, x, y, eps: float = 1e-12):
    return float(f(_clip01(x, eps), _clip01(y, eps)))

def _finite_diff(f, x, y, h: float = 2e-4, eps: float = 1e-12):
    return (_clipped_call(f, x + h, y, eps) - _clipped_call(f, x - h, y, eps)) / (2.0 * h)

def _mixed_finite_diff(C, u, v, h: float = 2e-4, eps: float = 1e-12):
    return (
        _clipped_call(C, u + h, v + h, eps)
        - _clipped_call(C, u + h, v - h, eps)
        - _clipped_call(C, u - h, v + h, eps)
        + _clipped_call(C, u - h, v - h, eps)
    ) / (4.0 * h * h)

def _mixed_finite_diff_richardson(C, u, v, h=1e-4, eps=1e-12):
    """
    Richardson extrapolation:
      D(h)  = mixed FD with step h  (O(h^2))
      D(h/2)= mixed FD with step h/2
      => D* = (4*D(h/2) - D(h))/3  (O(h^4))
    """
    D1 = _mixed_finite_diff(C, u, v, h=h, eps=eps)
    D2 = _mixed_finite_diff(C, u, v, h=h/2, eps=eps)
    return (4.0 * D2 - D1) / 3.0


def _make(theta: float, delta: float) -> BB3Copula:
    c = BB3Copula()
    c.set_parameters([float(theta), float(delta)])
    return c


_BOUNDS = BB3Copula().get_bounds()
THETA_LO, THETA_HI = map(float, _BOUNDS[0])
DELTA_LO, DELTA_HI = map(float, _BOUNDS[1])

THETA_HI_CAP = THETA_HI if math.isfinite(THETA_HI) else 20.0
DELTA_HI_CAP = DELTA_HI if math.isfinite(DELTA_HI) else 20.0


@st.composite
def valid_theta(draw):
    return draw(st.floats(
        min_value=THETA_LO, max_value=min(THETA_HI_CAP, 10.0),
        exclude_min=True, exclude_max=True,
        allow_nan=False, allow_infinity=False,
        allow_subnormal=False, width=64
    ))

@st.composite
def valid_delta(draw):
    return draw(st.floats(
        min_value=DELTA_LO, max_value=min(DELTA_HI_CAP, 10.0),
        exclude_min=True, exclude_max=True,
        allow_nan=False, allow_infinity=False,
        allow_subnormal=False, width=64
    ))

@st.composite
def valid_params(draw):
    return float(draw(valid_theta())), float(draw(valid_delta()))


@pytest.fixture(scope="module")
def copula_default():
    th = max(THETA_LO + 0.5, min(2.0, min(THETA_HI_CAP, 10.0) - 0.5))
    de = max(DELTA_LO + 0.5, min(1.5, min(DELTA_HI_CAP, 10.0) - 0.5))
    return _make(th, de)


# ---------------------------------------------------------------------
# 1) PARAMETERS
# ---------------------------------------------------------------------

@given(theta=valid_theta(), delta=valid_delta())
def test_parameter_roundtrip(theta, delta):
    c = _make(theta, delta)
    p = c.get_parameters()
    assert math.isclose(p[0], theta, rel_tol=1e-12)
    assert math.isclose(p[1], delta, rel_tol=1e-12)


@pytest.mark.parametrize("bad", [[], [2.0], [2.0, 1.0, 3.0]])
def test_parameter_wrong_size(bad):
    c = BB3Copula()
    with pytest.raises(ValueError):
        c.set_parameters(bad)


@given(theta=st.floats(max_value=THETA_LO, allow_nan=False, allow_infinity=False),
       delta=valid_delta())
def test_theta_out_of_bounds(theta, delta):
    c = BB3Copula()
    with pytest.raises(ValueError):
        c.set_parameters([float(theta), float(delta)])


@given(theta=valid_theta(),
       delta=st.floats(max_value=DELTA_LO, allow_nan=False, allow_infinity=False))
def test_delta_out_of_bounds(theta, delta):
    c = BB3Copula()
    with pytest.raises(ValueError):
        c.set_parameters([float(theta), float(delta)])


# ---------------------------------------------------------------------
# 2) CDF INVARIANTS + BOUNDARIES
# ---------------------------------------------------------------------

@given(data=valid_params(), u=unit_interval(), v=unit_interval())
@settings(max_examples=80, deadline=None)
def test_cdf_bounds(data, u, v):
    theta, delta = data
    c = _make(theta, delta)
    val = float(c.get_cdf(u, v))
    assert 0.0 <= val <= 1.0


@given(data=valid_params(), u1=unit_interval(), u2=unit_interval(), v=unit_interval())
@settings(max_examples=80, deadline=None)
def test_cdf_monotone_in_u(data, u1, u2, v):
    theta, delta = data
    if u1 > u2:
        u1, u2 = u2, u1
    c = _make(theta, delta)
    assert float(c.get_cdf(u1, v)) <= float(c.get_cdf(u2, v)) + 1e-12


@given(data=valid_params(), u=unit_interval(), v=unit_interval())
@settings(max_examples=80, deadline=None)
def test_cdf_symmetry(data, u, v):
    theta, delta = data
    c = _make(theta, delta)
    assert math.isclose(float(c.get_cdf(u, v)), float(c.get_cdf(v, u)), rel_tol=1e-10, abs_tol=1e-12)


@given(data=valid_params(), u=unit_interval(1e-2))
def test_cdf_boundary_v_zero(data, u):
    theta, delta = data
    c = _make(theta, delta)
    assert math.isclose(float(c.get_cdf(u, 0.0)), 0.0, abs_tol=1e-12)

@given(data=valid_params(), v=unit_interval(1e-2))
def test_cdf_boundary_u_zero(data, v):
    theta, delta = data
    c = _make(theta, delta)
    assert math.isclose(float(c.get_cdf(0.0, v)), 0.0, abs_tol=1e-12)

@given(data=valid_params(), u=unit_interval(1e-2))
def test_cdf_boundary_v_one(data, u):
    theta, delta = data
    c = _make(theta, delta)
    assert math.isclose(float(c.get_cdf(u, 1.0)), float(u), rel_tol=1e-10, abs_tol=1e-10)

@given(data=valid_params(), v=unit_interval(1e-2))
def test_cdf_boundary_u_one(data, v):
    theta, delta = data
    c = _make(theta, delta)
    assert math.isclose(float(c.get_cdf(1.0, v)), float(v), rel_tol=1e-10, abs_tol=1e-10)


# ---------------------------------------------------------------------
# 3) PDF INVARIANTS + FD CHECKS
# ---------------------------------------------------------------------

@given(data=valid_params(), u=unit_interval(5e-2), v=unit_interval(5e-2))
@settings(max_examples=80, deadline=None)
def test_pdf_nonnegative(data, u, v):
    theta, delta = data
    c = _make(theta, delta)
    pdf = float(c.get_pdf(u, v))
    assert math.isfinite(pdf)
    assert pdf >= 0.0


@given(data=valid_params(), u=unit_interval(2e-1), v=unit_interval(2e-1))
@settings(max_examples=60, deadline=None)
def test_pdf_matches_mixed_derivative(data, u, v):
    theta, delta = data
    c = _make(theta, delta)

    # FD is fragile in extreme tails; keep params moderate already via caps
    C = lambda a, b: float(c.get_cdf(a, b))
    pdf_num = _mixed_finite_diff_richardson(C, u, v, h=2e-4)
    pdf_ana = float(c.get_pdf(u, v))

    assert math.isclose(pdf_ana, pdf_num, rel_tol=0.15, abs_tol=0.15)

@pytest.mark.parametrize("theta,delta", [(2.0, 1.5), (3.0, 2.0), (5.0, 3.0)])
def test_pdf_integrates_matches_rectangle_probability(theta, delta):
    c = _make(theta, delta)

    eps = 0.01
    a = eps
    b = 1.0 - eps
    area = (b - a) ** 2

    # Rectangle probability via CDF (exact identity)
    p_rect = (
        float(c.get_cdf(b, b))
        - float(c.get_cdf(a, b))
        - float(c.get_cdf(b, a))
        + float(c.get_cdf(a, a))
    )

    # QMC estimate of ∫∫_rect c(u,v) du dv
    m = 2**18
    engine = qmc.Sobol(d=2, scramble=True, seed=0)  # IMPORTANT
    pts = engine.random_base2(int(np.log2(m)))

    u = a + (b - a) * pts[:, 0]
    v = a + (b - a) * pts[:, 1]

    est = float(np.mean(c.get_pdf(u, v))) * area

    assert math.isclose(est, p_rect, rel_tol=5e-2, abs_tol=5e-2)


# ---------------------------------------------------------------------
# 4) H-FUNCTIONS
# ---------------------------------------------------------------------

@given(data=valid_params(), u=unit_interval(5e-2), v=unit_interval(5e-2))
@settings(max_examples=80, deadline=None)
def test_h_functions_are_probabilities(data, u, v):
    theta, delta = data
    c = _make(theta, delta)
    h1 = float(c.partial_derivative_C_wrt_u(u, v))
    h2 = float(c.partial_derivative_C_wrt_v(u, v))
    assert -1e-12 <= h1 <= 1.0 + 1e-12
    assert -1e-12 <= h2 <= 1.0 + 1e-12


@given(data=valid_params(), u=unit_interval(1e-2))
def test_h_u_boundary_in_v(data, u):
    theta, delta = data
    c = _make(theta, delta)
    assert math.isclose(float(c.partial_derivative_C_wrt_u(u, 0.0)), 0.0, abs_tol=1e-12)
    assert math.isclose(float(c.partial_derivative_C_wrt_u(u, 1.0)), 1.0, abs_tol=1e-12)


@given(data=valid_params(), v=unit_interval(1e-2))
def test_h_v_boundary_in_u(data, v):
    theta, delta = data
    c = _make(theta, delta)
    assert math.isclose(float(c.partial_derivative_C_wrt_v(0.0, v)), 0.0, abs_tol=1e-12)
    assert math.isclose(float(c.partial_derivative_C_wrt_v(1.0, v)), 1.0, abs_tol=1e-12)


@given(data=valid_params(), u=unit_interval(5e-2), v=unit_interval(5e-2))
@settings(max_examples=80, deadline=None)
def test_h_functions_cross_symmetry(data, u, v):
    theta, delta = data
    c = _make(theta, delta)
    assert math.isclose(
        float(c.partial_derivative_C_wrt_u(u, v)),
        float(c.partial_derivative_C_wrt_v(v, u)),
        rel_tol=1e-10, abs_tol=1e-12
    )


# ---------------------------------------------------------------------
# 5) DERIVATIVES (FD)
# ---------------------------------------------------------------------

@given(data=valid_params(), u=unit_interval(5e-2), v=unit_interval(5e-2))
@settings(max_examples=60, deadline=None)
def test_partial_derivative_matches_finite_diff(data, u, v):
    theta, delta = data
    c = _make(theta, delta)
    C = lambda a, b: float(c.get_cdf(a, b))

    num_du = _finite_diff(C, u, v, h=2e-4)
    num_dv = _finite_diff(lambda x, y: C(y, x), v, u, h=2e-4)

    ana_du = float(c.partial_derivative_C_wrt_u(u, v))
    ana_dv = float(c.partial_derivative_C_wrt_v(u, v))

    assert math.isclose(ana_du, num_du, rel_tol=0.08, abs_tol=0.05)
    assert math.isclose(ana_dv, num_dv, rel_tol=0.08, abs_tol=0.05)


# ---------------------------------------------------------------------
# 6) TAIL DEPENDENCE
# ---------------------------------------------------------------------

@given(data=valid_params())
@settings(max_examples=60, deadline=None)
def test_tail_dependence_formulas(data):
    theta, delta = data
    c = _make(theta, delta)

    assert math.isclose(float(c.LTDC()), 1.0, abs_tol=0.0)  # theta>1 in this arch
    expected_ut = 2.0 - 2.0 ** (1.0 / theta)
    assert math.isclose(float(c.UTDC()), expected_ut, rel_tol=1e-12, abs_tol=1e-12)


# ---------------------------------------------------------------------
# 7) BLOMQVIST
# ---------------------------------------------------------------------

@given(data=valid_params())
@settings(max_examples=60, deadline=None)
def test_blomqvist_beta_matches_definition(data):
    theta, delta = data
    c = _make(theta, delta)
    beta_def = 4.0 * float(c.get_cdf(0.5, 0.5)) - 1.0
    assert math.isclose(float(c.blomqvist_beta()), beta_def, rel_tol=1e-12, abs_tol=1e-12)


# ---------------------------------------------------------------------
# 8) KENDALL TAU
# ---------------------------------------------------------------------

@given(data=valid_params())
@settings(max_examples=40, deadline=None)
def test_kendall_tau_range(data):
    theta, delta = data
    c = _make(theta, delta)
    tau = float(c.kendall_tau(n_quad=100))
    assert -1.0 <= tau <= 1.0


@pytest.mark.slow
@pytest.mark.parametrize("theta,delta", [(2.0, 1.5), (3.0, 2.0)])
def test_empirical_kendall_tau_close(theta, delta):
    c = _make(theta, delta)
    data = c.sample(6000, rng=np.random.default_rng(0))
    tau_emp, _ = stx.kendalltau(data[:, 0], data[:, 1])
    tau_theo = float(c.kendall_tau(n_quad=160))

    n = len(data)
    sigma0 = math.sqrt(2 * (2 * n + 5) / (9 * n * (n - 1)))
    assert math.isclose(float(tau_emp), tau_theo, abs_tol=4 * sigma0 + 0.05)


# ---------------------------------------------------------------------
# 9) INIT FROM DATA (slow)
# ---------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("theta,delta", [(2.0, 1.5), (3.0, 2.0)])
def test_init_from_data_roundtrip(theta, delta):
    c_true = _make(theta, delta)
    data = c_true.sample(6000, rng=np.random.default_rng(0))

    c_fit = BB3Copula()
    c_fit.init_from_data(data[:, 0], data[:, 1])
    th_hat, de_hat = map(float, c_fit.get_parameters())

    assert abs(th_hat - theta) / max(theta, 1e-6) < 0.6
    assert abs(de_hat - delta) / max(delta, 1e-6) < 0.6


# ---------------------------------------------------------------------
# 10) IAD / AD disabled
# ---------------------------------------------------------------------

def test_iad_ad_disabled(copula_default):
    assert np.isnan(copula_default.IAD(None))
    assert np.isnan(copula_default.AD(None))


# ---------------------------------------------------------------------
# 11) Vectorisation
# ---------------------------------------------------------------------

def test_vectorised_shapes(copula_default):
    u = np.linspace(0.05, 0.95, 11)
    v = np.linspace(0.05, 0.95, 11)
    assert np.asarray(copula_default.get_cdf(u, v)).shape == (11,)
    assert np.asarray(copula_default.get_pdf(u, v)).shape == (11,)
    s = copula_default.sample(256, rng=np.random.default_rng(0))
    assert s.shape == (256, 2)


def test_vectorised_inputs_are_pairwise_not_grid(copula_default):
    u = np.array([0.2, 0.8])
    v = np.array([0.3, 0.7])
    cdf_vec = np.asarray(copula_default.get_cdf(u, v))
    cdf_pair = np.array([
        float(copula_default.get_cdf(float(u[0]), float(v[0]))),
        float(copula_default.get_cdf(float(u[1]), float(v[1]))),
    ])
    assert cdf_vec.shape == (2,)
    assert np.allclose(cdf_vec, cdf_pair, rtol=1e-8, atol=1e-10)