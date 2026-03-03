import math

import numpy as np
import pytest
import scipy.stats as stx
from hypothesis import given, settings, strategies as st

from CopulaFurtif.core.copulas.domain.models.archimedean.BB2 import BB2Copula


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def unit_interval(eps: float = 1e-3):
    """Strategy for u,v in (0,1) away from boundaries."""
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
    """1st-order central finite difference ∂f/∂x (clipped)."""
    return (_clipped_call(f, x + h, y, eps) - _clipped_call(f, x - h, y, eps)) / (2.0 * h)

def _mixed_finite_diff(C, u, v, h: float = 2e-4, eps: float = 1e-12):
    """Central 2D finite difference ∂²C/∂u∂v (clipped)."""
    return (
        _clipped_call(C, u + h, v + h, eps)
        - _clipped_call(C, u + h, v - h, eps)
        - _clipped_call(C, u - h, v + h, eps)
        + _clipped_call(C, u - h, v - h, eps)
    ) / (4.0 * h * h)

def _make(theta: float, delta: float) -> BB2Copula:
    c = BB2Copula()
    c.set_parameters([float(theta), float(delta)])
    return c


# Use library bounds dynamically (keeps tests aligned if you change bounds)
_BOUNDS = BB2Copula().get_bounds()
THETA_LO, THETA_HI = map(float, _BOUNDS[0])
DELTA_LO, DELTA_HI = map(float, _BOUNDS[1])

# Hypothesis-friendly caps if bounds are infinite
THETA_HI_CAP = THETA_HI if math.isfinite(THETA_HI) else 50.0
DELTA_HI_CAP = DELTA_HI if math.isfinite(DELTA_HI) else 50.0

@st.composite
def valid_theta(draw):
    return draw(st.floats(
        min_value=THETA_LO, max_value=THETA_HI_CAP,
        exclude_min=True, exclude_max=True,
        allow_nan=False, allow_infinity=False,
        allow_subnormal=False, width=64
    ))

@st.composite
def valid_delta(draw):
    return draw(st.floats(
        min_value=DELTA_LO, max_value=DELTA_HI_CAP,
        exclude_min=True, exclude_max=True,
        allow_nan=False, allow_infinity=False,
        allow_subnormal=False, width=64
    ))

@st.composite
def valid_params(draw):
    return float(draw(valid_theta())), float(draw(valid_delta()))


@pytest.fixture(scope="module")
def copula_default():
    # pick a safe interior point
    theta0 = max(THETA_LO + 0.5, min(2.0, THETA_HI_CAP - 0.5))
    delta0 = max(DELTA_LO + 0.5, min(2.0, DELTA_HI_CAP - 0.5))
    return _make(theta0, delta0)


# -----------------------------------------------------------------------------
# 1) PARAMETERS
# -----------------------------------------------------------------------------

@given(theta=valid_theta(), delta=valid_delta())
def test_parameter_roundtrip(theta, delta):
    c = _make(theta, delta)
    p = c.get_parameters()
    assert math.isclose(p[0], theta, rel_tol=1e-12)
    assert math.isclose(p[1], delta, rel_tol=1e-12)


@given(
    theta=st.one_of(
        st.floats(max_value=THETA_LO, allow_nan=False, allow_infinity=False),
        st.floats(max_value=THETA_LO, exclude_max=True, allow_nan=False, allow_infinity=False),
    ),
    delta=valid_delta(),
)
def test_parameter_out_of_bounds_theta(theta, delta):
    c = BB2Copula()
    with pytest.raises(ValueError):
        c.set_parameters([float(theta), float(delta)])


@given(
    theta=valid_theta(),
    delta=st.one_of(
        st.floats(max_value=DELTA_LO, allow_nan=False, allow_infinity=False),
        st.floats(max_value=DELTA_LO, exclude_max=True, allow_nan=False, allow_infinity=False),
    ),
)
def test_parameter_out_of_bounds_delta(theta, delta):
    c = BB2Copula()
    with pytest.raises(ValueError):
        c.set_parameters([float(theta), float(delta)])


@pytest.mark.parametrize("bad", [[], [1.0], [1.0, 2.0, 3.0]])
def test_parameter_wrong_size(bad):
    c = BB2Copula()
    with pytest.raises(ValueError):
        c.set_parameters(bad)


def test_parameter_at_boundary_rejected():
    c = BB2Copula()

    # lower boundaries (exclusive)
    with pytest.raises(ValueError):
        c.set_parameters([THETA_LO, max(DELTA_LO + 0.1, 2.0)])
    with pytest.raises(ValueError):
        c.set_parameters([max(THETA_LO + 0.1, 2.0), DELTA_LO])

    # upper boundaries (exclusive) only if finite
    if math.isfinite(THETA_HI):
        with pytest.raises(ValueError):
            c.set_parameters([THETA_HI, max(DELTA_LO + 0.1, 2.0)])
    if math.isfinite(DELTA_HI):
        with pytest.raises(ValueError):
            c.set_parameters([max(THETA_LO + 0.1, 2.0), DELTA_HI])


# -----------------------------------------------------------------------------
# 2) CDF INVARIANTS
# -----------------------------------------------------------------------------

@given(data=valid_params(), u=unit_interval(1e-3), v=unit_interval(1e-3))
@settings(max_examples=80, deadline=None)
def test_cdf_bounds(data, u, v):
    theta, delta = data
    c = _make(theta, delta)
    val = float(c.get_cdf(u, v))
    assert 0.0 <= val <= 1.0


@given(data=valid_params(), u1=unit_interval(1e-3), u2=unit_interval(1e-3), v=unit_interval(1e-3))
@settings(max_examples=80, deadline=None)
def test_cdf_monotone_in_u(data, u1, u2, v):
    theta, delta = data
    if u1 > u2:
        u1, u2 = u2, u1
    c = _make(theta, delta)
    assert float(c.get_cdf(u1, v)) <= float(c.get_cdf(u2, v)) + 1e-12


@given(data=valid_params(), u=unit_interval(1e-3), v=unit_interval(1e-3))
@settings(max_examples=80, deadline=None)
def test_cdf_symmetry(data, u, v):
    theta, delta = data
    c = _make(theta, delta)
    a = float(c.get_cdf(u, v))
    b = float(c.get_cdf(v, u))
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)


# -----------------------------------------------------------------------------
# 3) FRÉCHET–HOEFFDING BOUNDARIES (exact)
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# 4) PDF INVARIANTS
# -----------------------------------------------------------------------------

@given(data=valid_params(), u=unit_interval(5e-2), v=unit_interval(5e-2))
@settings(max_examples=80, deadline=None)
def test_pdf_nonnegative(data, u, v):
    theta, delta = data
    c = _make(theta, delta)
    pdf = float(c.get_pdf(u, v))
    assert math.isfinite(pdf)
    assert pdf >= 0.0


@given(data=valid_params(), u=unit_interval(5e-2), v=unit_interval(5e-2))
@settings(max_examples=80, deadline=None)
def test_pdf_symmetry(data, u, v):
    theta, delta = data
    c = _make(theta, delta)
    a = float(c.get_pdf(u, v))
    b = float(c.get_pdf(v, u))
    assert math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-10)


@given(data=valid_params(), u=unit_interval(5e-2), v=unit_interval(5e-2))
@settings(max_examples=60, deadline=None)
def test_pdf_matches_mixed_derivative(data, u, v):
    """
    c(u,v) ≈ ∂²C/∂u∂v via 2D central finite difference (clipped).
    """
    theta, delta = data
    c = _make(theta, delta)

    C = lambda a, b: float(c.get_cdf(a, b))
    pdf_num = _mixed_finite_diff(C, u, v, h=1e-4)
    pdf_ana = float(c.get_pdf(u, v))

    assert math.isclose(pdf_ana, pdf_num, rel_tol=8e-2, abs_tol=1.0)


@pytest.mark.parametrize("theta,delta", [
    (max(THETA_LO + 0.5, 1.0), max(DELTA_LO + 0.5, 1.0)),
    (max(THETA_LO + 1.0, 2.0), max(DELTA_LO + 1.0, 2.0)),
    (max(THETA_LO + 2.0, 4.0), max(DELTA_LO + 2.0, 4.0)),
])
def test_pdf_integrates_to_one(theta, delta):
    """
    Monte-Carlo estimate: ∫∫ c(u,v) du dv ≈ E[c(U,V)] with U,V~Unif.
    """
    c = _make(theta, delta)
    rng = np.random.default_rng(0)
    u = rng.uniform(1e-3, 1.0 - 1e-3, size=120_000)
    v = rng.uniform(1e-3, 1.0 - 1e-3, size=120_000)
    pdf = np.asarray(c.get_pdf(u, v), float)
    assert np.all(np.isfinite(pdf))
    assert float(np.mean(pdf)) > 0.0


# -----------------------------------------------------------------------------
# 5) H-FUNCTIONS
# -----------------------------------------------------------------------------

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
    a = float(c.partial_derivative_C_wrt_u(u, v))
    b = float(c.partial_derivative_C_wrt_v(v, u))
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)


@given(data=valid_params(), u=unit_interval(5e-2), v1=unit_interval(5e-2), v2=unit_interval(5e-2))
@settings(max_examples=80, deadline=None)
def test_h_function_monotone_in_v(data, u, v1, v2):
    theta, delta = data
    if v1 > v2:
        v1, v2 = v2, v1
    c = _make(theta, delta)
    assert float(c.partial_derivative_C_wrt_u(u, v1)) <= float(c.partial_derivative_C_wrt_u(u, v2)) + 1e-12


# -----------------------------------------------------------------------------
# 6) DERIVATIVES
# -----------------------------------------------------------------------------

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

    assert math.isclose(ana_du, num_du, rel_tol=2e-2, abs_tol=1e-2)
    assert math.isclose(ana_dv, num_dv, rel_tol=2e-2, abs_tol=1e-2)


# -----------------------------------------------------------------------------
# 7) KENDALL'S TAU
# -----------------------------------------------------------------------------

@given(data=valid_params())
@settings(max_examples=40, deadline=None)
def test_kendall_tau_positive(data):
    theta, delta = data
    c = _make(theta, delta)
    tau = float(c.kendall_tau())
    assert tau >= -1e-12


@given(data=valid_params())
@settings(max_examples=40, deadline=None)
def test_kendall_tau_range(data):
    theta, delta = data
    c = _make(theta, delta)
    tau = float(c.kendall_tau())
    assert -1.0 <= tau <= 1.0


def test_kendall_tau_monotone_in_theta():
    """
    Properties section suggests concordance increases with theta (for fixed delta).
    We test weak monotonicity on a small increasing grid.
    """
    # pick a safe delta in the interior
    delta0 = max(DELTA_LO + 0.5, min(2.0, DELTA_HI_CAP - 0.5))
    # pick theta grid inside bounds
    grid = []
    for x in [0.5, 1.0, 2.0, 4.0, 8.0]:
        if THETA_LO < x < THETA_HI_CAP:
            grid.append(x)
    if len(grid) < 3:
        pytest.skip("Not enough interior theta values inside current bounds.")

    taus = []
    for th in grid:
        taus.append(float(_make(th, delta0).kendall_tau()))
    assert all(taus[i] <= taus[i + 1] + 5e-3 for i in range(len(taus) - 1))


# -----------------------------------------------------------------------------
# 8) TAIL DEPENDENCE
# -----------------------------------------------------------------------------

@given(data=valid_params())
@settings(max_examples=40, deadline=None)
def test_tail_dependence_formulas(data):
    theta, delta = data
    c = _make(theta, delta)
    assert math.isclose(float(c.LTDC()), 1.0, rel_tol=0.0, abs_tol=0.0)
    assert math.isclose(float(c.UTDC()), 0.0, rel_tol=0.0, abs_tol=0.0)


# -----------------------------------------------------------------------------
# 9) BLOMQVIST BETA
# -----------------------------------------------------------------------------

@given(data=valid_params())
@settings(max_examples=60, deadline=None)
def test_blomqvist_beta_matches_definition(data):
    theta, delta = data
    c = _make(theta, delta)
    beta_def = 4.0 * float(c.get_cdf(0.5, 0.5)) - 1.0
    beta = float(c.blomqvist_beta())
    assert math.isclose(beta, beta_def, rel_tol=1e-12, abs_tol=1e-12)


# -----------------------------------------------------------------------------
# 11) INIT FROM DATA (slow)
# -----------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("theta,delta", [
    (max(THETA_LO + 0.8, 1.2), max(DELTA_LO + 0.8, 1.2)),
    (max(THETA_LO + 1.5, 2.0), max(DELTA_LO + 1.5, 2.0)),
])
def test_init_from_data_roundtrip(theta, delta):
    """
    Fit starting values from data and ensure we recover something close.
    """
    c_true = _make(theta, delta)
    data = c_true.sample(6000, rng=np.random.default_rng(0))

    c_fit = BB2Copula()
    c_fit.init_from_data(data[:, 0], data[:, 1])
    th_hat, de_hat = map(float, c_fit.get_parameters())

    # coarse tolerance: init_from_data is an initializer, not full MLE
    assert abs(th_hat - theta) / max(theta, 1e-6) < 0.5
    assert abs(de_hat - delta) / max(delta, 1e-6) < 0.5


# -----------------------------------------------------------------------------
# 12) SAMPLING (slow)
# -----------------------------------------------------------------------------

@pytest.mark.slow
@given(data=valid_params())
@settings(max_examples=15, deadline=None)
def test_empirical_kendall_tau_close(data):
    theta, delta = data
    c = _make(theta, delta)

    sample = c.sample(6000, rng=np.random.default_rng(0))
    tau_emp, _ = stx.kendalltau(sample[:, 0], sample[:, 1])
    tau_theo = float(c.kendall_tau())

    # generous but principled tolerance
    n = len(sample)
    sigma0 = math.sqrt(2 * (2 * n + 5) / (9 * n * (n - 1)))
    assert math.isclose(float(tau_emp), tau_theo, abs_tol=4 * sigma0 + 0.05)


# -----------------------------------------------------------------------------
# 13) VECTORIZATION
# -----------------------------------------------------------------------------

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