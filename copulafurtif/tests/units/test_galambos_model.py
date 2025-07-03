"""
Unit-tests for the Galambos extreme-value copula.

Run:  pytest tests/units/test_galambos_model.py -q
"""
import math, numpy as np, pytest, scipy.stats as stx
from hypothesis import given, strategies as st, settings

from CopulaFurtif.core.copulas.domain.models.archimedean.galambos import GalambosCopula

# --------------------------------------------------------------------------- #
# Hypothesis helpers
# --------------------------------------------------------------------------- #
delta_valid   = st.floats(min_value=0.02, max_value=40.0,
                          allow_nan=False, allow_infinity=False)

delta_invalid = st.one_of(
    st.floats(max_value=0.0,  allow_nan=False),          # ≤ 0
    st.floats(min_value=55.0, allow_nan=False),          # > 50 (above bounds)
)

unit_interior = st.floats(min_value=1e-3, max_value=0.999,
                          allow_nan=False, allow_infinity=False)


def _fd(f, x, y, h=1e-5):
    """central finite difference wrt first arg"""
    return (f(x + h, y) - f(x - h, y)) / (2 * h)

# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
@given(delta=delta_valid)
def test_param_roundtrip(delta):
    c = GalambosCopula()
    c.set_parameters([delta])
    assert math.isclose(c.get_parameters()[0], delta, rel_tol=1e-12)


@given(delta=delta_invalid)
def test_param_out_of_bounds(delta):
    c = GalambosCopula()
    with pytest.raises(ValueError):
        c.set_parameters([delta])

# --------------------------------------------------------------------------- #
# CDF & PDF invariants
# --------------------------------------------------------------------------- #
@given(delta=delta_valid, u=unit_interior, v=unit_interior)
def test_cdf_bounds(delta, u, v):
    c = GalambosCopula(); c.set_parameters([delta])
    val = c.get_cdf(u, v)
    assert 0.0 <= val <= 1.0


@given(delta=delta_valid, u=unit_interior, v=unit_interior)
def test_pdf_nonneg(delta, u, v):
    c = GalambosCopula(); c.set_parameters([delta])
    pdf = c.get_pdf(u, v)
    # allows a very slight negative due to IEEE-754 rounding
    assert pdf >= -1e-12, f"pdf={pdf}"


@given(delta=delta_valid, u=unit_interior, v=unit_interior)
def test_cdf_symmetry(delta, u, v):
    c = GalambosCopula(); c.set_parameters([delta])
    assert math.isclose(c.get_cdf(u, v), c.get_cdf(v, u), rel_tol=1e-12)

# --------------------------------------------------------------------------- #
# Derivatives vs. finite-diff
# --------------------------------------------------------------------------- #
@given(delta=st.floats(min_value=0.05, max_value=6.0, allow_nan=False),
       u=unit_interior, v=unit_interior)
@settings(max_examples=40)
def test_partials(delta, u, v):
    c = GalambosCopula(); c.set_parameters([delta])

    C = c.get_cdf
    num_du = _fd(C, u, v)
    num_dv = _fd(lambda x, y: C(y, x), v, u)

    ana_du = c.partial_derivative_C_wrt_u(u, v)
    ana_dv = c.partial_derivative_C_wrt_v(u, v)

    assert math.isclose(ana_du, num_du, rel_tol=1e-2, abs_tol=1e-3)
    assert math.isclose(ana_dv, num_dv, rel_tol=1e-2, abs_tol=1e-3)

# --------------------------------------------------------------------------- #
# Kendall τ : empirical big-vs-small
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@settings(max_examples=20, deadline=None)
@given(delta=delta_valid)
def test_kendall_tau_monte_carlo(delta):
    rng = np.random.default_rng(321)

    cop = GalambosCopula(); cop.set_parameters([delta])

    uv_big   = cop.sample(200_000, rng=rng)
    tau_big, _ = stx.kendalltau(uv_big[:, 0], uv_big[:, 1])

    uv_small = cop.sample(50_000,  rng=rng)
    tau_small, _ = stx.kendalltau(uv_small[:, 0], uv_small[:, 1])

    se = 1 / math.sqrt(9 * 50_000 / 2)        # σ(τ̂) ≈ √[2(2n+5)] / 9√n(n−1)
    assert math.isclose(tau_small, tau_big,
                        abs_tol=4*se + 0.02), (
        f"δ={delta:.3f}: big τ={tau_big:.4f}, small τ={tau_small:.4f}"
    )

# --------------------------------------------------------------------------- #
# Tail dependence
# --------------------------------------------------------------------------- #
@given(delta=delta_valid)
def test_tail_dependence(delta):
    cop = GalambosCopula(); cop.set_parameters([delta])
    assert cop.LTDC() == 0.0
    expected_lambda_u = 2.0 ** (-1.0 / delta)
    assert math.isclose(cop.UTDC(), expected_lambda_u, rel_tol=1e-12)

# --------------------------------------------------------------------------- #
# Sampler shape & disabled metrics
# --------------------------------------------------------------------------- #
def test_sample_and_disabled_metrics():
    cop = GalambosCopula(); cop.set_parameters([2.0])
    samp = cop.sample(512)
    assert samp.shape == (512, 2)
    assert np.isnan(cop.IAD(None))
    assert np.isnan(cop.AD(None))
