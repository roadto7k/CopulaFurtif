# # dash_bot/core/signals.py
# import numpy as np
# from typing import Tuple, Dict
# from .copula_engine import ecdf_value, copula_h_funcs
from __future__ import annotations

# dash_bot/core/signals.py
import numpy as np
from typing import Tuple, Dict
from .copula_engine import ecdf_value, copula_h_funcs, marginal_cdf_value, marginal_cdf_array


def generate_signals_reference_copula(
    cop,
    sorted_s1: np.ndarray,
    sorted_s2: np.ndarray,
    s1_val: float,
    s2_val: float,
    entry: float,
    exit: float,
    marginal1: dict | None = None,
    marginal2: dict | None = None,
) -> Tuple[int, Dict[str, float]]:
    """
    Retourne (signal, details):
      signal: +1 => long coin1 / short coin2
              -1 => short coin1 / long coin2
               0 => close/flat (no new)
    """
    if marginal1 is not None and marginal2 is not None:
        u = marginal_cdf_value(marginal1, s1_val)
        v = marginal_cdf_value(marginal2, s2_val)
    else:
        u = ecdf_value(sorted_s1, s1_val)
        v = ecdf_value(sorted_s2, s2_val)
    h12, h21 = copula_h_funcs(cop, u, v)

    # Table 4 conditions
    open_short_coin1_long_coin2 = (h12 < entry) and (h21 > 1 - entry)  # => signal -1
    open_long_coin1_short_coin2 = (h12 > 1 - entry) and (h21 < entry)  # => signal +1
    close_cond = (abs(h12 - 0.5) < exit) and (abs(h21 - 0.5) < exit)

    sig = 0
    if close_cond:
        sig = 0
    elif open_long_coin1_short_coin2:
        sig = +1
    elif open_short_coin1_long_coin2:
        sig = -1

    details = dict(u=u, v=v, h12=h12, h21=h21, close=float(close_cond))
    return sig, details

def generate_signals_reference_copula_array(
    cop,
    sorted_s1: np.ndarray,
    sorted_s2: np.ndarray,
    s1_vals: np.ndarray,
    s2_vals: np.ndarray,
    entry: float,
    exit: float,
    marginal1: dict | None = None,
    marginal2: dict | None = None,
):
    """
    Vectorized signal generation over full arrays of spread values.

    Returns
    -------
    sig : np.ndarray
        Integer signal array.

    det : dict
        Arrays containing u, v, h12, h21 and close.
    """
    s1_vals = np.asarray(
        s1_vals,
        dtype=float,
    ).ravel()

    s2_vals = np.asarray(
        s2_vals,
        dtype=float,
    ).ravel()

    if s1_vals.shape != s2_vals.shape:
        raise ValueError(
            "s1_vals and s2_vals must have the same shape."
        )

    # ------------------------------------------------------------------
    # Marginal PIT
    # ------------------------------------------------------------------
    if marginal1 is not None and marginal2 is not None:
        u = marginal_cdf_array(
            marginal1,
            s1_vals,
        )

        v = marginal_cdf_array(
            marginal2,
            s2_vals,
        )

    else:
        n1 = len(sorted_s1)
        n2 = len(sorted_s2)

        if n1 == 0 or n2 == 0:
            raise ValueError(
                "Empty empirical marginals."
            )

        u = np.clip(
            np.searchsorted(
                sorted_s1,
                s1_vals,
                side="right",
            ) / (n1 + 1.0),
            1e-6,
            1.0 - 1e-6,
        )

        v = np.clip(
            np.searchsorted(
                sorted_s2,
                s2_vals,
                side="right",
            ) / (n2 + 1.0),
            1e-6,
            1.0 - 1e-6,
        )

    # ------------------------------------------------------------------
    # H-functions
    #
    # Keep exactly the same clipping convention as copula_h_funcs().
    # This ensures that vectorization does not alter historical signals.
    # ------------------------------------------------------------------
    h_eps = 1e-4

    u_eval = np.clip(
        u,
        h_eps,
        1.0 - h_eps,
    )

    v_eval = np.clip(
        v,
        h_eps,
        1.0 - h_eps,
    )

    if (
        hasattr(cop, "conditional_cdf_u_given_v")
        and hasattr(cop, "conditional_cdf_v_given_u")
    ):
        try:
            h12 = np.asarray(
                cop.conditional_cdf_u_given_v(
                    u_eval,
                    v_eval,
                ),
                dtype=float,
            ).ravel()

            h21 = np.asarray(
                cop.conditional_cdf_v_given_u(
                    u_eval,
                    v_eval,
                ),
                dtype=float,
            ).ravel()

            if h12.size != u.size or h21.size != u.size:
                raise ValueError(
                    "Vectorized h-functions returned an invalid shape."
                )

            h12 = np.clip(
                h12,
                0.0,
                1.0,
            )

            h21 = np.clip(
                h21,
                0.0,
                1.0,
            )

        except Exception:
            h12 = np.empty_like(u)
            h21 = np.empty_like(v)

            for i in range(len(u)):
                h12[i], h21[i] = copula_h_funcs(
                    cop,
                    float(u[i]),
                    float(v[i]),
                )

    else:
        h12 = np.empty_like(u)
        h21 = np.empty_like(v)

        for i in range(len(u)):
            h12[i], h21[i] = copula_h_funcs(
                cop,
                float(u[i]),
                float(v[i]),
            )

    # ------------------------------------------------------------------
    # Trading conditions
    # ------------------------------------------------------------------
    open_short_coin1_long_coin2 = (
        (h12 < entry)
        & (h21 > 1.0 - entry)
    )

    open_long_coin1_short_coin2 = (
        (h12 > 1.0 - entry)
        & (h21 < entry)
    )

    close_cond = (
        (np.abs(h12 - 0.5) < exit)
        & (np.abs(h21 - 0.5) < exit)
    )

    sig = np.zeros(
        len(u),
        dtype=np.int8,
    )

    sig[
        open_long_coin1_short_coin2
    ] = +1

    sig[
        open_short_coin1_long_coin2
    ] = -1

    sig[
        close_cond
    ] = 0

    det = {
        "u": u,
        "v": v,
        "h12": h12,
        "h21": h21,
        "close": close_cond.astype(
            np.float64
        ),
    }

    return sig, det