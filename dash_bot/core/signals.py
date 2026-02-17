# dash_bot/core/signals.py
import numpy as np
from typing import Tuple, Dict
from .copula_engine import ecdf_value, copula_h_funcs

def generate_signals_reference_copula(
    cop,
    sorted_s1: np.ndarray,
    sorted_s2: np.ndarray,
    s1_val: float,
    s2_val: float,
    entry: float,
    exit: float,
) -> Tuple[int, Dict[str, float]]:
    """
    Retourne (signal, details):
      signal: +1 => long coin1 / short coin2
              -1 => short coin1 / long coin2
               0 => close/flat (no new)
    """
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