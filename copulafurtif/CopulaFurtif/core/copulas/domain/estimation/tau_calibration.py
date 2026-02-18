# copulafurtif/CopulaFurtif/core/copulas/domain/estimation/tau_calibration.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import brentq

from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel


@dataclass(frozen=True)
class TauCalibrationResult:
    tau_target: float
    param: np.ndarray
    tau_achieved: float
    param_index: int


def calibrate_to_kendall_tau(
    copula: CopulaModel,
    tau_target: float,
    *,
    param_index: int = 0,
    fixed_params: Optional[Dict[int, float]] = None,
    eps: float = 1e-10,
    maxiter: int = 200,
    scan_n: int = 80,
) -> TauCalibrationResult:
    """
    Calibrate one copula parameter to hit a target Kendall's tau.

    Uses ONLY the existing CopulaModel interface:
      - get_bounds()
      - get_parameters()
      - set_parameters()
      - kendall_tau(param)

    For multi-parameter copulas, you typically vary param_index and provide
    fixed_params={other_index: value, ...}.
    """
    tau_target = float(np.clip(float(tau_target), -0.999, 0.999))

    bounds = copula.get_bounds()
    if not bounds:
        raise ValueError(f"{type(copula).__name__}: get_bounds() returned empty")

    idx = int(param_index)
    if idx < 0 or idx >= len(bounds):
        raise ValueError(f"{type(copula).__name__}: param_index={idx} out of range (n={len(bounds)})")

    # Start from current params if available; else midpoints of bounds
    try:
        p0 = np.asarray(copula.get_parameters(), dtype=float).ravel()
    except Exception:
        p0 = np.array([(lo + hi) * 0.5 for (lo, hi) in bounds], dtype=float)

    if p0.size != len(bounds):
        p0 = np.array([(lo + hi) * 0.5 for (lo, hi) in bounds], dtype=float)

    p = p0.copy()

    # Apply fixed params
    if fixed_params:
        for k, v in fixed_params.items():
            kk = int(k)
            if 0 <= kk < p.size:
                p[kk] = float(v)

    lo, hi = bounds[idx]
    lo = float(lo) + float(eps)
    hi = float(hi) - float(eps)
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        raise ValueError(f"{type(copula).__name__}: invalid bounds for param[{idx}]={bounds[idx]}")

    def f(x: float) -> float:
        pp = p.copy()
        pp[idx] = float(x)
        return float(copula.kendall_tau(pp)) - tau_target

    # Try direct bracketing with bounds
    flo, fhi = f(lo), f(hi)

    a, b = lo, hi
    if not (np.isfinite(flo) and np.isfinite(fhi)) or flo * fhi > 0:
        # No sign change: scan to find a bracket
        grid = np.linspace(lo, hi, int(scan_n))
        vals = np.array([f(xx) for xx in grid], dtype=float)
        ok = np.isfinite(vals)
        if ok.sum() < 2:
            raise ValueError(f"{type(copula).__name__}: kendall_tau() not evaluable on scan grid")

        grid = grid[ok]
        vals = vals[ok]
        s = np.sign(vals)
        changes = np.where(s[:-1] * s[1:] < 0)[0]

        if changes.size == 0:
            # Still no bracket: take closest point
            j = int(np.argmin(np.abs(vals)))
            p[idx] = float(grid[j])
            tau_ach = float(copula.kendall_tau(p))
            return TauCalibrationResult(tau_target=tau_target, param=p, tau_achieved=tau_ach, param_index=idx)

        a = float(grid[changes[0]])
        b = float(grid[changes[0] + 1])

    root = float(brentq(lambda xx: f(xx), a=a, b=b, maxiter=int(maxiter)))
    p[idx] = root
    tau_ach = float(copula.kendall_tau(p))

    return TauCalibrationResult(tau_target=tau_target, param=p, tau_achieved=tau_ach, param_index=idx)
