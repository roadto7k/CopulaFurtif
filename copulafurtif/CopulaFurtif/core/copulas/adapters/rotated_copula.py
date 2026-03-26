"""
CopulaFurtif — RotatedCopula wrapper
=====================================
Applies a 0 / 90 / 180 / 270° rotation to any CopulaFurtif copula object.

Rotation formulas
-----------------
Let C = base CDF, c = base PDF, h12 = ∂C/∂v, h21 = ∂C/∂u.

R0   (identity):
    C_0(u,v)    = C(u, v)
    h12_0(u|v)  = h12(u, v)
    h21_0(v|u)  = h21(u, v)

R90  (lower-right tail):
    C_90(u,v)   = v - C(1-u, v)
    h12_90(u|v) = 1 - h12(1-u, v)
    h21_90(v|u) =     h21(1-u, v)

R180 (upper-right tail / survival):
    C_180(u,v)  = u + v - 1 + C(1-u, 1-v)
    h12_180     = 1 - h12(1-u, 1-v)
    h21_180     = 1 - h21(1-u, 1-v)

R270 (upper-left tail):
    C_270(u,v)  = u - C(u, 1-v)
    h12_270     =     h12(u, 1-v)
    h21_270     = 1 - h21(u, 1-v)

Fitting philosophy
------------------
The wrapper transforms pseudo-observations BEFORE fitting so that parameters
are estimated in the correct rotated space.  For R90 this means the base
copula is fitted on (1-u, v) — NOT on (u, v) with a post-hoc rotation.

Coordinate transform applied before fitting:
    R0   : (u, v)       identity
    R90  : (1-u, v)
    R180 : (1-u, 1-v)
    R270 : (u,  1-v)

Since _fit_tau_core builds pseudo-obs internally from raw (x, y), we mirror
the same transform at the raw-data level: rank(1-x) ≈ 1 - rank(x).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

_EPS = 1e-10


def _clip(x: Any) -> Any:
    return np.clip(x, _EPS, 1.0 - _EPS)


class RotatedCopula:
    """
    Thin wrapper that rotates a CopulaFurtif copula by 0 / 90 / 180 / 270°.

    Parameters
    ----------
    base_copula : CopulaFurtif copula object
        A freshly-created, unfitted copula instance.
    rotation : {0, 90, 180, 270}
        Rotation angle in degrees.
    """

    VALID_ROTATIONS = (0, 90, 180, 270)

    def __init__(self, base_copula: Any, rotation: int) -> None:
        if rotation not in self.VALID_ROTATIONS:
            raise ValueError(
                f"rotation must be one of {self.VALID_ROTATIONS}, got {rotation}"
            )
        self._base = base_copula
        self.rotation = rotation
        self.log_likelihood_: float | None = None
        self.n_obs: int | None = None

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------

    def _transform_uv(self, u: Any, v: Any) -> Tuple[Any, Any]:
        """Transform pseudo-obs (u,v) into the rotated space before fitting."""
        u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
        if self.rotation == 0:
            return u, v
        elif self.rotation == 90:
            return 1.0 - u, v
        elif self.rotation == 180:
            return 1.0 - u, 1.0 - v
        else:  # 270
            return u, 1.0 - v

    def _transform_raw(self, x: Any, y: Any) -> Tuple[Any, Any]:
        """
        Transform raw data so that after rank-transform inside gof_summary /
        fit_cmle the pseudo-obs are the rotated ones.
        rank(1 - x_i) ≈ 1 - rank(x_i)
        """
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if self.rotation == 0:
            return x, y
        elif self.rotation == 90:
            return 1.0 - x, y
        elif self.rotation == 180:
            return 1.0 - x, 1.0 - y
        else:  # 270
            return x, 1.0 - y

    # ------------------------------------------------------------------
    # Fitting interface  (called by fitter.fit_tau / fitter.fit_cmle)
    # ------------------------------------------------------------------

    def init_from_data(self, u: Any, v: Any) -> Any:
        """
        Called by _fit_tau_core after it has built pseudo-obs (u, v).
        We rotate them before delegating to the base copula.
        """
        ut, vt = self._transform_uv(_clip(u), _clip(v))
        return self._base.init_from_data(ut, vt)

    # ------------------------------------------------------------------
    # Parameter access
    # ------------------------------------------------------------------

    def get_parameters(self) -> np.ndarray:
        return self._base.get_parameters()

    def set_parameters(self, theta: Any) -> None:
        self._base.set_parameters(np.asarray(theta, dtype=float))

    # ------------------------------------------------------------------
    # CDF / PDF
    # ------------------------------------------------------------------

    def get_cdf(self, u: Any, v: Any) -> Any:
        u, v = _clip(u), _clip(v)
        if self.rotation == 0:
            return self._base.get_cdf(u, v)
        elif self.rotation == 90:
            return v - self._base.get_cdf(1.0 - u, v)
        elif self.rotation == 180:
            return u + v - 1.0 + self._base.get_cdf(1.0 - u, 1.0 - v)
        else:  # 270
            return u - self._base.get_cdf(u, 1.0 - v)

    def get_pdf(self, u: Any, v: Any, theta: Any = None) -> Any:
        """Jacobian of all rotations is 1 — PDF = base PDF at transformed coords."""
        ut, vt = self._transform_uv(_clip(u), _clip(v))
        ut, vt = _clip(ut), _clip(vt)
        try:
            return (
                self._base.get_pdf(ut, vt, theta)
                if theta is not None
                else self._base.get_pdf(ut, vt)
            )
        except TypeError:
            return self._base.get_pdf(ut, vt)

    # ------------------------------------------------------------------
    # H-functions
    # ------------------------------------------------------------------

    def conditional_cdf_u_given_v(self, u: Any, v: Any) -> Any:
        """h1|2 for the rotated copula."""
        u, v = _clip(u), _clip(v)
        if self.rotation == 0:
            return self._base.conditional_cdf_u_given_v(u, v)
        elif self.rotation == 90:
            return 1.0 - self._base.conditional_cdf_u_given_v(1.0 - u, v)
        elif self.rotation == 180:
            return 1.0 - self._base.conditional_cdf_u_given_v(1.0 - u, 1.0 - v)
        else:  # 270
            return self._base.conditional_cdf_u_given_v(u, 1.0 - v)

    def conditional_cdf_v_given_u(self, u: Any, v: Any) -> Any:
        """h2|1 for the rotated copula."""
        u, v = _clip(u), _clip(v)
        if self.rotation == 0:
            return self._base.conditional_cdf_v_given_u(u, v)
        elif self.rotation == 90:
            return self._base.conditional_cdf_v_given_u(1.0 - u, v)
        elif self.rotation == 180:
            return 1.0 - self._base.conditional_cdf_v_given_u(1.0 - u, 1.0 - v)
        else:  # 270
            return 1.0 - self._base.conditional_cdf_v_given_u(u, 1.0 - v)

    # ------------------------------------------------------------------
    # GOF
    # ------------------------------------------------------------------

    def gof_summary(self, data: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Pass transformed raw data to the base copula's gof_summary.
        Propagates log_likelihood_ and n_obs so AIC/BIC are correct.
        """
        x = np.asarray(data[0], dtype=float)
        y = np.asarray(data[1], dtype=float)
        xt, yt = self._transform_raw(x, y)
        if self.log_likelihood_ is not None:
            self._base.log_likelihood_ = self.log_likelihood_
        if self.n_obs is not None:
            self._base.n_obs = self.n_obs
        return self._base.gof_summary((xt, yt), **kwargs)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    @property
    def base_name(self) -> str:
        return getattr(self._base, "name", type(self._base).__name__)

    def __repr__(self) -> str:
        return f"RotatedCopula({self.base_name}, R{self.rotation})"