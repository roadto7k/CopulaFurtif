"""
CopulaFurtif - Copula Model Mixins (Selection, GOF, Conditionals)

This module provides reusable mixins/interfaces that extend copula models with:
- Model selection criteria (AIC, BIC)
- Kendall's tau mismatch (empirical vs theoretical)
- Goodness-of-fit (GOF) helpers (AD / IAD) with safe subsampling and bootstrap
- Tail dependence diagnostics (Huang empirical tails + model LTDC/UTDC when available)
- Conditional CDFs via copula partial derivatives

Design notes
------------
1) GOF functions live in: CopulaFurtif.core.copulas.domain.estimation.gof
   This mixin wraps them into consistent methods on the copula object.

2) Some copula models historically exposed AD()/IAD() as placeholders returning NaN.
   This file keeps AD()/IAD() as compatibility wrappers that redirect to gof_AD/gof_IAD
   when available, but the recommended entry points are:
       - gof_AD(...)
       - gof_IAD(...)
       - gof_tail_metrics(...)
       - gof_summary(...)

3) Data format:
   Most methods accept either:
     - a tuple/list (x, y) of 1D arrays
     - a 2D array of shape (n_samples, 2)

4) Performance:
   AD/IAD computations can be O(n^2). Use subsampling (default) or bootstrap
   via the wrappers in estimation/gof.py.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

# Base selection / error metrics (always expected)
from CopulaFurtif.core.copulas.domain.estimation.gof import (
    compute_aic,
    compute_bic,
    kendall_tau_distance,
)

# Optional "extended" GOF helpers (depend on your gof.py)
try:
    from CopulaFurtif.core.copulas.domain.estimation.gof import (
        AD_score_subsampled,
        IAD_score_subsampled,
        AD_score_bootstrap,
        tail_metrics_huang,
        rosenblatt_pit_metrics,
    )
    _HAS_GOF_EXT = True
except Exception:
    AD_score_subsampled = None
    IAD_score_subsampled = None
    AD_score_bootstrap = None
    tail_metrics_huang = None
    _HAS_GOF_EXT = False


ArrayLike = Union[np.ndarray, list, tuple]


def _as_xy(data: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize input data to two 1D arrays (x, y).

    Accepted formats
    ----------------
    - np.ndarray of shape (n, 2) or (n, >=2)  -> uses first two columns
    - (x, y) tuple/list of array-like

    Returns
    -------
    (x, y): Tuple[np.ndarray, np.ndarray]
        Both arrays are 1D and have the same length.

    Raises
    ------
    ValueError
        If the input format is not recognized or lengths mismatch.
    """
    if isinstance(data, np.ndarray):
        arr = np.asarray(data)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            x = np.asarray(arr[:, 0]).ravel()
            y = np.asarray(arr[:, 1]).ravel()
            if len(x) != len(y):
                raise ValueError("Input array columns have different lengths.")
            return x, y

    if isinstance(data, (list, tuple)) and len(data) == 2:
        x = np.asarray(data[0]).ravel()
        y = np.asarray(data[1]).ravel()
        if len(x) != len(y):
            raise ValueError("Mismatch: len(x) != len(y).")
        return x, y

    raise ValueError("data must be a (n,2) array or a (x,y) tuple/list.")


class SupportsTailDependence:
    """
    Interface for copulas that provide tail dependence coefficients.

    Implementations should follow:
        LTDC(param) -> float : lower tail dependence
        UTDC(param) -> float : upper tail dependence
    """

    def LTDC(self, param):
        """
        Lower Tail Dependence Coefficient.

        Parameters
        ----------
        param : array-like
            Copula parameters.

        Returns
        -------
        float
            Lower tail dependence coefficient.

        Raises
        ------
        NotImplementedError
            Must be implemented by the copula model.
        """
        raise NotImplementedError

    def UTDC(self, param):
        """
        Upper Tail Dependence Coefficient.

        Parameters
        ----------
        param : array-like
            Copula parameters.

        Returns
        -------
        float
            Upper tail dependence coefficient.

        Raises
        ------
        NotImplementedError
            Must be implemented by the copula model.
        """
        raise NotImplementedError


class ModelSelectionMixin:
    """
    Mixin adding selection metrics and GOF helpers to a copula model.

    Expected copula attributes/methods
    ---------------------------------
    - log_likelihood_ : float (set after fitting)
    - n_obs : int (set after fitting; required for BIC)
    - get_bounds() -> Sequence[Tuple[float,float]] (used for k = #params)
    - get_parameters() -> array-like

    Main methods
    ------------
    - AIC(), BIC()
    - kendall_tau_error(data)
    - gof_AD(...), gof_IAD(...)
    - gof_tail_metrics(data)
    - gof_summary(data, ...)

    Notes
    -----
    The GOF methods here call helpers in estimation/gof.py. By default, they use
    subsampling to avoid O(n^2) blow-ups.
    """

    # -----------------------
    # Model selection criteria
    # -----------------------
    def AIC(self) -> float:
        """
        Compute Akaike Information Criterion.

        Returns
        -------
        float
            AIC score (lower is better).

        Raises
        ------
        RuntimeError
            If the model is not fitted (log_likelihood_ is missing).
        """
        if getattr(self, "log_likelihood_", None) is None:
            raise RuntimeError("Copula must be fitted before computing AIC.")
        return float(compute_aic(self))

    def BIC(self) -> float:
        """
        Compute Bayesian Information Criterion.

        Returns
        -------
        float
            BIC score (lower is better).

        Raises
        ------
        RuntimeError
            If log_likelihood_ or n_obs is missing.
        """
        if getattr(self, "log_likelihood_", None) is None or getattr(self, "n_obs", None) is None:
            raise RuntimeError("Missing log_likelihood_ or n_obs. Fit the copula first.")
        return float(compute_bic(self))

    # -----------------------
    # Error / mismatch metrics
    # -----------------------
    def kendall_tau_error(self, data: ArrayLike) -> float:
        """
        Absolute distance between empirical Kendall's tau and the model tau.

        Parameters
        ----------
        data : array-like
            Either (x,y) or (n,2) array of observed samples.

        Returns
        -------
        float
            |tau_empirical - tau_model|, or NaN if the model doesn't implement kendall_tau().
        """
        x, y = _as_xy(data)
        return float(kendall_tau_distance(self, (x, y)))

    # -----------------------
    # GOF helpers (AD / IAD)
    # -----------------------
    def gof_AD(
        self,
        data: ArrayLike,
        *,
        mode: str = "subsample",
        m: int = 300,
        seed: int = 0,
        n_boot: int = 3,
        return_std: bool = False,
    ):
        """
        Anderson-Darling style goodness-of-fit score (tail sensitive).

        Parameters
        ----------
        data : array-like
            Observed samples (x,y) or (n,2) array.
        mode : {"subsample", "bootstrap"}
            - "subsample": compute a single AD score on a tail-aware subsample
            - "bootstrap": compute mean/std over n_boot subsamples
        m : int
            Subsample size for AD (typical: 200-400).
        seed : int
            RNG seed used by the subsampling procedure.
        n_boot : int
            Number of bootstrap resamples when mode="bootstrap".
        return_std : bool
            If True and mode="bootstrap", returns (mean, std).
            Otherwise returns mean only.

        Returns
        -------
        float or (float, float)
            AD score (lower is better). If bootstrap+return_std -> (mean, std).

        Raises
        ------
        RuntimeError
            If extended GOF helpers are not available.
        """
        if not _HAS_GOF_EXT or AD_score_subsampled is None or AD_score_bootstrap is None:
            raise RuntimeError("Extended GOF helpers not available. Check estimation/gof.py exports.")

        x, y = _as_xy(data)

        if mode == "bootstrap":
            mu, sig = AD_score_bootstrap(self, x, y, m=m, n_boot=n_boot, seed=seed)
            return (float(mu), float(sig)) if return_std else float(mu)

        # default: subsample
        return float(AD_score_subsampled(self, x, y, m=m, seed=seed))

    def gof_IAD(
        self,
        data: ArrayLike,
        *,
        mode: str = "subsample",
        m: int = 250,
        seed: int = 0,
    ) -> float:
        """
        Integrated Anderson-Darling style goodness-of-fit score.

        Parameters
        ----------
        data : array-like
            Observed samples (x,y) or (n,2) array.
        mode : {"subsample"}
            Currently only "subsample" is supported via IAD_score_subsampled.
        m : int
            Subsample size (typical: 150-300).
        seed : int
            RNG seed used by the subsampling procedure.

        Returns
        -------
        float
            IAD score (lower is better).

        Raises
        ------
        RuntimeError
            If extended GOF helpers are not available.
        """
        if not _HAS_GOF_EXT or IAD_score_subsampled is None:
            raise RuntimeError("Extended GOF helpers not available. Check estimation/gof.py exports.")

        x, y = _as_xy(data)
        return float(IAD_score_subsampled(self, x, y, m=m, seed=seed))

    # -----------------------
    # Tail diagnostics
    # -----------------------
    def gof_tail_metrics(self, data: ArrayLike) -> Dict[str, float]:
        """
        Tail dependence metrics using Huang empirical estimates and model LTDC/UTDC.

        This requires:
          - estimation/gof.py: tail_metrics_huang
          - the copula model implements LTDC/UTDC (SupportsTailDependence)

        Parameters
        ----------
        data : array-like
            Observed samples (x,y) or (n,2) array.

        Returns
        -------
        dict
            A dictionary like:
              {
                "lambdaL_emp_huang": ...,
                "lambdaU_emp_huang": ...,
                "lambdaL_model": ...,
                "lambdaU_model": ...,
              }
            Empty dict if unavailable.
        """
        if not _HAS_GOF_EXT or tail_metrics_huang is None:
            return {}

        x, y = _as_xy(data)
        try:
            out = tail_metrics_huang(self, (x, y))
            return dict(out) if isinstance(out, dict) else {}
        except Exception:
            return {}

    # -----------------------
    # PIT diagnostics
    # -----------------------

    def gof_PIT(
            self,
            data: ArrayLike,
            *,
            m: int = 400,
            seed: int = 0,
            q_tail: float = 0.10,
            tail_frac: float = 0.33,
    ):
        """
        Rosenblatt-PIT diagnostics (KS vs U(0,1) on z2 = F_{V|U}(V)).
        Returns dict (PIT_ks_D, PIT_ks_pvalue, optional tail slices...).
        """
        if not _HAS_GOF_EXT or rosenblatt_pit_metrics is None:
            return {"PIT": np.nan}

        try:
            return rosenblatt_pit_metrics(self, data, max_n=m, seed=seed, q_tail=q_tail, tail_frac=tail_frac)
        except Exception:
            return {"PIT": np.nan}

    # -----------------------
    # Convenience bundle
    # -----------------------
    def gof_summary(
        self,
        data: ArrayLike,
        *,
        include_aic: bool = True,
        include_bic: bool = True,
        include_tau: bool = True,
        include_tails: bool = True,
        include_ad: bool = True,
        include_iad: bool = True,
        include_pit: bool = True,
        ad_mode: str = "subsample",
        ad_m: int = 300,
        ad_seed: int = 0,
        ad_boot: int = 0,
        iad_m: int = 250,
        iad_seed: int = 0,
        pit_m: int = 400,
        pit_seed: int = 0,
        pit_q_tail: float = 0.10,
        pit_tail_frac: float = 0.33,
    ) -> Dict[str, Any]:
        """
        Compute a diagnostic bundle used for model selection.

        Parameters
        ----------
        data : array-like
            Observed samples (x,y) or (n,2) array.
        include_aic, include_bic, include_tau, include_tails, include_ad, include_iad : bool
            Enable/disable each family of metrics.
        ad_mode : {"subsample","bootstrap"}
            AD computation mode.
        ad_m : int
            AD subsample size.
        ad_seed : int
            AD RNG seed.
        ad_boot : int
            If >0, forces bootstrap mode with n_boot=ad_boot.
        iad_m : int
            IAD subsample size.
        iad_seed : int
            IAD RNG seed.

        Returns
        -------
        dict
            Dictionary of computed metrics. Missing metrics are omitted or set to NaN.
        """
        res: Dict[str, Any] = {}

        if include_aic:
            try:
                res["AIC"] = self.AIC()
            except Exception:
                res["AIC"] = np.nan

        if include_bic:
            try:
                res["BIC"] = self.BIC()
            except Exception:
                res["BIC"] = np.nan

        if include_tau:
            try:
                res["KT_err"] = self.kendall_tau_error(data)
            except Exception:
                res["KT_err"] = np.nan

        if include_tails:
            res.update(self.gof_tail_metrics(data))

        if include_ad and _HAS_GOF_EXT:
            try:
                if int(ad_boot) > 0:
                    mu, sig = self.gof_AD(
                        data,
                        mode="bootstrap",
                        m=ad_m,
                        seed=ad_seed,
                        n_boot=int(ad_boot),
                        return_std=True,
                    )
                    res["AD_boot_mean"] = mu
                    res["AD_boot_std"] = sig
                    res["AD"] = mu  # convenient alias for selection
                else:
                    res["AD"] = self.gof_AD(data, mode=ad_mode, m=ad_m, seed=ad_seed)
            except Exception:
                res["AD"] = np.nan

        if include_iad and _HAS_GOF_EXT:
            try:
                res["IAD"] = self.gof_IAD(data, m=iad_m, seed=iad_seed)
            except Exception:
                res["IAD"] = np.nan

        if include_pit:
            res.update(self.gof_PIT(
                data, m=pit_m, seed=pit_seed, q_tail=pit_q_tail, tail_frac=pit_tail_frac
            ))

        return res


class AdvancedCopulaFeatures:
    """
    Mixin adding conditional distributions and derivative-based helpers.

    A copula model should implement the partial derivatives:
      - partial_derivative_C_wrt_u(u, v, param=None) = ∂C(u,v)/∂u
      - partial_derivative_C_wrt_v(u, v, param=None) = ∂C(u,v)/∂v

    Then conditional CDFs are:
      - U|V=v : C_u|v(u|v) = ∂C(u,v)/∂v
      - V|U=u : C_v|u(v|u) = ∂C(u,v)/∂u

    Compatibility note
    ------------------
    Historically, some classes exposed AD()/IAD() placeholders returning NaN.
    This mixin keeps AD()/IAD() as wrappers:
      - If the object also has gof_AD/gof_IAD, those are called.
      - Otherwise returns NaN.
    """

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        """
        Compute ∂C(u,v)/∂u.

        Parameters
        ----------
        u, v : float or array-like
            Values in (0,1).
        param : array-like, optional
            Copula parameters. If None, implementations may use self.get_parameters().

        Raises
        ------
        NotImplementedError
            Must be implemented by concrete copula models.
        """
        raise NotImplementedError

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        """
        Compute ∂C(u,v)/∂v.

        Parameters
        ----------
        u, v : float or array-like
            Values in (0,1).
        param : array-like, optional
            Copula parameters. If None, implementations may use self.get_parameters().

        Raises
        ------
        NotImplementedError
            Must be implemented by concrete copula models.
        """
        raise NotImplementedError

    def conditional_cdf_u_given_v(self, u, v, param=None):
        """
        Conditional CDF of U given V=v, i.e. P(U <= u | V=v).

        Default implementation uses:
            ∂C(u,v)/∂v

        Returns
        -------
        float or np.ndarray
            Conditional CDF values.
        """
        return self.partial_derivative_C_wrt_v(u, v, param)

    def conditional_cdf_v_given_u(self, u, v, param=None):
        """
        Conditional CDF of V given U=u, i.e. P(V <= v | U=u).

        Default implementation uses:
            ∂C(u,v)/∂u

        Returns
        -------
        float or np.ndarray
            Conditional CDF values.
        """
        return self.partial_derivative_C_wrt_u(u, v, param)

    # -----------------------
    # Backward-compatible GOF wrappers
    # -----------------------
    def AD(self, data, **kwargs):
        """
        Backward-compatible AD method.

        Prefer calling:
            self.gof_AD(data, ...)

        Returns
        -------
        float
            AD score, or NaN if unavailable.
        """
        if hasattr(self, "gof_AD"):
            try:
                return float(self.gof_AD(data, **kwargs))
            except Exception:
                return np.nan
        return np.nan

    def IAD(self, data, **kwargs):
        """
        Backward-compatible IAD method.

        Prefer calling:
            self.gof_IAD(data, ...)

        Returns
        -------
        float
            IAD score, or NaN if unavailable.
        """
        if hasattr(self, "gof_IAD"):
            try:
                return float(self.gof_IAD(data, **kwargs))
            except Exception:
                return np.nan
        return np.nan
