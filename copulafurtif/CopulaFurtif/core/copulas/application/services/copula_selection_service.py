from dataclasses import dataclass
import numpy as np

from CopulaFurtif.core.copulas.domain.copula_type import CopulaType
from CopulaFurtif.core.copulas.domain.factories.copula_factory import CopulaFactory
from CopulaFurtif.core.copulas.application.services.fit_copula import CopulaFitter
from CopulaFurtif.core.copulas.domain.estimation.estimation import pseudo_obs


@dataclass
class SelectionConfig:
    candidate_types: list[CopulaType]

    mode: str = "medium"  # "ultra" | "medium" | "slow"

    # subsampling budgets
    pit_m: int = 400
    ad_m: int = 250
    iad_m: int = 220

    # only compute expensive stuff for top-k
    iad_topk: int = 0       # 0 => off
    refine_topk: int = 0    # 0 => off (refine via quick_fit cmle)

    # weights (lower is better for all these scores)
    w_aic: float = 1.0
    w_tau: float = 1.0
    w_tail: float = 1.0
    w_pit: float = 1.0
    w_iad: float = 0.0


def _robust_z(x):
    """Median/MAD z-score; NaN-safe."""
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad < 1e-12:
        return x - med
    return (x - med) / (1.4826 * mad)


def _tail_gap(metrics: dict) -> float:
    """Empirical vs model tail dependence mismatch."""
    a = metrics.get("lambdaL_emp_huang", np.nan)
    b = metrics.get("lambdaL_model", np.nan)
    c = metrics.get("lambdaU_emp_huang", np.nan)
    d = metrics.get("lambdaU_model", np.nan)
    vals = [abs(a - b), abs(c - d)]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.sum(vals)) if vals else np.nan


def _copula_only_loglik(copula, u, v, theta=None):
    """Fast copula-only loglik on pseudo-obs."""
    theta = copula.get_parameters() if theta is None else theta
    pdf = copula.get_pdf(u, v, theta)
    pdf = np.clip(pdf, 1e-12, None)
    return float(np.sum(np.log(pdf)))


class CopulaSelector:
    def __init__(self, cfg: SelectionConfig):
        self.cfg = cfg
        self.fitter = CopulaFitter()

    def _fit_stage1_tau(self, copula, data):
        # tau init only
        self.fitter.fit_tau(data, copula)
        # store loglik quickly (copula-only)
        u, v = pseudo_obs(data)
        ll = _copula_only_loglik(copula, u, v, copula.get_parameters())
        copula.log_likelihood_ = ll
        copula.n_obs = len(u)

    def select(self, data):
        """
        Returns
        -------
        best_copula, ranking
        ranking: list of dicts (one per candidate), sorted by total score
        """
        rows = []
        copulas = []

        # -------- Stage 1: fast fit + fast diagnostics for all candidates
        for ct in self.cfg.candidate_types:
            try:
                cop = CopulaFactory.create(ct)
            except Exception:
                continue

            try:
                self._fit_stage1_tau(cop, data)

                # fast summary: tau + tails + PIT (+ AIC)
                summ = cop.gof_summary(
                    data,
                    include_aic=True,
                    include_bic=False,
                    include_tau=True,
                    include_tails=True,
                    include_ad=False,
                    include_iad=False,
                    include_pit=True,
                    pit_m=self.cfg.pit_m,
                )

                summ["CopulaType"] = ct
                summ["CopulaName"] = getattr(cop, "name", str(ct))
                summ["AIC"] = summ.get("AIC", np.nan)
                summ["TailGap"] = _tail_gap(summ)
                summ["PIT"] = summ.get("PIT", summ.get("PIT_ks_D", np.nan))
                rows.append(summ)
                copulas.append(cop)

            except Exception:
                # candidate rejected
                continue

        if not rows:
            return None, []

        # -------- Stage 1 scoring (robust normalization)
        aic = np.array([r.get("AIC", np.nan) for r in rows], float)
        tau = np.array([r.get("KT_err", np.nan) for r in rows], float)
        tail = np.array([r.get("TailGap", np.nan) for r in rows], float)
        pit = np.array([r.get("PIT", np.nan) for r in rows], float)

        za = _robust_z(aic)
        zt = _robust_z(tau)
        zl = _robust_z(tail)
        zp = _robust_z(pit)

        total = (
            self.cfg.w_aic * za +
            self.cfg.w_tau * zt +
            self.cfg.w_tail * zl +
            self.cfg.w_pit * zp
        )

        for i, r in enumerate(rows):
            r["Score_stage1"] = float(total[i])

        order = np.argsort(total)
        rows = [rows[i] for i in order]
        copulas = [copulas[i] for i in order]

        # -------- Stage 2: optional IAD on top-k (expensive O(m^2))
        if int(self.cfg.iad_topk) > 0:
            k = min(int(self.cfg.iad_topk), len(copulas))
            for i in range(k):
                cop = copulas[i]
                try:
                    iad = cop.gof_IAD(data, m=self.cfg.iad_m, seed=0)
                except Exception:
                    iad = np.nan
                rows[i]["IAD"] = iad

            # recompute total if w_iad > 0
            if self.cfg.w_iad > 0:
                iad_arr = np.array([r.get("IAD", np.nan) for r in rows], float)
                zi = _robust_z(iad_arr)
                total2 = np.array([r["Score_stage1"] for r in rows], float) + self.cfg.w_iad * zi
                for i, r in enumerate(rows):
                    r["Score_total"] = float(total2[i])
                ord2 = np.argsort(total2)
                rows = [rows[i] for i in ord2]
                copulas = [copulas[i] for i in ord2]
            else:
                for r in rows:
                    r["Score_total"] = r["Score_stage1"]
        else:
            for r in rows:
                r["Score_total"] = r["Score_stage1"]

        best = copulas[0] if copulas else None
        return best, rows
