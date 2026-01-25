import pandas as pd
import numpy as np
from CopulaFurtif.copulas import CopulaFactory, CopulaType
from CopulaFurtif.copulas import CopulaDiagnostics, CopulaFitter
from statsmodels.distributions.copula.api import (
        GaussianCopula, StudentTCopula, ClaytonCopula, FrankCopula, GumbelCopula
    )
from scipy.stats import t as student_t
HAS_COPULAFURTIF = True
HAS_SM_COPULA = False

def aic_val(loglik, k): return 2*k - 2*loglik

def fit_copulas(x, y, *, method="quick", maxiter=120, optimizer="L-BFGS-B"):
    msgs, results = [], []
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if x.size == 0 or y.size == 0:
        return pd.DataFrame(columns=["name","params","loglik","aic","tail_dep_L","tail_dep_U"]), ["Pas de données."]

    if HAS_COPULAFURTIF:
        fitter = CopulaFitter()
        candidates = [
            ("Gaussian",  CopulaType.GAUSSIAN),
            ("Student-t", CopulaType.STUDENT),
            ("Clayton",   CopulaType.CLAYTON),
            ("Gumbel",    CopulaType.GUMBEL),
            ("Frank",     CopulaType.FRANK),
            ("Galambos",  CopulaType.GALAMBOS),
            ("Joe",       CopulaType.JOE),
            ("Plackett",  CopulaType.PLACKETT),
        ]

        for name, ctype in candidates:
            try:
                cop = CopulaFactory.create(ctype)

                if method == "tau":
                    theta = fitter.fit_tau(data=(x, y), copula=cop)
                    ll = float(getattr(cop, "log_likelihood_", np.nan))
                else:
                    out = fitter.fit_quick(
                        data=(x, y),
                        copula=cop,
                        mode="cmle",
                        maxiter=maxiter,
                        optimizer=optimizer,
                        return_metrics=True,
                    )
                    theta = out["theta"]
                    ll = float(out["loglik"])

                # tail dep si dispo
                try: tdL = float(cop.LTDC())
                except Exception: tdL = np.nan
                try: tdU = float(cop.UTDC())
                except Exception: tdU = np.nan

                k = len(np.atleast_1d(theta))
                aic = (2*k - 2*ll) if np.isfinite(ll) else np.nan

                results.append(dict(
                    name=name,
                    params=np.asarray(theta, dtype=float),
                    loglik=ll,
                    aic=aic,
                    tail_dep_L=tdL,
                    tail_dep_U=tdU
                ))
            except Exception as e:
                msgs.append(f"{name} fit failed: {e}")

    df = pd.DataFrame(results).sort_values("aic", ascending=True).reset_index(drop=True) if results else \
         pd.DataFrame(columns=["name","params","loglik","aic","tail_dep_L","tail_dep_U"])
    return df, msgs