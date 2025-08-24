import numpy as np
import pandas as pd

from CopulaFurtif.core.copulas.application.services.fit_copula import CopulaFitter
from CopulaFurtif.core.copulas.domain.copula_type import CopulaType
from CopulaFurtif.core.copulas.domain.factories.copula_factory import CopulaFactory
from DataAnalysis.main_plotly import aic_val

def aic_val(loglik, k): return 2*k - 2*loglik

def fit_copulas(u, v):
    from scipy.optimize import minimize
    msgs, results = [], []
    u = np.asarray(u).reshape(-1, 1)
    v = np.asarray(v).reshape(-1, 1)
    if u.size == 0 or v.size == 0:
        return pd.DataFrame(columns=['name','params','loglik','aic','tail_dep_L','tail_dep_U']), ["Pas de données copule."]
    data = np.hstack([u, v])

    candidates = [
        ('Gaussian', CopulaType.GAUSSIAN),
        ('Student-t', CopulaType.STUDENT),
        ('Clayton', CopulaType.CLAYTON),
        ('Gumbel', CopulaType.GUMBEL),
        ('Frank', CopulaType.FRANK),
        ('Galambos', CopulaType.JOE),
        ('Joe', CopulaType.GALAMBOS),
        ('Plackett', CopulaType.PLACKETT),
    ]
    for name, ctype in candidates:
        try:
            cop = CopulaFactory.create(ctype)
            fitted_params, loglik = CopulaFitter().fit_cmle([u.ravel(), v.ravel()], copula=cop) # todo to adapt later with ifm or mle
            cop.set_parameters(np.array(fitted_params))
            try:
                tdL = cop.LTDC()
                tdU = cop.UTDC()
            except Exception:
                tdL = tdU = np.nan
            results.append({
                'name': name,
                'params': np.array(fitted_params, dtype=float),
                'loglik': float(loglik),
                'aic': float(aic_val(loglik, len(np.atleast_1d(fitted_params)))),
                'tail_dep_L': tdL, 'tail_dep_U': tdU
            })
        except Exception as e:
            msgs.append(f"{name} (CopulaFurtif) fit failed: {e}")
