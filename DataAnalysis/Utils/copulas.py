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

def pseudo_obs(x: pd.Series) -> np.ndarray:
    """Ranks/(n+1) -> U(0,1) pseudo-observations."""
    vals = pd.Series(x).dropna().values
    n = len(vals)
    if n == 0:
        return np.array([])
    ranks = pd.Series(vals).rank(method="average").to_numpy()
    return ranks / (n + 1.0)

def aic_val(loglik, k): return 2*k - 2*loglik

#HERE : important ->
def fit_copulas(u, v):
    from scipy.optimize import minimize
    msgs, results = [], []
    u = np.asarray(u).reshape(-1, 1)
    v = np.asarray(v).reshape(-1, 1)
    if u.size == 0 or v.size == 0:
        return pd.DataFrame(columns=['name','params','loglik','aic','tail_dep_L','tail_dep_U']), ["Pas de données copule."]
    data = np.hstack([u, v])

    if HAS_COPULAFURTIF:
        candidates = [
            ('Gaussian', CopulaType.GAUSSIAN),
            ('Student-t', CopulaType.STUDENT),
            ('Clayton', CopulaType.CLAYTON),
            ('Gumbel', CopulaType.GUMBEL),
            ('Frank', CopulaType.FRANK),
            ('Galambos', CopulaType.GALAMBOS),
            ('Joe', CopulaType.JOE),
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

    elif HAS_SM_COPULA:
        def fit_sm(name, cls):
            try:
                lname = name.lower()
                if lname.startswith('gaussian') or lname.startswith('student'):
                    rho0 = 0.0
                    if 'student' in lname:
                        df0 = 5.0
                        def nll(x):
                            rho, df = np.tanh(x[0]), 2.1 + np.exp(x[1])
                            c = cls(rho, df=df)
                            return -np.sum(c.logpdf(data))
                        res = minimize(nll, x0=np.array([np.arctanh(rho0+1e-6), np.log(df0-2.1+1e-6)]), method='Nelder-Mead')
                        rho_hat, df_hat = np.tanh(res.x[0]), 2.1 + np.exp(res.x[1])
                        cop_hat = cls(rho_hat, df=df_hat)
                        ll = np.sum(cop_hat.logpdf(data))
                        arg = -np.sqrt((df_hat + 1.0) * (1.0 - rho_hat) / (1.0 + rho_hat))
                        lam = 2.0 * student_t.cdf(arg, df=df_hat + 1.0)
                        tdL = tdU = float(lam)
                        return dict(name=name, params=np.array([rho_hat, df_hat]), loglik=float(ll), aic=float(aic_val(ll, 2)),
                                    tail_dep_L=tdL, tail_dep_U=tdU)
                    else:
                        def nll(x):
                            rho = np.tanh(x[0])
                            c = cls(rho)
                            return -np.sum(c.logpdf(data))
                        res = minimize(nll, x0=np.array([np.arctanh(rho0+1e-6)]), method='Nelder-Mead')
                        rho_hat = np.tanh(res.x[0])
                        cop_hat = cls(rho_hat)
                        ll = np.sum(cop_hat.logpdf(data))
                        return dict(name=name, params=np.array([rho_hat]), loglik=float(ll), aic=float(aic_val(ll, 1)),
                                    tail_dep_L=0.0, tail_dep_U=0.0)
                else:
                    th0 = 1.0
                    def nll(x):
                        theta = 1e-6 + np.exp(x[0])
                        c = cls(theta)
                        return -np.sum(c.logpdf(data))
                    res = minimize(nll, x0=np.array([np.log(th0)]), method='Nelder-Mead')
                    th_hat = 1e-6 + np.exp(res.x[0])
                    cop_hat = cls(th_hat)
                    ll = np.sum(cop_hat.logpdf(data))
                    tdL = tdU = 0.0
                    if lname.startswith('clayton'):
                        tdL, tdU = 2**(-1/th_hat), 0.0
                    elif lname.startswith('gumbel'):
                        tdL, tdU = 0.0, 2 - 2**(1/th_hat)
                    return dict(name=name, params=np.array([th_hat]), loglik=float(ll), aic=float(aic_val(ll, 1)),
                                tail_dep_L=float(tdL), tail_dep_U=float(tdU))
            except Exception as e:
                return None
        fams = [
            ('Gaussian', GaussianCopula),
            ('Student-t', StudentTCopula),
            ('Clayton', ClaytonCopula),
            ('Gumbel', GumbelCopula),
            ('Frank', FrankCopula),
        ]
        for name, cls in fams:
            out = fit_sm(name, cls)
            if out is not None:
                results.append(out)
    else:
        msgs.append("Aucun backend copule disponible (installez CopulaFurtif ou statsmodels>=0.13).")

    df = pd.DataFrame(results).sort_values('aic', ascending=True).reset_index(drop=True) if results else \
         pd.DataFrame(columns=['name','params','loglik','aic','tail_dep_L','tail_dep_U'])
    return df, msgs