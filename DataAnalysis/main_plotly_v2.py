import matplotlib
matplotlib.use('Agg')

import os, io, base64

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, dash_table, no_update, callback_context
import plotly.graph_objs as go
import plotly.express as px

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from scipy import stats
from scipy.stats import norm, t, cauchy, kendalltau
import matplotlib.pyplot as plt

from DataAnalysis.config import DATA_PATH

HAS_COPULAFURTIF = False
HAS_SM_COPULA = False
try:
    from CopulaFurtif.copulas import CopulaFactory, CopulaType
    from CopulaFurtif.copulas import CopulaDiagnostics, CopulaFitter
    HAS_COPULAFURTIF = True
except Exception:
    pass

try:
    from statsmodels.distributions.copula.api import (
        GaussianCopula, StudentTCopula, ClaytonCopula, FrankCopula, GumbelCopula
    )
    HAS_SM_COPULA = True
except Exception:
    pass

REFERENCE_ASSET = 'BTCUSDT'
MAX_HEATMAP_COINS = 20
ROLLING_WIN = 200
UIREVISION_LOCK = "freeze"  # pb remise à 0 de y

def load_all_prices():
    data = {}
    for file in os.listdir(DATA_PATH):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(DATA_PATH, file), index_col=0, parse_dates=True)
            if 'close' in df.columns:
                data[file[:-4]] = df['close']
            else:
                lower = {c.lower(): c for c in df.columns}
                if 'close' in lower:
                    data[file[:-4]] = df[lower['close']]
    return pd.DataFrame(data).sort_index()

prices = load_all_prices()
if REFERENCE_ASSET not in prices.columns:
    raise ValueError(f"{REFERENCE_ASSET} absent des données chargées.")
PAIRS = [(REFERENCE_ASSET, coin) for coin in prices.columns if coin != REFERENCE_ASSET]
coins_list = [c for c in prices.columns if c != REFERENCE_ASSET]

def compute_beta(x, y):
    x, y = x.align(y, join='inner')
    mask = (~x.isna()) & (~y.isna()) & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 30 or np.std(y[mask]) < 1e-12:
        return np.nan
    return np.polyfit(y[mask], x[mask], 1)[0]

def compute_spread(reference, coin):
    beta = compute_beta(reference, coin)
    spread = reference - beta * coin if np.isfinite(beta) else pd.Series(index=reference.index, dtype=float)
    return spread.dropna(), beta

def run_adf_test(series):
    x = series.dropna().values
    if len(x) < 30 or np.std(x) < 1e-12:
        return np.nan, 1.0, {}
    result = adfuller(x, autolag='AIC')
    stat, pvalue, _, _, crit, _ = result
    return stat, pvalue, crit

#TT ca important aussi HERE
def kss_test(series):
    x = np.array(series.dropna(), dtype=float)
    if len(x) < 40:
        return np.nan, -1.92
    dx = np.diff(x)
    x_lag = x[:-1]
    y = dx
    z = x_lag**3
    try:
        beta = np.linalg.lstsq(z[:, None], y, rcond=None)[0][0]
        res = y - z * beta
        s2 = np.sum(res**2) / max(len(y) - 1, 1)
        se = np.sqrt(s2 / np.sum(z**2))
        t_stat = beta / (se if se > 0 else np.nan)
    except Exception:
        t_stat = np.nan
    return t_stat, -1.92  # environ 10%

def johansen_stat(x, y):
    arr = pd.concat([x, y], axis=1).dropna().values
    if arr.shape[0] < 40:
        return np.nan, np.nan
    result = coint_johansen(arr, det_order=0, k_ar_diff=1)
    trace_stat = float(result.lr1[0])
    crit_val = float(result.cvt[0,1])  # 5%
    return trace_stat, crit_val

def fit_distributions(series):
    s = series.dropna().values
    if len(s) < 30:
        return {
            "normal": {"params": (np.nan, np.nan), "aic": np.inf},
            "student": {"params": (np.nan, np.nan, np.nan), "aic": np.inf},
            "cauchy": {"params": (np.nan, np.nan), "aic": np.inf},
        }
    mu, sigma = norm.fit(s)
    loglik_norm = np.sum(norm.logpdf(s, mu, sigma))
    aic_norm = 2*2 - 2*loglik_norm
    params_t = t.fit(s)
    loglik_t = np.sum(t.logpdf(s, *params_t))
    aic_t = 2*3 - 2*loglik_t
    params_cauchy = cauchy.fit(s)
    loglik_cauchy = np.sum(cauchy.logpdf(s, *params_cauchy))
    aic_cauchy = 2*2 - 2*loglik_cauchy
    return {
        "normal": {"params": (mu, sigma), "aic": aic_norm},
        "student": {"params": params_t, "aic": aic_t},
        "cauchy": {"params": params_cauchy, "aic": aic_cauchy}
    }

def plot_acf_pacf(series, nlags=40):
    fig_acf, ax_acf = plt.subplots(facecolor='#181818')
    plot_acf(series.dropna(), ax=ax_acf, lags=nlags, color='white')
    ax_acf.set_facecolor('#181818')
    ax_acf.tick_params(colors='white', which='both')
    ax_acf.title.set_color('white')
    ax_acf.yaxis.label.set_color('white')
    ax_acf.xaxis.label.set_color('white')
    for sp in ax_acf.spines.values():
        sp.set_color('white')
    fig_acf.tight_layout()

    fig_pacf, ax_pacf = plt.subplots(facecolor='#181818')
    plot_pacf(series.dropna(), ax=ax_pacf, lags=nlags, color='white', method='ywm')
    ax_pacf.set_facecolor('#181818')
    ax_pacf.tick_params(colors='white', which='both')
    ax_pacf.title.set_color('white')
    ax_pacf.yaxis.label.set_color('white')
    ax_pacf.xaxis.label.set_color('white')
    for sp in ax_pacf.spines.values():
        sp.set_color('white')
    fig_pacf.tight_layout()

    buf_acf, buf_pacf = io.BytesIO(), io.BytesIO()
    fig_acf.savefig(buf_acf, format="png", facecolor='#181818', bbox_inches='tight', dpi=150)
    fig_pacf.savefig(buf_pacf, format="png", facecolor='#181818', bbox_inches='tight', dpi=150)
    buf_acf.seek(0); buf_pacf.seek(0)
    acf_img = base64.b64encode(buf_acf.read()).decode()
    pacf_img = base64.b64encode(buf_pacf.read()).decode()
    plt.close(fig_acf); plt.close(fig_pacf)
    return acf_img, pacf_img

def plot_qq(series, dist, params):
    fig, ax = plt.subplots(facecolor='#181818')
    s = series.dropna().values.astype(float)

    if dist == 'normal':
        mu, sigma = params
        z = (s - mu) / (sigma if sigma > 0 else 1.0)
        stats.probplot(z, dist="norm", plot=ax)

    elif dist == 'student':
        df, loc, scale = params
        z = (s - loc) / (scale if scale > 0 else 1.0)
        stats.probplot(z, dist="t", sparams=(df,), plot=ax)

    elif dist == 'cauchy':
        loc, scale = params
        z = (s - loc) / (scale if scale > 0 else 1.0)
        stats.probplot(z, dist="cauchy", plot=ax)

    # --- mise en forme dark theme ---
    ax.set_facecolor('#181818')
    fig.patch.set_facecolor('#181818')
    ax.tick_params(colors='white', which='both')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_color('white')

    # recolor lines + légende
    lines = ax.get_lines()
    if len(lines) >= 2:
        lines[0].set_color('cyan')   # points
        lines[1].set_color('white')  # droite de référence
        leg = ax.legend(loc='best')
        for text in leg.get_texts():
            text.set_color('white')
        leg.get_frame().set_facecolor('#181818')
        leg.get_frame().set_edgecolor('white')

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor='#181818',
                bbox_inches='tight', dpi=150)
    buf.seek(0)
    qq_img = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return qq_img


def compute_kendall_all(prices, ref=REFERENCE_ASSET):
    tau_list = []
    ref_series = prices[ref]
    for coin in prices.columns:
        if coin == ref: continue
        spread, _ = compute_spread(ref_series, prices[coin])
        if spread.empty:
            continue
        tau, _ = kendalltau(spread.dropna(), ref_series.loc[spread.dropna().index])
        tau_list.append((ref, coin, tau if tau is not None else np.nan))
    tau_list = [x for x in tau_list if np.isfinite(x[2])]
    tau_list = sorted(tau_list, key=lambda x: -abs(x[2]))
    return tau_list

def pseudo_obs(x: pd.Series):
    x = pd.Series(x).dropna().values
    n = len(x)
    if n == 0: return np.array([])
    ranks = pd.Series(x).rank(method='average').values
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

def fig_uv_scatter(u, v, nmax=5000):
    if len(u) > nmax:
        idx = np.linspace(0, len(u)-1, nmax).astype(int)
        uu, vv = u[idx], v[idx]
    else:
        uu, vv = u, v
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=uu, y=vv, mode='markers', marker=dict(size=4), name='(u,v)'))
    fig.update_layout(template='plotly_dark', title='Pseudo-observations (u,v)',
                      xaxis_title='u', yaxis_title='v',
                      height=380, uirevision=UIREVISION_LOCK)
    return fig

def fig_empirical_copula(u, v, nbins=30):
    hist, xe, ye = np.histogram2d(u, v, bins=nbins, range=[[0,1],[0,1]], density=True)
    fig = go.Figure(data=go.Heatmap(z=hist.T, x=xe, y=ye, coloraxis='coloraxis'))
    fig.update_layout(template='plotly_dark', title='Empirical copula density',
                      xaxis_title='u', yaxis_title='v',
                      coloraxis=dict(colorscale='Viridis'),
                      height=380, uirevision=UIREVISION_LOCK)
    return fig

def halflife_ar1(s: pd.Series):
    """Half-life (jours) via AR(1): S_t = phi S_{t-1} + e."""
    s = pd.Series(s).dropna()
    if len(s) < 50:
        return np.nan, np.nan
    y = s.iloc[1:].values
    x = s.shift(1).dropna().values
    if np.std(x) < 1e-12:
        return np.nan, np.nan
    phi = np.polyfit(x, y, 1)[0]
    if not np.isfinite(phi) or phi <= 0 or phi >= 1:
        return phi, np.nan
    hl = -np.log(2) / np.log(phi)
    return phi, hl

def screen_coins_vs_ref(prices: pd.DataFrame, ref: str, alpha: float = 0.1, min_len: int = 200):
    rows = []
    spreads = {}

    ref_series = prices[ref].dropna()

    for coin in prices.columns:
        if coin == ref:
            continue

        s, beta = compute_spread(ref_series, prices[coin])
        s = s.dropna()

        if len(s) < min_len or np.std(s) < 1e-12:
            rows.append(dict(
                coin=coin, beta=beta, n=len(s),
                adf_stat=np.nan, adf_pvalue=np.nan,
                kss_stat=np.nan, johansen_trace=np.nan,
                kendall_tau=np.nan, accepted=False
            ))
            continue

        try:
            adf_stat, adf_p, _ = run_adf_test(s)
        except Exception:
            adf_stat, adf_p = np.nan, np.nan

        try:
            kss_stat, _ = kss_test(s)
        except Exception:
            kss_stat = np.nan

        try:
            x = ref_series.loc[s.index]
            y = prices[coin].dropna().loc[s.index]
            joh_trace, joh_crit = johansen_stat(x, y)
        except Exception:
            joh_trace = np.nan

        # Kendall τ entre spread et ref (a voir)
        try:
            tau, _ = kendalltau(s, ref_series.loc[s.index])
        except Exception:
            tau = np.nan

        accepted = (adf_p < alpha) if np.isfinite(adf_p) else False

        rows.append(dict(
            coin=coin, beta=beta, n=len(s),
            adf_stat=adf_stat, adf_pvalue=adf_p,
            kss_stat=kss_stat, johansen_trace=joh_trace,
            kendall_tau=tau, accepted=accepted
        ))
        spreads[coin] = s

    cols = ['coin','beta','n','adf_stat','adf_pvalue','kss_stat','johansen_trace','kendall_tau','accepted']
    summary = pd.DataFrame(rows, columns=cols)

    if not summary.empty:
        summary['abs_tau'] = summary['kendall_tau'].abs()
        summary = summary.sort_values(by='abs_tau', ascending=False, na_position='last').reset_index(drop=True)
    else:
        summary = pd.DataFrame(columns=cols + ['abs_tau'])

    return summary, spreads

def _pseudo_obs_aligned(s1: pd.Series, s2: pd.Series):
    s1 = pd.Series(s1).dropna()
    s2 = pd.Series(s2).dropna()
    s1, s2 = s1.align(s2, join='inner')
    n = len(s1)
    if n == 0:
        return np.array([]), np.array([])
    u = s1.rank(method='average').to_numpy() / (n + 1.0)
    v = s2.rank(method='average').to_numpy() / (n + 1.0)
    return u, v

def candidate_pairs_from_top(summary: pd.DataFrame,
                             spreads: dict[str, pd.Series],
                             top_k: int = 10,
                             q_tail: float = 0.05,
                             min_len_pair: int = 150) -> pd.DataFrame:
    """
    Sélectionne les paires parmi les coins ACCEPTÉS (ADF), prend les Top-K (|tau|).
    Evaluatiuon des paires via via Kendall τ(s_a, s_b) et tail dependance empiriques.
    Retourne un DataFrame trié par 'pair_score'.
    """
    if summary is None or summary.empty:
        return pd.DataFrame(columns=[
            'coin_a','coin_b','n','tau_a','tau_b','kendall_uv','lambda_L','lambda_U','pair_score'
        ])

    top = summary[summary['accepted'] == True].copy()
    if top.empty:
        return pd.DataFrame(columns=[
            'coin_a','coin_b','n','tau_a','tau_b','kendall_uv','lambda_L','lambda_U','pair_score'
        ])

    if 'abs_tau' not in top.columns:
        top['abs_tau'] = top['kendall_tau'].abs()

    top = top.sort_values(by='abs_tau', ascending=False, na_position='last').head(int(top_k))

    coins = top['coin'].tolist()
    if len(coins) < 2:
        return pd.DataFrame(columns=[
            'coin_a','coin_b','n','tau_a','tau_b','kendall_uv','lambda_L','lambda_U','pair_score'
        ])

    tau_map = top.set_index('coin')['kendall_tau'].to_dict()

    rows = []
    for a_idx in range(len(coins)):
        for b_idx in range(a_idx + 1, len(coins)):
            ca, cb = coins[a_idx], coins[b_idx]
            s1 = spreads.get(ca, pd.Series(dtype=float))
            s2 = spreads.get(cb, pd.Series(dtype=float))
            if s1.empty or s2.empty:
                continue
            s1, s2 = s1.align(s2, join='inner')
            if len(s1) < min_len_pair:
                continue

            try:
                tau_uv, _ = kendalltau(s1, s2)
            except Exception:
                tau_uv = np.nan

            u, v = _pseudo_obs_aligned(s1, s2)
            if u.size == 0:
                continue
            q = float(q_tail)
            ll = np.mean((u < q) & (v < q)) / q if q > 0 else np.nan  # λ_L^hat
            uu = np.mean((u > 1-q) & (v > 1-q)) / q if q > 0 else np.nan  # λ_U^hat

            score = (np.abs(tau_uv) if np.isfinite(tau_uv) else 0.0) + 0.5 * (
                (ll if np.isfinite(ll) else 0.0) + (uu if np.isfinite(uu) else 0.0)
            )

            rows.append(dict(
                coin_a=ca, coin_b=cb, n=len(u),
                tau_a=tau_map.get(ca, np.nan),
                tau_b=tau_map.get(cb, np.nan),
                kendall_uv=tau_uv,
                lambda_L=ll, lambda_U=uu,
                pair_score=score
            ))

    pairs = pd.DataFrame(rows, columns=[
        'coin_a','coin_b','n','tau_a','tau_b','kendall_uv','lambda_L','lambda_U','pair_score'
    ])

    if not pairs.empty:
        pairs = pairs.sort_values(by='pair_score', ascending=False, na_position='last').reset_index(drop=True)

    return pairs

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.index_string ='''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
/* Tabs style (DCC tabs) */
.tab--selected, .dash-tabs .tab--selected {
    background-color: #232323 !important;
    color: #ffffff !important;
    border-color: #2c2c2c !important;
    font-weight: bold;
}
.tab, .dash-tabs .tab {
    background-color: #101010 !important;
    color: #cccccc !important;
    border: 1px solid #222 !important;
}
.dash-tabs {
    border-bottom: 1px solid #444 !important;
}
/* Dropdown style */
.Select-control, .VirtualizedSelectOption {
    background-color: #232323 !important;
    color: #fff !important;
    border-color: #444 !important;
}
.Select-menu-outer, .Select-menu {
    background-color: #181818 !important;
    color: #fff !important;
}
.Select-placeholder, .Select-value-label, .Select-arrow-zone {
    color: #fff !important;
}
.Select-option.is-selected, .Select-option.is-focused {
    background: #333 !important;
    color: #fff !important;
}
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

server = app.server

app.layout = dbc.Container(fluid=True, children=[
    html.H1("Cointegration & Copula Dashboard", style={'textAlign': 'center'}),
    html.H5("Formation auto ⇒ filtering (ADF/KSS) ⇒ ranking ⇒ pairs ⇒ copula", style={'textAlign': 'center', "color": "#AAAAAA"}),

    dbc.Row([
        dbc.Col([
            html.Label("Pair (reference vs coin)", style={'color': 'white'}),
            dcc.Dropdown(
                id='pair-select',
                options=[{'label': f'{p[0]} vs {p[1]}', 'value': f'{p[0]}-{p[1]}'} for p in PAIRS],
                value=f'{REFERENCE_ASSET}-{PAIRS[0][1]}' if PAIRS else None,
                style={'backgroundColor': 'black', 'color': 'white'}
            ),
        ], width=4),

        dbc.Col([
            html.Label("Copula coins (spreads vs ref)", style={'color': 'white'}),
            dcc.Dropdown(id='cop-coin1', options=[{'label': c, 'value': c} for c in coins_list],
                         value=coins_list[0] if coins_list else None),
            dcc.Dropdown(id='cop-coin2', options=[{'label': c, 'value': c} for c in coins_list],
                         value=coins_list[1] if len(coins_list)>1 else None, style={'marginTop':'6px'}),
        ], width=4),

        dbc.Col([
            dbc.Alert(
                "Copula backend: " + ("CopulaFurtif ✅" if HAS_COPULAFURTIF else ("statsmodels ✅" if HAS_SM_COPULA else "None ❌")),
                color='secondary', is_open=True
            )
        ], width=4),
    ]),

    html.Br(),
    dcc.Tabs([

        dcc.Tab(label='Formation (scanner auto)', children=[
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Label("alpha ADF", style={'color':'#ddd'}),
                        dcc.Slider(id='alpha-adf', min=0.01, max=0.2, step=0.005, value=0.10,
                                   marks={0.05:'0.05',0.1:'0.10',0.15:'0.15',0.2:'0.20'}),
                        html.Br(),
                        html.Label("Min length (points)", style={'color':'#ddd'}),
                        dcc.Input(id='min-len', type='number', value=200, min=40, step=10, style={'width':'100%'}),
                        html.Br(),html.Br(),
                        html.Label("Top-K coins accepted", style={'color':'#ddd'}),
                        dcc.Slider(id='top-k', min=4, max=30, step=1, value=12,
                                   marks={4:'4',8:'8',12:'12',20:'20',30:'30'}),
                        html.Br(),
                        html.Label("q_tail (empirical tail dep)", style={'color':'#ddd'}),
                        dcc.Slider(id='q-tail', min=0.80, max=0.99, step=0.01, value=0.95,
                                   marks={0.80:'0.80',0.9:'0.90',0.95:'0.95',0.99:'0.99'}),
                        html.Br(),
                        dbc.Button("Run scan", id='btn-run-scan', color='primary', className='me-2'),
                        html.Span(id='scan-status', style={'marginLeft':'10px', 'color':'#aaa'})
                    ], style={'background':'#222','padding':'12px','borderRadius':'8px'})
                ], width=3),

                dbc.Col([
                    html.H5("Summary (coins vs reference)", style={'color':'#eee'}),
                    dash_table.DataTable(
                        id='tbl-summary',
                        columns=[],
                        data=[],
                        style_table={'overflowX': 'auto', 'maxHeight':'420px'},
                        style_header={'backgroundColor': '#222', 'color': 'white', 'fontWeight':'bold'},
                        style_data={'backgroundColor': '#111', 'color': 'white'},
                        filter_action='native', sort_action='native', page_size=10
                    )
                ], width=5),

                dbc.Col([
                    html.H5("Candidate pairs (among Top-K)", style={'color':'#eee'}),
                    dash_table.DataTable(
                        id='tbl-pairs',
                        columns=[],
                        data=[],
                        style_table={'overflowX': 'auto', 'maxHeight':'420px'},
                        style_header={'backgroundColor': '#222', 'color': 'white', 'fontWeight':'bold'},
                        style_data={'backgroundColor': '#111', 'color': 'white'},
                        sort_action='native', page_size=10
                    ),
                    html.Br(),
                    dbc.Button("↪️ Set Copula pickers from selected row", id='btn-set-copula', color='info'),
                    html.Div(id='pairs-note', style={'color':'#aaa','marginTop':'8px'})
                ], width=4),
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='adf-hist', config={"displayModeBar": False})
                ], width=6),
                dbc.Col([
                    dcc.Graph(id='kendall-hist', config={"displayModeBar": False})
                ], width=6),
            ])
        ]),

        dcc.Tab(label='General', children=[
            dbc.Row([
                dbc.Col(dcc.Graph(id='price-plot', config={"displayModeBar": False}, style={"height": "380px"}), width=6),
                dbc.Col(dcc.Graph(id='spread-plot', config={"displayModeBar": False}, style={"height": "380px"}), width=6),
            ]),
            dbc.Row([
                dbc.Col(html.Div(id='beta-info', style={'color': '#17BECF', 'fontSize': 18}), width=3),
                dbc.Col(dcc.Graph(id='zscore-plot', config={"displayModeBar": False}, style={"height": "340px"}), width=9),
            ]),
        ]),

        dcc.Tab(label='Advanced cointegration', children=[
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.H5("Test ADF", style={'color': 'white'}),
                    html.Div(id='adf-results', style={'color': 'white'}),
                    dcc.Graph(id='adf-plot', config={"displayModeBar": False}, style={"height": "360px"}),
                ], width=4),
                dbc.Col([
                    html.H5("Test KSS (non linear)", style={'color': 'white'}),
                    html.Div(id='kss-results', style={'color': 'white'}),
                    dcc.Graph(id='kss-plot', config={"displayModeBar": False}, style={"height": "360px"}),
                ], width=4),
                dbc.Col([
                    html.H5("Test Johansen", style={'color': 'white'}),
                    html.Div(id='johansen-results', style={'color': 'white'}),
                    html.Hr(),
                    html.H6("ADF Rolling (win=200)", style={'color': 'white'}),
                    dcc.Graph(id='adf-rolling', config={"displayModeBar": False}, style={"height": "250px"}),
                ], width=4),
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.H5("Heatmap ADF sur sous-ensemble (max 20)", style={'color': 'white'}),
                    dcc.Graph(id='adf-heatmap', config={"displayModeBar": False}, style={"height": "420px"}),
                ], width=12)
            ])
        ]),

        dcc.Tab(label='Distribution, Stationnarité', children=[
            html.Br(),
            dbc.Row([
                dbc.Col(dcc.Graph(id='hist-plot', config={"displayModeBar": False}, style={"height": "360px"}), width=6),
                dbc.Col(html.Img(id='qq-plot', style={'width': '100%', 'height': '360px', 'objectFit': 'contain'}), width=6),
            ]),
            dbc.Row([
                dbc.Col(html.Img(id='acf-plot', style={'width': '100%', 'height': '360px', 'objectFit': 'contain'}), width=6),
                dbc.Col(html.Img(id='pacf-plot', style={'width': '100%', 'height': '360px', 'objectFit': 'contain'}), width=6),
            ]),
        ]),

        dcc.Tab(label='Classement, scores', children=[
            html.H4("Classement des paires par Tau de Kendall", style={'color': 'white'}),
            html.Table(id='kendall-table', style={'color': 'white', 'width': '80%', "marginLeft": "10px"}),
        ]),

        dcc.Tab(label='Copula (fit & diagnostics)', children=[
            html.Br(),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='uv-scatter', config={"displayModeBar": False}, style={"height": "380px"}),
                    dcc.Graph(id='uv-empirical', config={"displayModeBar": False}, style={"height": "380px"}),
                ], width=6),
                dbc.Col([
                    html.H5("Fit candidate copulas", style={'color':'white'}),
                    html.Div(id='fit-messages', style={'color':'#FFDD57'}),
                    dash_table.DataTable(
                        id='tbl-copula',
                        columns=[{"name": c, "id": c} for c in ['name','params','loglik','aic','tail_dep_L','tail_dep_U']],
                        data=[],
                        style_table={'overflowX': 'auto'},
                        style_header={'backgroundColor': '#222', 'color': 'white', 'fontWeight':'bold'},
                        style_data={'backgroundColor': '#111', 'color': 'white', 'whiteSpace': 'normal', 'height': 'auto'},
                        sort_action='native', page_size=8
                    ),
                    html.Br(),
                    html.Div(id='best-copula-card')
                ], width=6),
            ]),
        ]),

        dcc.Tab(label='Maths & Explications', children=[
            dcc.Markdown('''
#### Cointégration — Formules

Soient deux séries de prix \\( P_1, P_2 \\).  
On estime le spread stationnaire :
\\[
S_t = P_{1, t} - \\beta P_{2, t}
\\]
avec
\\[
\\beta = \\arg\\min_{\\beta} \\sum_t (P_{1, t} - \\beta P_{2, t})^2
\\]

**ADF (Augmented Dickey-Fuller) :**
\\[
\\Delta S_t = \\rho S_{t-1} + \\sum_{i=1}^{p} \\gamma_i \\Delta S_{t-i} + \\varepsilon_t
\\]
- H0 : \\( \\rho = 0 \\) (non stationnaire), H1 : \\( \\rho < 0 \\)
- Seuil critique 10% : p-value < 0.1

**KSS (Kapetanios–Shin–Snell) :**
\\[
\\Delta S_t = \\delta (S_{t-1})^3 + \\varepsilon_t
\\]
- H0 : \\( \\delta = 0 \\) (racine unitaire), H1 : \\( \\delta < 0 \\)
- Seuil 10% : t < -1.92

**Johansen :**
\\[
\\mathbf{X}_t = [P_{1,t}, P_{2,t}]^T
\\]
Comparer le "trace statistic" à la valeur critique (5%).

**Kendall Tau :**
\\[
\\tau = \\frac{(\\text{concordants} - \\text{discordants})}{N(N-1)/2}
\\]

**Z-score du spread :**
\\[
z_t = \\frac{S_t - \\mu_S}{\\sigma_S}
\\]

**Copules (AIC) :** pseudo-observations \\(u=\\hat F(S_a), v=\\hat F(S_b)\\), log-vraisemblance \\(\\sum \\log c(u,v;\\theta)\\),  
\\(\\mathrm{AIC}=2k-2\\log L\\). Dépendance de queue : Clayton (\\(\\lambda_L>0\\)), Gumbel (\\(\\lambda_U>0\\)), Student (bilat).
''', mathjax=True, style={'color': 'white', 'backgroundColor': '#222', 'padding': '20px'})
        ])
    ])
], style={'backgroundColor': '#181818', 'color': "#1A1414"})

def empty_fig(msg="Pas de données"):
    return go.Figure(layout=dict(
        template='plotly_dark',
        title=msg,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[{
            "text": msg,
            "xref": "paper",
            "yref": "paper",
            "showarrow": False,
            "font": {"size": 18, "color": "red"},
            "x": 0.5, "y": 0.5
        }],
        height=340,
        uirevision=UIREVISION_LOCK
    ))

# tt les callbacks

@app.callback(
    Output('tbl-summary', 'data'),
    Output('tbl-summary', 'columns'),
    Output('tbl-pairs', 'data'),
    Output('tbl-pairs', 'columns'),
    Output('adf-hist', 'figure'),
    Output('kendall-hist', 'figure'),
    Output('cop-coin1', 'options'),
    Output('cop-coin2', 'options'),
    Output('scan-status', 'children'),
    Input('btn-run-scan', 'n_clicks'),
    State('alpha-adf', 'value'),
    State('min-len', 'value'),
    State('top-k', 'value'),
    State('q-tail', 'value'),
    prevent_initial_call=False
)
def run_scan(n_clicks, alpha, min_len, top_k, q_tail):
    summary, spreads = screen_coins_vs_ref(prices, REFERENCE_ASSET, alpha=float(alpha), min_len=int(min_len))
    pairs = candidate_pairs_from_top(summary, spreads, top_k=int(top_k), q_tail=float(q_tail))

    def cols(df): return [{"name": c, "id": c} for c in df.columns]
    sum_data = summary.to_dict('records') if not summary.empty else []
    sum_cols = cols(summary) if not summary.empty else []
    pairs_data = pairs.to_dict('records') if not pairs.empty else []
    pairs_cols = cols(pairs) if not pairs.empty else []

    if not summary.empty and 'adf_pvalue' in summary:
        fig_adf = (px.histogram(summary.dropna(subset=['adf_pvalue']), x='adf_pvalue', nbins=30,
                                title='Distribution des p-values ADF')
                   .update_layout(template='plotly_dark', bargap=0.05, uirevision=UIREVISION_LOCK))
    else:
        fig_adf = empty_fig("No ADF values")

    if not summary.empty and 'kendall_tau' in summary:
        dfh = summary.dropna(subset=['kendall_tau']).copy()
        dfh['abs_tau'] = dfh['kendall_tau'].abs()
        fig_tau = (px.histogram(dfh, x='abs_tau', nbins=30,
                                title='Distribution |Kendall τ| (vs ref)')
                .update_layout(template='plotly_dark', bargap=0.05, uirevision=UIREVISION_LOCK))
    else:
        fig_tau = empty_fig("No Kendall values")

    options = [{'label': c, 'value': c} for c in summary['coin'].tolist()] if not summary.empty else []
    status = f"Scanné: {len(summary)} coins | Acceptés: {int(summary['accepted'].sum()) if 'accepted' in summary else 0} | Paires candidates: {len(pairs)}"
    return sum_data, sum_cols, pairs_data, pairs_cols, fig_adf, fig_tau, options, options, status

@app.callback(
    Output('cop-coin1', 'value'),
    Output('cop-coin2', 'value'),
    Input('tbl-pairs', 'data'),
    Input('btn-set-copula', 'n_clicks'),
    State('tbl-pairs', 'active_cell'),
    prevent_initial_call=False
)
def set_copula_values(pairs_data, n_clicks, active_cell):
    # Quel composant a déclenché ?
    trig = callback_context.triggered[0]['prop_id'].split('.')[0] if callback_context.triggered else None

    if trig == 'btn-set-copula':
        if active_cell and pairs_data:
            r = active_cell.get('row')
            if r is not None and 0 <= r < len(pairs_data):
                row = pairs_data[r]
                return row.get('coin_a', no_update), row.get('coin_b', no_update)
        return no_update, no_update

    # Recalcule de pairs_data (après scan) -> prendre la première paire dispo
    if pairs_data and len(pairs_data) > 0:
        row0 = pairs_data[0]
        return row0.get('coin_a', no_update), row0.get('coin_b', no_update)

    # Sinon rien changer
    return no_update, no_update


@app.callback(
    Output('price-plot', 'figure'),
    Output('spread-plot', 'figure'),
    Output('beta-info', 'children'),
    Input('pair-select', 'value')
)
def update_price_spread(pair_value):
    if not pair_value:
        return empty_fig(), empty_fig(), "β (OLS): N/A"
    c1, c2 = pair_value.split('-')
    df = prices[[c1, c2]].dropna()
    spread, beta = compute_spread(df[c1], df[c2])

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=df.index, y=df[c1], name=c1, line={'color':'#18FFFF'}))
    fig_price.add_trace(go.Scatter(x=df.index, y=df[c2], name=c2, line={'color':'#FF6B81'}))
    fig_price.update_layout(
        template="plotly_dark",
        title="Asset prices", xaxis_title="Date", yaxis_title="Price",
        plot_bgcolor='#181818', paper_bgcolor='#181818',
        height=380, uirevision=UIREVISION_LOCK
    )

    fig_spread = go.Figure()
    fig_spread.add_trace(go.Scatter(x=spread.index, y=spread, name='Spread', line={'color':'#00E676'}))
    ystd = spread.std() if spread.size else 1.0
    fig_spread.update_yaxes(range=[-4*ystd, 4*ystd], fixedrange=True)
    fig_spread.update_layout(
        template="plotly_dark",
        title=f"Stationary spread: {c1} - β*{c2}", xaxis_title="Date", yaxis_title="Spread",
        plot_bgcolor='#181818', paper_bgcolor='#181818',
        height=380, uirevision=UIREVISION_LOCK
    )
    beta_txt = f"β (OLS): {beta:.4f}" if np.isfinite(beta) else "β (OLS): N/A"
    return fig_price, fig_spread, beta_txt

@app.callback(
    Output('adf-results', 'children'),
    Output('adf-plot', 'figure'),
    Output('kss-results', 'children'),
    Output('kss-plot', 'figure'),
    Output('johansen-results', 'children'),
    Output('adf-rolling', 'figure'),
    Output('adf-heatmap', 'figure'),
    Input('pair-select', 'value')
)
def update_coint_stats(pair_value):
    if not pair_value:
        dummy = empty_fig()
        return ("Pas assez de données", dummy, "Pas assez de données", dummy, "Pas assez de données", dummy, dummy)

    c1, c2 = pair_value.split('-')
    df = prices[[c1, c2]].dropna()
    spread, _ = compute_spread(df[c1], df[c2])
    spread = spread.dropna()

    if len(spread) < 30:
        dummy = empty_fig("Pas assez de données")
        return ("Pas assez de données", dummy, "Pas assez de données", dummy, "Pas assez de données", dummy, dummy)

    try:
        stat, pval, crit = run_adf_test(spread)
        adf_text = (
            f"Stat ADF : {stat:.3f} | p-value : {pval:.3f} | "
            f"Result : {'Stationary (cointegrated)' if pval < 0.1 else 'Non-stationary'}<br>"
            f"Critiques : { {k: f'{v:.2f}' for k,v in crit.items()} }"
        )
        spread_centered = spread - spread.mean()
        fig_adf = go.Figure()
        fig_adf.add_trace(go.Scatter(x=spread.index, y=spread_centered, name='Spread centré', line={'color':'#00E676'}))
        fig_adf.add_hline(y=0, line_dash='dash', line_color='red', annotation_text='Moyenne')
        ystd = spread_centered.std() if spread_centered.size else 1.0
        fig_adf.update_yaxes(range=[-4*ystd, 4*ystd], fixedrange=True)
        fig_adf.update_layout(
            template="plotly_dark",
            title="Spread (ADF stationarity)", plot_bgcolor='#181818', paper_bgcolor='#181818',
            yaxis_title="Spread (centered)", xaxis_title="Date",
            height=360, uirevision=UIREVISION_LOCK
        )
    except Exception as e:
        adf_text = f"Erreur calcul ADF: {e}"
        fig_adf = empty_fig("Erreur ADF")

    try:
        kss_stat, kss_crit = kss_test(spread)
        if pd.isna(kss_stat):
            kss_text = "Pas assez de données pour KSS"
            fig_kss = empty_fig("KSS N/A")
        else:
            verdict_kss = "Stationary (non-linear)" if kss_stat < kss_crit else "Non-stationary"
            kss_text = f"Stat KSS : {kss_stat:.3f} | Threshold {kss_crit} | Result : {verdict_kss}"
            dspread = np.diff(spread)
            dspread_centered = dspread - np.mean(dspread)
            fig_kss = go.Figure()
            fig_kss.add_trace(go.Scatter(x=spread.index[1:], y=dspread_centered, name='centered ΔSpread', line={'color':'#FFC300'}))
            fig_kss.add_hline(y=0, line_dash='dot', line_color='red')
            ystd2 = np.std(dspread_centered) if len(dspread_centered) else 1.0
            fig_kss.update_yaxes(range=[-4*ystd2, 4*ystd2], fixedrange=True)
            fig_kss.update_layout(
                template="plotly_dark", title="ΔSpread (KSS)", plot_bgcolor='#181818', paper_bgcolor='#181818',
                yaxis_title="centered ΔSpread", xaxis_title="Date", height=360, uirevision=UIREVISION_LOCK
            )
    except Exception as e:
        kss_text = f"Erreur KSS: {e}"
        fig_kss = empty_fig("Erreur KSS")

    try:
        johansen_statistic, johansen_crit = johansen_stat(df[c1], df[c2])
        if pd.isna(johansen_statistic):
            johansen_text = "Pas assez de données pour Johansen"
        else:
            verdict_j = "✅ Cointegration" if johansen_statistic > johansen_crit else "❌ No cointegration"
            johansen_text = f"Trace stat : {johansen_statistic:.2f} | Threshold 5% : {johansen_crit:.2f} | Result : {verdict_j}"
    except Exception as e:
        johansen_text = f"Erreur Johansen: {e}"

    try:
        if len(spread) < ROLLING_WIN:
            fig_rolling = empty_fig("Pas assez de points pour rolling ADF")
        else:
            def safe_adf(x):
                try:
                    if not np.all(np.isfinite(x)) or np.std(x) == 0 or len(x) < 20:
                        return np.nan
                    return adfuller(x, autolag='AIC')[1]
                except Exception:
                    return np.nan
            roll_adf = spread.rolling(ROLLING_WIN).apply(lambda a: safe_adf(a.values) if hasattr(a, 'values') else safe_adf(a), raw=False)
            fig_rolling = go.Figure()
            fig_rolling.add_trace(go.Scatter(x=roll_adf.index, y=roll_adf.values, name='p-value ADF rolling', line={'color':'#00BFFF'}))
            fig_rolling.add_hline(y=0.1, line_dash='dot', line_color='red', annotation_text='Seuil 10%')
            fig_rolling.update_yaxes(range=[0,1], fixedrange=True)
            fig_rolling.update_layout(template='plotly_dark', title='Rolling p-value ADF',
                                      plot_bgcolor='#181818', paper_bgcolor='#181818',
                                      height=250, uirevision=UIREVISION_LOCK)
    except Exception:
        fig_rolling = empty_fig("Erreur rolling ADF")

    try:
        cols = [REFERENCE_ASSET] + [c for c in prices.columns if c != REFERENCE_ASSET]
        stds = prices[cols].std().sort_values(ascending=False).index.tolist()
        keep = stds[:MAX_HEATMAP_COINS]
        coins = [c for c in keep]
        adf_pval_mat = np.full((len(coins), len(coins)), np.nan)
        for i, cA in enumerate(coins):
            for j, cB in enumerate(coins):
                if i==j: continue
                sp, _ = compute_spread(prices[cA], prices[cB])
                if sp.empty:
                    p = np.nan
                else:
                    try:
                        _, p, _ = run_adf_test(sp)
                    except Exception:
                        p = np.nan
                adf_pval_mat[i,j] = p
        fig_heatmap = px.imshow(
            adf_pval_mat,
            labels=dict(x="Coin", y="Coin", color="p-value ADF"),
            x=coins, y=coins, color_continuous_scale='bluered_r', aspect='auto'
        )
        fig_heatmap.update_layout(template='plotly_dark', title=f'ADF p-value heatmap (spreads) — subset {len(coins)}',
                                  plot_bgcolor='#181818', paper_bgcolor='#181818',
                                  height=420, uirevision=UIREVISION_LOCK)
    except Exception:
        fig_heatmap = empty_fig("Erreur heatmap ADF")

    return adf_text, fig_adf, kss_text, fig_kss, johansen_text, fig_rolling, fig_heatmap

@app.callback(
    Output('hist-plot', 'figure'),
    Output('qq-plot', 'src'),
    Output('acf-plot', 'src'),
    Output('pacf-plot', 'src'),
    Input('pair-select', 'value')
)
def update_dist(pair_value):
    if not pair_value:
        dummy = empty_fig()
        blank = "data:image/png;base64,"
        return dummy, blank, blank, blank
    c1, c2 = pair_value.split('-')
    df = prices[[c1, c2]].dropna()
    spread, _ = compute_spread(df[c1], df[c2])
    if spread.empty:
        dummy = empty_fig("Pas de spread")
        blank = "data:image/png;base64,"
        return dummy, blank, blank, blank

    fits = fit_distributions(spread)
    best_fit = min(fits, key=lambda k: fits[k]["aic"])
    x = np.linspace(spread.min(), spread.max(), 200)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=spread, nbinsx=40, histnorm='probability density', name='Spread', marker_color='#FFD700'))
    if np.isfinite(fits["normal"]["aic"]):
        fig.add_trace(go.Scatter(x=x, y=norm.pdf(x, *fits["normal"]["params"]), name='Normal', line={'color':'#00E676'}))
    if np.isfinite(fits["student"]["aic"]):
        fig.add_trace(go.Scatter(x=x, y=t.pdf(x, *fits["student"]["params"]), name='Student-t', line={'color':'#FF6B81'}))
    if np.isfinite(fits["cauchy"]["aic"]):
        fig.add_trace(go.Scatter(x=x, y=cauchy.pdf(x, *fits["cauchy"]["params"]), name='Cauchy', line={'color':'#18FFFF'}))
    fig.update_layout(
        template="plotly_dark",
        title=f"Histogramme du spread (Best: {best_fit}, AIC={fits[best_fit]['aic']:.1f})",
        plot_bgcolor='#181818', paper_bgcolor='#181818',
        height=360, uirevision=UIREVISION_LOCK
    )

    qq_img = plot_qq(spread, best_fit, fits[best_fit]["params"])
    acf_img, pacf_img = plot_acf_pacf(spread)
    return fig, f"data:image/png;base64,{qq_img}", f"data:image/png;base64,{acf_img}", f"data:image/png;base64,{pacf_img}"

@app.callback(
    Output('zscore-plot', 'figure'),
    Input('pair-select', 'value')
)
def update_zscore(pair_value):
    if not pair_value:
        return empty_fig()
    c1, c2 = pair_value.split('-')
    df = prices[[c1, c2]].dropna()
    spread, _ = compute_spread(df[c1], df[c2])
    if spread.std() == 0 or spread.empty:
        return empty_fig("Z-score N/A")
    z = (spread - spread.mean()) / spread.std()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=z.index, y=z, name='Z-score', line={'color':'#FFD700'}))
    for y, dash, col, txt in [(2,'dash','red',"+2σ"), (-2,'dash','blue',"-2σ"), (1,'dot','orange',"+1σ"), (-1,'dot','orange',"-1σ")]:
        fig.add_hline(y=y, line_dash=dash, line_color=col, annotation_text=txt)
    # 🔒 plage fixe pour éviter le "y qui grandit"
    fig.update_yaxes(range=[-4, 4], fixedrange=True)
    fig.update_layout(
        template="plotly_dark",
        title="Z-score du spread", xaxis_title="Date", yaxis_title="Z-score",
        plot_bgcolor='#181818', paper_bgcolor='#181818',
        height=340, uirevision=UIREVISION_LOCK
    )
    return fig

@app.callback(
    Output('kendall-table', 'children'),
    Input('pair-select', 'value')
)
def update_kendall_table(_):
    taus = compute_kendall_all(prices, ref=REFERENCE_ASSET)
    header = html.Tr([html.Th("Asset 1"), html.Th("Asset 2"), html.Th("Tau de Kendall")])
    rows = [header]
    for c1, c2, tau in taus[:15]:
        rows.append(html.Tr([html.Td(c1), html.Td(c2), html.Td(f"{tau:.3f}")]))
    return rows

@app.callback(
    Output('uv-scatter', 'figure'),
    Output('uv-empirical', 'figure'),
    Output('tbl-copula', 'data'),
    Output('fit-messages', 'children'),
    Output('best-copula-card', 'children'),
    Input('cop-coin1', 'value'),
    Input('cop-coin2', 'value')
)
def update_copula_views(c1, c2):
    if not c1 or not c2 or c1 == c2:
        empty = empty_fig("Sélectionnez deux coins distincts")
        return empty, empty, [], "Choisissez deux coins.", None

    s1, _ = compute_spread(prices[REFERENCE_ASSET], prices[c1])
    s2, _ = compute_spread(prices[REFERENCE_ASSET], prices[c2])
    s1, s2 = s1.align(s2, join='inner')
    if s1.empty or s2.empty:
        empty = empty_fig("Spreads introuvables")
        return empty, empty, [], "Spreads insuffisants.", None

    u, v = pseudo_obs(s1), pseudo_obs(s2)
    if len(u)==0 or len(v)==0:
        empty = empty_fig("Pseudo-observations vides")
        return empty, empty, [], "Impossible de former (u,v).", None

    fig_sc = fig_uv_scatter(u, v)
    fig_emp = fig_empirical_copula(u, v, nbins=35)

    df_fit, msgs = fit_copulas(u, v)
    msg_box = html.Ul([html.Li(m) for m in msgs]) if msgs else ""

    if not df_fit.empty:
        best = df_fit.iloc[0]
        tdL = best.get('tail_dep_L', np.nan)
        tdU = best.get('tail_dep_U', np.nan)
        tdL_txt = "{:.3f}".format(tdL) if isinstance(tdL, (int, float, np.floating)) else str(tdL)
        tdU_txt = "{:.3f}".format(tdU) if isinstance(tdU, (int, float, np.floating)) else str(tdU)
        card = dbc.Card([
            dbc.CardHeader("Best by AIC"),
            dbc.CardBody([
                html.H4(str(best['name'])),
                html.P(f"AIC = {best['aic']:.2f} | loglik = {best['loglik']:.2f}"),
                html.P(f"params = {np.array2string(np.atleast_1d(best['params']), precision=3)}"),
                html.P(f"Tail dep. λL={tdL_txt}, λU={tdU_txt}")
            ])
        ], color="dark", inverse=True)
    else:
        card = None

    return fig_sc, fig_emp, (df_fit.to_dict('records') if not df_fit.empty else []), msg_box, card

if __name__ == "__main__":
    app.run(debug=True, port=8042)
