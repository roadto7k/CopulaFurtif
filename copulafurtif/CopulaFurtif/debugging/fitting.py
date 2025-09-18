import numpy as np
import pandas as pd

from CopulaFurtif.core.copulas.application.services.fit_copula import CopulaFitter
from CopulaFurtif.core.copulas.domain.estimation.tail_dependance import huang_lambda


def aic_val(loglik, k):
    return 2 * k - 2 * loglik

def _maybe_pseudo_obs(u, v):
    """If u,v are not in (0,1), convert to pseudo-obs ranks; else pass-through."""
    u = np.asarray(u).ravel()
    v = np.asarray(v).ravel()
    if u.size == 0 or v.size == 0:
        return u, v
    if np.min(u) < 0.0 or np.max(u) > 1.0 or np.min(v) < 0.0 or np.max(v) > 1.0:
        n = len(u)
        def ranks(x):
            return (np.argsort(np.argsort(x)) + 1) / (n + 1.0)
        return ranks(u), ranks(v)
    return u, v

def fit_copulas(u, v):
    msgs, results = [], []

    # ensure 1d arrays; if not uniform, convert to pseudo-obs
    u, v = _maybe_pseudo_obs(u, v)
    if u.size == 0 or v.size == 0:
        return pd.DataFrame(columns=['name','params','loglik','aic',
                                     'lambdaL_emp','lambdaU_emp','lambdaL_model','lambdaU_model']), \
               ["No copula data."]

    # candidates (FIXED: Joe/Galambos were swapped)
    candidates = [
        ('Gaussian',  CopulaType.GAUSSIAN),
        ('Student-t', CopulaType.STUDENT),
        ('Clayton',   CopulaType.CLAYTON),
        ('Gumbel',    CopulaType.GUMBEL),
        ('Frank',     CopulaType.FRANK),
        ('Galambos',  CopulaType.GALAMBOS),
        ('Joe',       CopulaType.JOE),
        ('Plackett',  CopulaType.PLACKETT),
    ]

    fitter = CopulaFitter()

    for name, ctype in candidates:
        try:
            cop = CopulaFactory.create(ctype)

            # try the new robust CMLE API (quick + return_metrics). fallback if older signature.
            try:
                fit_out = fitter.fit_cmle([u, v], copula=cop, quick=True, return_metrics=True)
                if fit_out is None:
                    raise RuntimeError("fit_cmle returned None")
                if len(fit_out) == 3:
                    fitted_params, loglik, extras = fit_out
                else:
                    fitted_params, loglik = fit_out
                    extras = {}
            except TypeError:
                # older signature
                fitted_params, loglik = fitter.fit_cmle([u, v], copula=cop)
                extras = {}

            cop.set_parameters(np.array(fitted_params))

            # theoretical tails from model
            try:
                params = cop.get_parameters()
                lambdaL_model = float(cop.LTDC(params))
                lambdaU_model = float(cop.UTDC(params))
            except Exception:
                lambdaL_model = np.nan
                lambdaU_model = np.nan

            # empirical tails via Huang (use extras if already computed)
            if isinstance(extras, dict) and "lambdaU_huang" in extras:
                lambdaU_emp = float(extras.get("lambdaU_huang"))
                lambdaL_emp = float(extras.get("lambdaL_huang"))
            else:
                lambdaU_emp = float(huang_lambda(u, v, side="upper"))
                lambdaL_emp = float(huang_lambda(u, v, side="lower"))

            k = len(np.atleast_1d(fitted_params))
            results.append({
                'name': name,
                'params': np.array(fitted_params, dtype=float),
                'loglik': float(loglik),
                'aic': float(aic_val(loglik, k)),
                'lambdaL_emp': lambdaL_emp, 'lambdaU_emp': lambdaU_emp,
                'lambdaL_model': lambdaL_model, 'lambdaU_model': lambdaU_model
            })

        except Exception as e:
            msgs.append(f"{name} fit failed: {e}")

    if not results:
        return pd.DataFrame(columns=['name','params','loglik','aic',
                                     'lambdaL_emp','lambdaU_emp','lambdaL_model','lambdaU_model']), msgs

    df = pd.DataFrame(results)
    df = df.sort_values("aic", kind="mergesort").reset_index(drop=True)
    return df, msgs

if __name__ == "__main__":
    from CopulaFurtif.core.copulas.domain.copula_type import CopulaType
    from CopulaFurtif.core.copulas.domain.factories.copula_factory import CopulaFactory

    def _sample_uv(ctype, params, n=5000, seed=123):
        """Sampling helper with rng fallback (some sample() impls don’t take rng)."""
        cop = CopulaFactory.create(ctype)
        try:
            rng = np.random.default_rng(seed)
            uv = cop.sample(n, param=params, rng=rng)
        except TypeError:
            uv = cop.sample(n, param=params)
        return uv[:, 0], uv[:, 1]

    # === choose a few ground-truth scenarios to sanity-check ===
    tests = [
        ("Gaussian ρ=0.6",      CopulaType.GAUSSIAN,  [0.60]),
        ("Student-t ρ=0.65, ν=7", CopulaType.STUDENT,   [0.65, 7.0]),
        ("Clayton θ=3",         CopulaType.CLAYTON,    [3.0]),
        ("Gumbel θ=2.2",        CopulaType.GUMBEL,     [2.2]),
        ("Frank θ=5",           CopulaType.FRANK,      [5.0]),
        ("Galambos θ=2.5",      CopulaType.GALAMBOS,   [2.5]),
        ("Joe θ=2.3",           CopulaType.JOE,        [2.3]),
        ("Plackett θ=4",        CopulaType.PLACKETT,   [4.0]),
    ]

    all_results = []
    all_msgs = []

    for label, ctype, true_params in tests:
        print("\n" + "=" * 80)
        print(f"TEST: {label}  |  true_params={true_params}")
        print("-" * 80)

        u, v = _sample_uv(ctype, true_params, n=5000, seed=123)
        df, msgs = fit_copulas(u, v)

        # keep for global view
        all_results.append((label, df))
        all_msgs.extend(msgs)

        # pretty print top-3 by AIC
        if df is not None and len(df) > 0:
            print(df.loc[:, ["name", "aic", "loglik", "lambdaL_emp", "lambdaU_emp", "lambdaL_model", "lambdaU_model"]]
                    .head(3)
                    .to_string(index=False, float_format=lambda x: f"{x:,.6f}"))
            print("\nBest by AIC:", df.iloc[0]["name"])
        else:
            print("No fit results.")

        if msgs:
            print("\nNotes/Warnings:")
            for m in msgs:
                print(" -", m)

    print("\n" + "#" * 80)
    print("GLOBAL SUMMARY (best AIC per test)")
    print("#" * 80)
    for label, df in all_results:
        best = df.iloc[0]["name"] if df is not None and len(df) else "—"
        print(f"{label:25s} -> {best}")

    if all_msgs:
        print("\nCollected warnings:")
        for m in all_msgs:
            print(" -", m)
