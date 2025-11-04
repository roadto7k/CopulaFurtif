import math
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from CopulaFurtif.core.copulas.domain.copula_type import CopulaType
from CopulaFurtif.core.copulas.domain.factories.copula_factory import CopulaFactory
from CopulaFurtif.core.copulas.application.services.fit_copula import CopulaFitter

# ---------- helpers: empirical metrics ----------

def empirical_beta(u, v):
    u = np.asarray(u); v = np.asarray(v)
    concord = np.mean(((u > 0.5) & (v > 0.5)) | ((u < 0.5) & (v < 0.5)))
    return 2.0 * concord - 1.0

def huang_lambda(u, v, side="upper", k=None):
    u, v = np.asarray(u), np.asarray(v)
    n = len(u)
    if k is None:
        k = int(np.sqrt(n))  # common rule-of-thumb
    if side == "upper":
        u_thresh = np.partition(u, n - k)[-k]
        v_thresh = np.partition(v, n - k)[-k]
        count = np.sum((u > u_thresh) & (v > v_thresh))
    else:
        u_thresh = np.partition(u, k)[k]
        v_thresh = np.partition(v, k)[k]
        count = np.sum((u < u_thresh) & (v < v_thresh))
    return count / max(1, k)

# ---------- theoretical helpers per family (only where trivial/closed) ----------

def tau_gaussian_or_student(rho):
    return (2.0 / np.pi) * np.arcsin(rho)

def beta_gaussian_or_student(rho):
    # for elliptical copulas (Gaussian,t) beta equals tau
    return (2.0 / np.pi) * np.arcsin(rho)

def metrics_theoretical(name, params, cop):
    """
    Return dict with theoretical metrics (tau, lambda_L, lambda_U, beta) for quick checks.
    Falls back to cop methods when convenient.
    """
    d = {"tau_th": None, "lambdaL_th": None, "lambdaU_th": None, "beta_th": None}

    # tails from copula if available
    try:
        d["lambdaL_th"] = float(cop.LTDC(params))
        d["lambdaU_th"] = float(cop.UTDC(params))
    except Exception:
        d["lambdaL_th"] = d["lambdaU_th"] = None

    # kendall's tau from copula if available
    try:
        d["tau_th"] = float(cop.kendall_tau(params))
    except Exception:
        d["tau_th"] = None

    # plug exact closed forms when faster/safer
    lname = name.lower()
    if lname.startswith("gaussian"):
        rho = float(params[0])
        d["tau_th"] = tau_gaussian_or_student(rho)
        d["beta_th"] = beta_gaussian_or_student(rho)
    elif lname.startswith("student"):
        rho = float(params[0])
        d["tau_th"] = tau_gaussian_or_student(rho)
        d["beta_th"] = beta_gaussian_or_student(rho)
    else:
        # try theoretical beta if method exists
        if hasattr(cop, "blomqvist_beta"):
            try:
                d["beta_th"] = float(cop.blomqvist_beta(params))
            except Exception:
                d["beta_th"] = None
        else:
            d["beta_th"] = None

    return d

# ---------- test bench using fit_tau (init-only) ----------

def test_fit_tau_init_only(n=10000, seed=12345):
    rng = np.random.default_rng(seed)
    fitter = CopulaFitter()

    cases = [
        # name, CopulaType, true_params
        ("Gaussian",  CopulaType.GAUSSIAN,  [0.60]),
        ("Student-t", CopulaType.STUDENT,   [0.65, 7.0]),
        ("Clayton",   CopulaType.CLAYTON,   [3.0]),
        ("Gumbel",    CopulaType.GUMBEL,    [2.2]),
        ("Frank",     CopulaType.FRANK,     [5.0]),
        ("Galambos",  CopulaType.GALAMBOS,  [2.5]),
        ("Joe",       CopulaType.JOE,       [2.3]),
        ("Plackett",  CopulaType.PLACKETT,  [4.0]),
    ]

    rows = []
    logs = []

    for name, ctype, true_params in cases:
        cop = CopulaFactory.create(ctype)

        # sample synthetic data in (0,1)^2
        try:
            data = cop.sample(n, param=true_params, rng=rng)
        except TypeError:
            data = cop.sample(n, param=true_params)

        X, Y = data[:, 0], data[:, 1]  # we pass raw (U,V); fit_tau will do ranks internally

        # empirical metrics on the simulated U,V (for reference)
        tau_emp, _ = kendalltau(X, Y)
        beta_emp = empirical_beta(X, Y)
        lamU_emp = huang_lambda(X, Y, side="upper")
        lamL_emp = huang_lambda(X, Y, side="lower")

        # theoretical metrics for the true params
        th_true = metrics_theoretical(name, true_params, cop)

        # ---- core: use the new init-only function ----
        try:
            theta0 = fitter.fit_tau(data=(X, Y), copula=cop)
        except AttributeError as e:
            theta0 = None
            logs.append(f"{name}: missing copula.init_from_data(u,v) -> {e}")
        except Exception as e:
            theta0 = None
            logs.append(f"{name}: fit_tau failed -> {e}")

        # metrics at guessed params (if any)
        th_guess = {"tau_th": None, "lambdaL_th": None, "lambdaU_th": None, "beta_th": None}
        if theta0 is not None:
            th_guess = metrics_theoretical(name, theta0, cop)

        # simple pass/fail heuristic (tolerant)
        def nz(x): return 0.0 if x is None else float(x)
        tau_ok = abs(nz(th_guess["tau_th"]) - nz(th_true["tau_th"])) <= 0.04 if th_true["tau_th"] is not None and th_guess["tau_th"] is not None else True
        lam_ok = abs(nz(th_guess["lambdaU_th"]) - nz(th_true["lambdaU_th"])) <= max(0.05, 0.25 * nz(th_true["lambdaU_th"]) + 0.02) if th_true["lambdaU_th"] is not None and th_guess["lambdaU_th"] is not None else True
        passed = bool(theta0 is not None and tau_ok and lam_ok)

        rows.append({
            "name": name,
            "true_params": np.array(true_params, dtype=float),
            "init_guess": None if theta0 is None else np.array(theta0, dtype=float),
            "tau_emp": float(tau_emp),
            "tau_true": None if th_true["tau_th"] is None else float(th_true["tau_th"]),
            "tau_guess": None if th_guess["tau_th"] is None else float(th_guess["tau_th"]),
            "lambdaU_emp": float(lamU_emp),
            "lambdaU_true": None if th_true["lambdaU_th"] is None else float(th_true["lambdaU_th"]),
            "lambdaU_guess": None if th_guess["lambdaU_th"] is None else float(th_guess["lambdaU_th"]),
            "beta_emp": float(beta_emp),
            "beta_true": None if th_true["beta_th"] is None else float(th_true["beta_th"]),
            "beta_guess": None if th_guess["beta_th"] is None else float(th_guess["beta_th"]),
            "pass": passed
        })

        logs.append(
            f"[{name}] true={true_params} | guess={theta0} | "
            f"tau_emp={tau_emp:.3f} tau_true={nz(th_true['tau_th']):.3f} tau_guess={nz(th_guess['tau_th']):.3f} | "
            f"lamU_emp={lamU_emp:.3f} lamU_true={nz(th_true['lambdaU_th']):.3f} lamU_guess={nz(th_guess['lambdaU_th']):.3f} | "
            f"pass={passed}"
        )

    df = pd.DataFrame(rows, columns=[
        "name","true_params","init_guess",
        "tau_emp","tau_true","tau_guess",
        "lambdaU_emp","lambdaU_true","lambdaU_guess",
        "beta_emp","beta_true","beta_guess",
        "pass"
    ])

    print("\n========== FIT_TAU INIT-ONLY TESTS ==========")
    for line in logs:
        print(line)

    print("\n-------------- SUMMARY TABLE ----------------")
    with pd.option_context("display.max_colwidth", None, "display.width", 160):
        print(df.to_string(index=False))

    print("\nPassed:", int(df['pass'].sum()), "/", len(df))
    return df

if __name__ == "__main__":
    test_fit_tau_init_only(n=12000, seed=42)
