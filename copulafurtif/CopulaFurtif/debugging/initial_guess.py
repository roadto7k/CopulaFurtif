import math
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from CopulaFurtif.core.copulas.domain.copula_type import CopulaType
from CopulaFurtif.core.copulas.domain.factories.copula_factory import CopulaFactory

# ---------- helpers: empirical metrics ----------

def empirical_beta(u, v):
    u = np.asarray(u); v = np.asarray(v)
    concord = np.mean(((u > 0.5) & (v > 0.5)) | ((u < 0.5) & (v < 0.5)))
    return 2.0 * concord - 1.0

def empirical_lambda(u, v, side="upper", qs=(0.90, 0.92, 0.94, 0.96, 0.98)):
    u = np.asarray(u); v = np.asarray(v)
    vals = []
    for q in qs:
        if side == "upper":
            qu, qv = np.quantile(u, q), np.quantile(v, q)
            joint = np.mean((u > qu) & (v > qv))
            denom = max(1e-9, 1.0 - q)
        else:
            qu, qv = np.quantile(u, 1.0 - q), np.quantile(v, 1.0 - q)
            joint = np.mean((u < qu) & (v < qv))
            denom = max(1e-9, 1.0 - q)
        vals.append(joint / denom)
    vals = np.array(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    vals.sort()
    # light trimming
    if vals.size >= 5:
        k = max(1, vals.size // 10)
        vals = vals[k:-k] if vals.size - 2 * k >= 1 else vals
    return float(np.median(vals))


def huang_lambda(u, v, side="upper", k=None):
    u, v = np.asarray(u), np.asarray(v)
    n = len(u)
    if k is None:
        k = int(np.sqrt(n))  # règle courante pour k

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

    try:
        # tails available on all families via mixin
        d["lambdaL_th"] = float(cop.LTDC(params))
        d["lambdaU_th"] = float(cop.UTDC(params))
    except Exception:
        d["lambdaL_th"] = d["lambdaU_th"] = None

    try:
        # use class method when implemented as closed form
        d["tau_th"] = float(cop.kendall_tau(params))
    except Exception:
        d["tau_th"] = None

    # plug exact closed forms when faster/safer
    if name.lower().startswith("gaussian"):
        rho = float(params[0])
        d["tau_th"] = tau_gaussian_or_student(rho)
        d["beta_th"] = beta_gaussian_or_student(rho)
    elif name.lower().startswith("student"):
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

# ---------- test bench ----------

def test_init_guesses(n=10000, seed=12345):
    rng = np.random.default_rng(seed)

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

        # sample synthetic data
        try:
            data = cop.sample(n, param=true_params, rng=rng)
        except TypeError:
            # some sample() signatures may not accept rng; fallback
            data = cop.sample(n, param=true_params)
        u, v = data[:, 0], data[:, 1]

        # empirical metrics
        tau_emp, _ = kendalltau(u, v)
        beta_emp = empirical_beta(u, v)
        lamU_emp = huang_lambda(u, v, side="upper")
        lamL_emp = huang_lambda(u, v, side="lower")

        # theoretical metrics for the true params
        th = metrics_theoretical(name, true_params, cop)

        # guess params
        try:
            guess = cop.init_from_data(u, v)
        except AttributeError as e:
            guess = None
            logs.append(f"{name}: missing init_from_data() -> {e}")
        except Exception as e:
            guess = None
            logs.append(f"{name}: init_from_data failed -> {e}")

        # compute "theoretical" metrics at guessed params (if any)
        th_guess = {"tau_th": None, "lambdaL_th": None, "lambdaU_th": None, "beta_th": None}
        if guess is not None:
            th_guess = metrics_theoretical(name, guess, cop)

        # simple pass/fail heuristic
        def safe_abs(x):
            return np.nan if x is None else abs(x)

        tau_true = th["tau_th"]
        tau_g = th_guess["tau_th"]
        lamU_true = th["lambdaU_th"]
        lamU_g = th_guess["lambdaU_th"]

        tau_ok = (safe_abs((tau_g or 0) - (tau_true or 0)) <= 0.04) if (tau_g is not None and tau_true is not None) else True
        lam_ok = (safe_abs((lamU_g or 0) - (lamU_true or 0)) <= max(0.05, 0.25 * (lamU_true or 0) + 0.02)) if (lamU_g is not None and lamU_true is not None) else True
        passed = bool(tau_ok and lam_ok and (guess is not None))

        # store row
        rows.append({
            "name": name,
            "true_params": np.array(true_params, dtype=float),
            "init_guess": None if guess is None else np.array(guess, dtype=float),
            "tau_emp": float(tau_emp),
            "tau_true": None if tau_true is None else float(tau_true),
            "tau_guess": None if tau_g is None else float(tau_g),
            "lambdaU_emp": float(lamU_emp),
            "lambdaU_true": None if lamU_true is None else float(lamU_true),
            "lambdaU_guess": None if lamU_g is None else float(lamU_g),
            "beta_emp": float(beta_emp),
            "beta_true": None if th["beta_th"] is None else float(th["beta_th"]),
            "beta_guess": None if th_guess["beta_th"] is None else float(th_guess["beta_th"]),
            "pass": passed
        })

        # verbose per-copula log
        tau_true_val = "None" if th["tau_th"] is None else f"{th['tau_th']:.3f}"
        tau_guess_val = "None" if th_guess["tau_th"] is None else f"{th_guess['tau_th']:.3f}"
        lamU_true_val = "None" if th["lambdaU_th"] is None else f"{th['lambdaU_th']:.3f}"
        lamU_guess_val = "None" if th_guess["lambdaU_th"] is None else f"{th_guess['lambdaU_th']:.3f}"

        logs.append(
            f"[{name}] true={true_params} | guess={guess} | "
            f"tau_emp={tau_emp:.3f} tau_true={tau_true_val} tau_guess={tau_guess_val} | "
            f"lamU_emp={lamU_emp:.3f} lamU_true={lamU_true_val} lamU_guess={lamU_guess_val} | "
            f"pass={passed}"
        )

    df = pd.DataFrame(rows, columns=[
        "name","true_params","init_guess",
        "tau_emp","tau_true","tau_guess",
        "lambdaU_emp","lambdaU_true","lambdaU_guess",
        "beta_emp","beta_true","beta_guess",
        "pass"
    ])

    # prints
    print("\n========== INIT GUESS TESTS (no MLE) ==========")
    for line in logs:
        print(line)

    print("\n-------------- SUMMARY TABLE ------------------")
    with pd.option_context("display.max_colwidth", None, "display.width", 160):
        print(df.to_string(index=False))

    print("\nPassed:", int(df['pass'].sum()), "/", len(df))
    return df

if __name__ == "__main__":
    test_init_guesses(n=12000, seed=42)
