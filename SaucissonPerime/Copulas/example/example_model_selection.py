import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta, lognorm, t as student_t

from SaucissonPerime.Copulas.archimedean.BB1 import BB1Copula
from SaucissonPerime.Copulas.archimedean.BB2 import BB2Copula
from SaucissonPerime.Copulas.archimedean.fgm import FGMCopula
from SaucissonPerime.Copulas.archimedean.frank import FrankCopula
from SaucissonPerime.Copulas.archimedean.clayton import ClaytonCopula
from SaucissonPerime.Copulas.archimedean.galambos import GalambosCopula
from SaucissonPerime.Copulas.archimedean.gumbel import GumbelCopula
from SaucissonPerime.Copulas.archimedean.joe import JoeCopula
from SaucissonPerime.Copulas.archimedean.plackett import PlackettCopula
from SaucissonPerime.Copulas.elliptical.gaussian import GaussianCopula
from SaucissonPerime.Copulas.elliptical.student import StudentCopula

from SaucissonPerime.Copulas.fitting.estimation import fit_mle, pseudo_obs
from SaucissonPerime.Copulas.metrics.model_selection import copula_diagnostics, \
    interpret_copula_results


def generate_data_beta_lognorm_student(n=1000, rho=0.7, nu=4):
    """
    Generates (X, Y) data with dependence induced by a Student copula.

    X ~ Beta(2, 5)
    Y ~ LogNorm(s=0.5, scale=exp(1))

    Parameters
    ----------
    n : int
        Number of samples.
    rho : float
        Correlation between the margins (copula parameter).
    nu : float
        Degrees of freedom for the Student copula.

    Returns
    -------
    X, Y : arrays
        Simulated data samples.
    """
    # 1) Generate t-Student copula samples
    cov = np.array([[1, rho], [rho, 1]])
    L = np.linalg.cholesky(cov)
    Z = np.random.standard_normal((n, 2))
    chi2 = np.random.chisquare(df=nu, size=n)
    T = (Z @ L.T) / np.sqrt(chi2 / nu)[:, None]

    # 2) Transform to uniform margins using the Student-t CDF
    U = student_t.cdf(T[:, 0], df=nu)
    V = student_t.cdf(T[:, 1], df=nu)

    # 3) Apply the desired marginals: X ~ Beta(2,5), Y ~ LogNorm(s=0.5, scale=exp(1))
    X = beta.ppf(U, a=2, b=5)
    Y = lognorm.ppf(V, s=0.5, scale=np.exp(1))

    return X, Y


def plot_tail_dependence(data, candidate_list, q_low=0.05, q_high=0.95):
    """
    Creates a two-panel plot of the pseudo-observations showing the lower and upper tail regions.
    Also displays a text box summarizing each candidate's theoretical tail dependence.

    Parameters
    ----------
    data : list or tuple of two arrays
        Raw data samples [X, Y].
    candidate_list : list
        List of candidate copula objects (their parameters should have been updated via MLE).
    q_low : float, optional
        Quantile for the lower tail (default=0.05).
    q_high : float, optional
        Quantile for the upper tail (default=0.95).
    """
    # Get pseudo-observations using the existing pseudo_obs() function.
    u, v = pseudo_obs(data)
    # Compute empirical tail dependence values.
    lower_mask = (u <= q_low) & (v <= q_low)
    upper_mask = (u > q_high) & (v > q_high)
    emp_lambda_L = np.sum(lower_mask) / np.sum(u <= q_low) if np.sum(u <= q_low) > 0 else 0.0
    emp_lambda_U = np.sum(upper_mask) / np.sum(u > q_high) if np.sum(u > q_high) > 0 else 0.0

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Lower tail plot
    axs[0].scatter(u, v, s=10, alpha=0.3, color='grey', label="All points")
    axs[0].scatter(u[lower_mask], v[lower_mask], s=10, color='red', label="Lower tail")
    axs[0].set_title(f"Lower Tail (u,v ≤ {q_low})\nEmpirical LTDC: {emp_lambda_L:.3f}")
    axs[0].set_xlabel("u")
    axs[0].set_ylabel("v")
    axs[0].grid(True)
    axs[0].legend()

    # Upper tail plot
    axs[1].scatter(u, v, s=10, alpha=0.3, color='grey', label="All points")
    axs[1].scatter(u[upper_mask], v[upper_mask], s=10, color='green', label="Upper tail")
    axs[1].set_title(f"Upper Tail (u,v > {q_high})\nEmpirical UTDC: {emp_lambda_U:.3f}")
    axs[1].set_xlabel("u")
    axs[1].set_ylabel("v")
    axs[1].grid(True)
    axs[1].legend()

    # Build a text summary of candidate theoretical tail dependencies.
    text_lines = ["Candidate Theoretical Tail Dependence:"]
    for copula in candidate_list:
        param = copula.parameters
        ltdc = copula.LTDC(param)
        utdc = copula.UTDC(param)
        text_lines.append(f"{copula.name}: LTDC = {ltdc:.3f}, UTDC = {utdc:.3f}")
    text_str = "\n".join(text_lines)

    # Add text box in the figure.
    fig.text(0.5, 0.02, text_str, ha="center", fontsize=10, bbox=dict(facecolor="white", alpha=0.8))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()


def main():
    np.random.seed(42)

    # --- Simulate Data ---
    true_rho, true_nu = 0.7, 7
    X, Y = generate_data_beta_lognorm_student(n=10000, rho=true_rho, nu=true_nu)
    data = [X, Y]

    print("=== Simulated Data using a Student Copula Structure ===")
    print(f"True rho: {true_rho}, True nu: {true_nu}")
    print("=" * 50)

    # --- Define Known Marginals for MLE ---
    marginals_known = [
        {"distribution": beta, "a": 2, "b": 5, "loc": 0, "scale": 1},
        {"distribution": lognorm, "s": 0.5, "loc": 0, "scale": np.exp(1)}
    ]

    # --- Instantiate Candidate Copula Objects ---
    frank = FrankCopula()
    clayton = ClaytonCopula()
    gumbel = GumbelCopula()
    BB1 = BB1Copula()
    BB2 = BB2Copula()
    fgm = FGMCopula()
    galambos = GalambosCopula()
    joe = JoeCopula()
    plackett = PlackettCopula()
    gaussian = GaussianCopula()
    student = StudentCopula()

    candidate_list = [frank, clayton, gumbel, BB1, BB2, fgm, galambos, joe, plackett, gaussian, student]

    # --- Fit Each Candidate Copula using MLE with Known Marginals ---
    for copula in candidate_list:
        try:
            fitted_params, loglik = fit_mle(data, copula, marginals=marginals_known,
                                            known_parameters=True,
                                            opti_method=copula.default_optim_method)
            copula.parameters = np.array(fitted_params)
            print(f"{copula.name} fitted successfully:")
            print(f"  Fitted parameters: {fitted_params}")
            print(f"  Log-likelihood: {loglik:.4f}")
        except Exception as e:
            print(f"MLE fitting for {copula.name} failed: {e}")

    # --- Compare model fitting ---
    result_df = copula_diagnostics(data, candidate_list, verbose=True, quick=True)
    print(result_df)
    msg = interpret_copula_results(result_df)
    print(msg)
    print('done')

if __name__ == "__main__":
    main()
