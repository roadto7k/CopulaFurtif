import numpy as np
from scipy.optimize import root_scalar

from Service.Copulas.base import BaseCopula


class AMHCopula(BaseCopula):
    """
    Ali–Mikhail–Haq (AMH) Copula class

    Attributes
    ----------
    family : str
        Identifier for the copula family. Here, "amh".
    name : str
        Human-readable name for output/logging.
    bounds_param : list of tuple
        Bounds for the copula parameter θ, here in (−1, 1).
    parameters : np.ndarray
        Initial guess for the copula parameter [θ].
    default_optim_method : str
        Default optimizer to use.
    """

    def __init__(self):
        super().__init__()
        self.type = "amh"
        self.name = "Ali–Mikhail–Haq Copula"
        # θ ∈ (−1, 1)
        self.bounds_param = [(-0.999999, 0.999999)]
        self.parameters = np.array([0.0])  # initial θ
        self.default_optim_method = "SLSQP"

    def get_cdf(self, u, v, param):
        """
        C(u,v) = u·v / [1 − θ·(1−u)·(1−v)]

        Parameters
        ----------
        u, v : float or array-like
            Pseudo-observations in [0, 1].
        param : iterable
            Copula parameter θ in (−1, 1).

        Returns
        -------
        float or np.ndarray
            Copula CDF at (u, v). :contentReference[oaicite:0]{index=0}
        """
        θ = param[0]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        denom = 1.0 - θ * (1.0 - u) * (1.0 - v)
        return (u * v) / denom

    def get_pdf(self, u, v, param):
        """
        PDF c(u,v) = [1 + θ·((1+u)(1+v) − 3) + θ²·(1−u)(1−v)]
                     / [1 − θ·(1−u)·(1−v)]³

        Parameters
        ----------
        u, v : float or array-like
            Pseudo-observations in [0, 1].
        param : iterable
            Copula parameter θ in (−1, 1).

        Returns
        -------
        float or np.ndarray
            Copula PDF at (u, v). :contentReference[oaicite:1]{index=1}
        """
        θ = param[0]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        one_minus_u = 1.0 - u
        one_minus_v = 1.0 - v
        D = 1.0 - θ * one_minus_u * one_minus_v
        num = (1.0
               + θ * ((1.0 + u) * (1.0 + v) - 3.0)
               + θ**2 * one_minus_u * one_minus_v)
        return num / (D**3)

    def kendall_tau(self, param):
        """
        Kendall's tau:
          τ(θ) = 1 − 2·[ (1−θ)²·ln(1−θ) + θ ] / (3·θ²)

        Parameters
        ----------
        param : iterable
            Copula parameter θ in (−1, 1).

        Returns
        -------
        float
            Kendall's τ. :contentReference[oaicite:2]{index=2}
        """
        θ = param[0]
        if abs(θ) < 1e-8:
            return 0.0
        return 1.0 - 2.0 * (((1 - θ)**2 * np.log(1 - θ) + θ) / (3 * θ**2))

    def partial_derivative_C_wrt_u(self, u, v, param):
        """
        ∂C/∂u = v·[1 − θ·(1−v)] / [1 − θ·(1−u)·(1−v)]²

        Parameters
        ----------
        u, v : float or array-like
            Values in (0,1).
        param : iterable
            Copula parameter θ.

        Returns
        -------
        float or np.ndarray
            ∂C/∂u.
        """
        θ = param[0]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        D = 1.0 - θ * (1.0 - u) * (1.0 - v)
        return v * (1.0 - θ * (1.0 - v)) / (D**2)

    def partial_derivative_C_wrt_v(self, u, v, param):
        """
        ∂C/∂v = u·[1 − θ·(1−u)] / [1 − θ·(1−u)·(1−v)]²

        Parameters
        ----------
        u, v : float or array-like
            Values in (0,1).
        param : iterable
            Copula parameter θ.

        Returns
        -------
        float or np.ndarray
            ∂C/∂v.
        """
        θ = param[0]
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        D = 1.0 - θ * (1.0 - u) * (1.0 - v)
        return u * (1.0 - θ * (1.0 - u)) / (D**2)

    def conditional_cdf_v_given_u(self, u, v, param):
        """
        P(V ≤ v | U = u) = ∂C/∂u(u,v)
        """
        return self.partial_derivative_C_wrt_u(u, v, param)

    def conditional_cdf_u_given_v(self, u, v, param):
        """
        P(U ≤ u | V = v) = ∂C/∂v(u,v)
        """
        return self.partial_derivative_C_wrt_v(u, v, param)

    def sample(self, n, param):
        """
        Generate n samples via conditional inversion:
          u ~ Uniform(0,1)
          v solves ∂C/∂u(u,v) = p by bisection

        Parameters
        ----------
        n : int
            Number of samples.
        param : iterable
            Copula parameter θ.

        Returns
        -------
        np.ndarray
            n×2 array of pseudo-observations.
        """
        θ = param[0]
        eps = 1e-6
        u = np.random.rand(n)
        v = np.empty(n)
        for i in range(n):
            p = np.random.rand()
            sol = root_scalar(
                lambda vv: self.partial_derivative_C_wrt_u(u[i], vv, param) - p,
                bracket=[eps, 1 - eps],
                method="bisect",
                xtol=1e-6
            )
            v[i] = sol.root
        return np.column_stack((u, v))

    def LTDC(self, param):
        """
        Lower tail dependence λ_L = 0 for AMH. :contentReference[oaicite:3]{index=3}
        """
        return 0.0

    def UTDC(self, param):
        """
        Upper tail dependence λ_U = 0 for AMH. :contentReference[oaicite:4]{index=4}
        """
        return 0.0

