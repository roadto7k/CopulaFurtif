import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from Service.Copulas.base import BaseCopula


class BB8Copula(BaseCopula):
    """
    BB8 copula as defined in Durante et al.

    C(u,v) = [1 − {1 − [1 − (1−u)^theta]^delta} * {1 − [1 − (1−v)^theta]^delta}]^{1/theta}

    Parameters:
        - theta (float): tail dependence (theta >= 1)
        - delta (float): asymmetry (0 < delta <= 1)

    This copula combines asymmetric behavior with tail dependence and generalizes both
    Clayton and Joe copulas in the Archimedean family.
    """

    def __init__(self):
        super().__init__()
        self.type = "bb8"
        self.name = "BB8 Copula (Durante)"
        self.bounds_param = [(1.0, None), (0.0, 1.0)]  # theta >= 1, delta in (0,1]
        self.parameters = np.array([2.0, 0.7])
        self.default_optim_method = "SLSQP"

    def get_cdf(self, u, v, param):
        """
        Compute the copula CDF C(u,v) using the BB8 formula.
        """
        theta, delta = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        if np.isclose(u, 0) or np.isclose(v, 0):
            return 0.0
        if np.isclose(u, 1):
            return v
        if np.isclose(v, 1):
            return u

        A = (1 - (1 - u)**theta)**delta
        B = (1 - (1 - v)**theta)**delta
        inner = 1 - (1 - A) * (1 - B)
        return inner**(1.0 / theta)

    def get_pdf(self, u, v, param):
        """
        Approximate the copula PDF c(u,v) = d^2C/du dv via finite differences.
        """
        eps = 1e-6
        du = dv = eps
        c1 = self.get_cdf(u, v, param)
        c2 = self.get_cdf(u + du, v, param)
        c3 = self.get_cdf(u, v + dv, param)
        c4 = self.get_cdf(u + du, v + dv, param)
        return (c4 - c3 - c2 + c1) / (du * dv)

    def partial_derivative_C_wrt_u(self, u, v, param):
        """
        Approximate the partial derivative dC/du using forward finite difference.
        """
        eps = 1e-6
        return (self.get_cdf(u + eps, v, param) - self.get_cdf(u, v, param)) / eps

    def partial_derivative_C_wrt_v(self, u, v, param):
        """
        Approximate the partial derivative dC/dv using forward finite difference.
        """
        eps = 1e-6
        return (self.get_cdf(u, v + eps, param) - self.get_cdf(u, v, param)) / eps

    def conditional_cdf_v_given_u(self, u, v, param):
        """
        Conditional CDF: P(V ≤ v | U = u)
        """
        return self.partial_derivative_C_wrt_u(u, v, param)

    def conditional_cdf_u_given_v(self, u, v, param):
        """
        Conditional CDF: P(U ≤ u | V = v)
        """
        return self.partial_derivative_C_wrt_v(u, v, param)

    def sample(self, n, param):
        """
        Generate n samples from the copula using conditional inversion.
        """
        u = np.random.rand(n)
        v = np.empty(n)
        for i in range(n):
            p = np.random.rand()
            sol = root_scalar(
                lambda vv: self.conditional_cdf_v_given_u(u[i], vv, param) - p,
                bracket=[1e-6, 1 - 1e-6],
                method='bisect',
                xtol=1e-6
            )
            v[i] = sol.root
        return np.column_stack((u, v))

    def kendall_tau(self, param):
        """
        Approximate Kendall's tau numerically via:
        τ = 1 − 4 ∫∫ C(u,v) dC(u,v)
        (Note: not analytically available for BB8)
        """
        u = np.linspace(0.01, 0.99, 50)
        v = np.linspace(0.01, 0.99, 50)
        uu, vv = np.meshgrid(u, v)
        C = self.get_cdf(uu, vv, param)
        integrand = C * self.get_pdf(uu, vv, param)
        tau = 1 - 4 * np.mean(integrand)
        return tau

    def LTDC(self, param):
        """
        Lower-tail dependence λ_L ≈ lim_{u→0} C(u,u)/u
        """
        u = 1e-4
        return self.get_cdf(u, u, param) / u

    def UTDC(self, param):
        """
        Upper-tail dependence λ_U ≈ lim_{u→1} (1 - 2u + C(u,u)) / (1 - u)
        """
        u = 1 - 1e-4
        return (1 - 2 * u + self.get_cdf(u, u, param)) / (1 - u)