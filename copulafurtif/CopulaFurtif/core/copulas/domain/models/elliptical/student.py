import numpy as np
from math import sqrt
from scipy.stats import t, multivariate_t
from scipy.special import gammaln, gamma, roots_genlaguerre
from scipy.stats import kendalltau, multivariate_normal
from domain.models.interfaces import CopulaModel
from domain.models.mixins import ModelSelectionMixin, SupportsTailDependence

class StudentCopula(CopulaModel, ModelSelectionMixin, SupportsTailDependence):
    def __init__(self):
        super().__init__()
        self.name = "Student Copula"
        self.type = "student"
        self.bounds_param = [(-0.999, 0.999), (2.01, 30.0)]  # [rho, df]
        self._parameters = np.array([0.5, 4.0])
        self.default_optim_method = "SLSQP"
        self.n_nodes = 64

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, param):
        param = np.asarray(param)
        for i, (low, high) in enumerate(self.bounds_param):
            if not (low <= param[i] <= high):
                raise ValueError(f"Parameter {i} out of bounds")
        self._parameters = param

    def get_cdf(self, u, v, param=None):
        if param is None:
            param = self.parameters
        rho, nu = param
        if u <= 0 or v <= 0:
            return 0.0
        if u >= 1 and v >= 1:
            return 1.0
        if u >= 1:
            return v
        if v >= 1:
            return u

        x = t.ppf(u, df=nu)
        y = t.ppf(v, df=nu)

        k = nu / 2.0
        alpha = k - 1.0
        z_nodes, w_weights = roots_genlaguerre(self.n_nodes, alpha)
        cov = [[1.0, rho], [rho, 1.0]]
        mvn = multivariate_normal(mean=[0.0, 0.0], cov=cov)
        total = 0.0
        for zi, wi in zip(z_nodes, w_weights):
            scale = sqrt(2.0 * zi / nu)
            total += wi * mvn.cdf([x * scale, y * scale])
        return total / gamma(k)

    def get_pdf(self, u, v, param=None):
        if param is None:
            param = self.parameters
        rho, nu = param
        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        u_q = t.ppf(u, df=nu)
        v_q = t.ppf(v, df=nu)
        det = 1 - rho**2
        quad_form = (u_q**2 - 2 * rho * u_q * v_q + v_q**2) / det
        log_num = gammaln((nu + 2) / 2) + gammaln(nu / 2)
        log_den = 2 * gammaln((nu + 1) / 2)
        log_det = 0.5 * np.log(det)
        log_prod = ((nu + 1) / 2) * (np.log1p((u_q**2) / nu) + np.log1p((v_q**2) / nu))
        log_dent = ((nu + 2) / 2) * np.log1p(quad_form / nu)
        log_c = log_num - log_den - log_det + log_prod - log_dent
        return np.exp(log_c)

    def sample(self, n, param=None):
        if param is None:
            param = self.parameters
        rho, nu = param
        cov = np.array([[1.0, rho], [rho, 1.0]])
        L = np.linalg.cholesky(cov)
        z = np.random.standard_normal((n, 2))
        chi2 = np.random.chisquare(df=nu, size=n)
        scaled = (z @ L.T) / np.sqrt((chi2 / nu)[:, None])
        u = t.cdf(scaled[:, 0], df=nu)
        v = t.cdf(scaled[:, 1], df=nu)
        return np.column_stack((u, v))

    def kendall_tau(self, param=None, n_samples=10000, random_state=None):
        if param is None:
            param = self.parameters
        rho, nu = param
        rng = np.random.RandomState(random_state) if random_state is not None else np.random
        cov = np.array([[1.0, rho], [rho, 1.0]])
        L = np.linalg.cholesky(cov)
        z = rng.standard_normal((n_samples, 2))
        chi2 = rng.chisquare(df=nu, size=n_samples)
        scaled = (z @ L.T) / np.sqrt((chi2 / nu)[:, None])
        u = t.cdf(scaled[:, 0], df=nu)
        v = t.cdf(scaled[:, 1], df=nu)
        tau, _ = kendalltau(u, v)
        return tau

    def LTDC(self, param=None):
        if param is None:
            param = self.parameters
        rho, nu = param
        return 2 * t.cdf(-sqrt((nu + 1) * (1 - rho) / (1 + rho)), df=nu + 1)

    def UTDC(self, param=None):
        return self.LTDC(param)

    def IAD(self, data):
        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan

    def partial_derivative_C_wrt_v(self, u, v, param=None):
        if param is None:
            rho, nu = self.parameters
        else:
            rho, nu = param

        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        tx = t.ppf(u, df=nu)
        ty = t.ppf(v, df=nu)

        df_c = nu + 1
        scale = sqrt((1 - rho**2) * (nu + ty**2) / df_c)
        loc = rho * ty
        z = (tx - loc) / scale
        return t.pdf(z, df=df_c) / scale

    def partial_derivative_C_wrt_u(self, u, v, param=None):
        return self.partial_derivative_C_wrt_v(v, u, param)

    def conditional_cdf_u_given_v(self, u, v, param=None):
        if param is None:
            rho, nu = self.parameters
        else:
            rho, nu = param

        eps = 1e-12
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        tx = t.ppf(u, df=nu)
        ty = t.ppf(v, df=nu)

        df_c = nu + 1
        loc_c = rho * ty
        scale_c = sqrt((nu + ty**2) * (1 - rho**2) / df_c)

        return t.cdf((tx - loc_c) / scale_c, df=df_c)

    def conditional_cdf_v_given_u(self, u, v, param=None):
        return self.conditional_cdf_u_given_v(v, u, param)