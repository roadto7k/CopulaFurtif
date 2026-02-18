"""
CopulaFurtif - Abstract Base Class for Copula Models

This module defines the abstract base class `CopulaModel`, which provides the
interface for all bivariate copula models within the CopulaFurtif framework.

All custom copula implementations should inherit from this class and provide 
concrete implementations of the abstract methods.

Attributes:
    _parameters (list or np.ndarray): Internal parameters of the copula.
    bounds_param (list of tuple): Parameter bounds for optimization.
    log_likelihood_ (float): Log-likelihood of the fitted model.
    n_obs (int): Number of observations used in fitting.

Abstract Methods:
    get_cdf(u, v, param): Compute the cumulative distribution function.
    get_pdf(u, v, param): Compute the probability density function.
    kendall_tau(param): Compute Kendall's tau from the model.
    sample(n, param): Generate random samples from the copula.
"""

from abc import ABC, abstractmethod
import numpy as np
from sympy import symbols, diff, pretty
from sympy.utilities.lambdify import lambdify
import autograd.numpy as anp
# from autograd import grad, hessian
from scipy.stats import norm, multivariate_normal
import jax
import jax.numpy as jnp
from jax import grad, hessian



class CopulaParameters:
    def __init__(self, values: np.ndarray, bounds: list[tuple], names: list[str] = None,
                 cdf_expr=None, pdf_expr=None, input_symbols=None, param_symbols=None, cdf_numeric: callable = None,
                    pdf_numeric: callable = None,
                    partial_u_numeric: callable = None,
                    partial_v_numeric: callable = None,):
        self.bounds = bounds
        self.expected_size = len(bounds)
        self.names = names or [f"param_{i}" for i in range(self.expected_size)]
        self.values = self._validate(values)

        self.input_symbols = input_symbols
        self.param_symbols = param_symbols

        all_symbols = input_symbols + param_symbols if input_symbols and param_symbols else []

        self.cdf_expr = cdf_expr
        self.pdf_expr = pdf_expr or (diff(cdf_expr, *input_symbols) if cdf_expr and input_symbols else None)

        self.cdf_numeric = cdf_numeric or (lambdify(all_symbols, self.cdf_expr, modules=['scipy', 'numpy']) if self.cdf_expr else None)
        self.pdf_numeric = pdf_numeric or (lambdify(all_symbols, self.pdf_expr, modules=['scipy', 'numpy']) if self.pdf_expr else None)

        if partial_u_numeric and partial_v_numeric:
            self.partial_u_numeric = partial_u_numeric
            self.partial_v_numeric = partial_v_numeric
        elif self.pdf_expr and input_symbols:
            self.partial_u_expr = diff(self.pdf_expr, self.input_symbols[0])
            self.partial_v_expr = diff(self.pdf_expr, self.input_symbols[1])
            self.partial_u_numeric = lambdify(all_symbols, self.partial_u_expr, modules=['scipy', 'numpy'])
            self.partial_v_numeric = lambdify(all_symbols, self.partial_v_expr, modules=['scipy', 'numpy'])
        else:
            self.partial_u_expr = self.partial_v_expr = None
            self.partial_u_numeric = self.partial_v_numeric = None

    def _validate(self, values):
        values = np.asarray(values, dtype=float)
        if len(values) != self.expected_size:
            raise ValueError(f"Expected {self.expected_size} parameters, got {len(values)}.")
        for i, (val, (lo, hi)) in enumerate(zip(values, self.bounds)):
            if not (lo < val < hi):
                raise ValueError(f"Parameter '{self.names[i]}'={val} out of bounds ({lo}, {hi}).")
        return values

    def __getitem__(self, idx):
        return self.values[idx]

    def as_array(self):
        return self.values.copy()

    def __len__(self):
        return self.values.__len__()
    
    def __repr__(self):
        return f"CopulaParameters({dict(zip(self.names, self.values))})"
    
    def get_numeric_values(self):
        return self.values.copy()

    def set_numeric_values(self, new_values):
        validated_values = self._validate(new_values)
        self.values = validated_values

    def get_symbolic_expressions(self):
        return {
            'cdf': self.cdf_expr,
            'pdf': self.pdf_expr,
            'partial_u': getattr(self, 'partial_u_expr', None),
            'partial_v': getattr(self, 'partial_v_expr', None)
        }
    
    def set_symbolic_expressions(self, cdf_expr=None, pdf_expr=None):
        if cdf_expr:
            self.cdf_expr = cdf_expr
        if pdf_expr:
            self.pdf_expr = pdf_expr

    def get_bounds(self):
        return self.bounds.copy()
    
    def set_bounds(self, new_bounds):
        if len(new_bounds) != self.expected_size:
            raise ValueError("Length of bounds must match expected size.")
        self.bounds = new_bounds

    def get_names(self):
        return self.names.copy()

    def set_names(self, new_names):
        if len(new_names) != self.expected_size:
            raise ValueError("Length of names must match expected size.")
        self.names = new_names

class CopulaModel(ABC):
    """
    Abstract base class for all bivariate copula models.

    This class defines the required interface for any copula implementation 
    used in the CopulaFurtif pipeline. Each concrete subclass must implement 
    the CDF, PDF, Kendall's tau, and sampling methods.

    Attributes:
        _parameters (list or np.ndarray): Internal copula parameters.
        bounds_param (list of tuple): Bounds for parameter optimization.
        log_likelihood_ (float): Log-likelihood from the most recent fit.
        n_obs (int): Number of observations in the dataset.
    """

    def __init__(self, *, use_jax: bool = False):
        """
        Initialize the copula model with default attributes.
        use_jax : bool
            False  → backend NumPy (copule «light»)
            True   → backend JAX  (copule high‑dimension, AD, JIT…)
        """
        self._use_jax = bool(use_jax)
        self._parameters = None
        self.log_likelihood_ = None
        self.n_obs = None
        self.name = ""

    def pretty_print(self, equation='cdf'):
        if equation == 'cdf':
            print(pretty(self._parameters.cdf_expr))
        elif equation == 'pdf':
            print(pretty(self._parameters.pdf_expr))
        else:
            raise ValueError("Equation must be 'cdf' or 'pdf'.")
    
    def get_parameters(self):
            return self._parameters.get_numeric_values()
        
    def init_parameters(self, params : CopulaParameters):
        self._parameters = params
        
    def set_parameters(self, params):
        self._parameters.set_numeric_values(params)

    def get_bounds(self):
        return self._parameters.get_bounds()

    def set_bounds(self, bounds):
        self._parameters.set_bounds(bounds)

    def get_parameters_names(self):
        return self._parameters.get_names()

    def set_names(self, names):
        self._parameters.set_names(names)

    def get_name(self):
        return self.name

    def get_log_likelihood(self):
        return self.log_likelihood_

    def set_log_likelihood(self, ll_value):
        self.log_likelihood_ = ll_value

    def get_n_obs(self):
        return self.n_obs

    def set_n_obs(self, n):
        self.n_obs = n

    def _xp(self):
        """Returns the active array module (np or jnp)."""
        return jnp if self._use_jax else np

    def _to_backend(self, x):
        """Cast user input to the current backend."""
        return self._xp().asarray(x)

    def get_cdf(self, u, v):
        """
        Compute the cumulative distribution function C(u, v).

        Args:
            u (float or np.ndarray): First marginal (in [0, 1]).
            v (float or np.ndarray): Second marginal (in [0, 1]).
            param (list or np.ndarray, optional): Copula parameters. Defaults to current parameters.

        Returns:
            float or np.ndarray: Value(s) of the CDF.
        """
        if self._parameters.cdf_numeric:
            return self._parameters.cdf_numeric(u, v, *self.get_parameters())
        raise NotImplementedError("CDF not defined symbolically. Override get_cdf method.")

    def get_pdf(self, u, v, param = None):
        """
        Compute the probability density function c(u, v).

        Args:
            u (float or np.ndarray): First marginal (in [0, 1]).
            v (float or np.ndarray): Second marginal (in [0, 1]).
            param (list or np.ndarray, optional): Copula parameters. Defaults to current parameters.

        Returns:
            float or np.ndarray: Value(s) of the PDF.
        """
        pass

    @abstractmethod
    def kendall_tau(self, param=None):
        """
        Compute the analytical Kendall's tau implied by the copula.

        Args:
            param (list or np.ndarray, optional): Copula parameters. Defaults to current parameters.

        Returns:
            float: Kendall's tau coefficient.
        """
        pass

    @abstractmethod
    def sample(self, n, param=None, rng=None):
        """
        Generate n random samples from the copula.

        Args:
            n (int): Number of samples to generate.
            param (list or np.ndarray, optional): Copula parameters. Defaults to current parameters.

        Returns:
            np.ndarray: Array of shape (n, 2) with samples from the copula.
        """
        pass
    
    def partial_derivative_C_wrt_u(self, u, v, param = None):
        """
        Compute the partial derivative ∂C(u,v)/∂u using numeric or symbolic implementation.

        Args:
            u (float or array-like): Value of U in (0,1).
            v (float or array-like): Value of V in (0,1).

        Returns:
            float or numpy.ndarray: Value of ∂C/∂u at (u, v).

        Raises:
            NotImplementedError: If no numeric implementation is provided.
        """

        pass

    def partial_derivative_C_wrt_v(self, u, v, param = None):
        """
        Compute the partial derivative ∂C(u,v)/∂v using numeric or symbolic implementation.

        Args:
            u (float or array-like): Value of U in (0,1).
            v (float or array-like): Value of V in (0,1).

        Returns:
            float or numpy.ndarray: Value of ∂C/∂v at (u, v).

        Raises:
            NotImplementedError: If no numeric implementation is provided.
        """

        return self.partial_derivative_C_wrt_u(v, u, param)

    def conditional_cdf_u_given_v(self, u, v, param=None, normalize=True):
        """
        Computes the conditional CDF P(U ≤ u | V = v).

        For any copula: ∂C(1,v)/∂v = 1. Hence normalization at u=1 is theoretically unnecessary,
        and evaluating at the boundary can be numerically unstable for some families.
        """
        if param is None:
            param = self.get_parameters()
        if param is None:
            raise ValueError("conditional_cdf_u_given_v: param is None and get_parameters() returned None")

        u = np.asarray(u, float)
        v = np.asarray(v, float)

        eps = 1e-10
        u = np.clip(u, eps, 1.0 - eps)
        v = np.clip(v, eps, 1.0 - eps)

        duv = self.partial_derivative_C_wrt_v(u, v, param)
        if duv is None:
            raise ValueError(f"{type(self).__name__}.partial_derivative_C_wrt_v returned None")

        duv = np.asarray(duv, float)

        # Theoretical normalization: d/dv C(1,v) = 1
        # Keep `normalize` switch for API compatibility, but avoid boundary evaluation.
        out = duv

        # Make sure it behaves like a CDF
        return np.clip(out, 1e-12, 1.0 - 1e-12)

    def conditional_cdf_v_given_u(self, u, v, param=None, normalize=True):
        """
        Computes the conditional CDF P(V ≤ v | U = u).

        For any copula: ∂C(u,1)/∂u = 1. Hence normalization at v=1 is theoretically unnecessary,
        and evaluating at the boundary can be numerically unstable for some families.
        """
        if param is None:
            param = self.get_parameters()
        if param is None:
            raise ValueError("conditional_cdf_v_given_u: param is None and get_parameters() returned None")

        u = np.asarray(u, float)
        v = np.asarray(v, float)

        eps = 1e-10
        u = np.clip(u, eps, 1.0 - eps)
        v = np.clip(v, eps, 1.0 - eps)

        duv = self.partial_derivative_C_wrt_u(u, v, param)
        if duv is None:
            raise ValueError(f"{type(self).__name__}.partial_derivative_C_wrt_u returned None")

        duv = np.asarray(duv, float)

        # Theoretical normalization: d/du C(u,1) = 1
        out = duv

        return np.clip(out, 1e-12, 1.0 - 1e-12)

    def IAD(self, data):
        print(f"[INFO] IAD is disabled for {self.name}.")
        return np.nan

    def AD(self, data):
        print(f"[INFO] AD is disabled for {self.name}.")
        return np.nan

    def LTDC(self, param: np.ndarray = None):

        return np.nan

    def UTDC(self, param: np.ndarray = None):

        return np.nan

    def blomqvist_beta(self, param=None):

        return np.nan

    def init_from_data(self, u, v):

        return np.nan