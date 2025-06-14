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
from scipy.stats import norm, multivariate_normal



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

    def __init__(self):
        """
        Initialize the copula model with default attributes.
        """
        self._parameters = None
        self.log_likelihood_ = None
        self.n_obs = None

    @property
    def parameters(self) -> np.ndarray:
        """Return parameters as numpy array."""
        return self._parameters
        return self._parameters.as_array()

    @parameters.setter
    def parameters(self, params: CopulaParameters):
        """Validate and set parameters. Uses bounds_param and expected size."""
        self._parameters = params

    def pretty_print(self, equation='cdf'):
        if equation == 'cdf':
            print(pretty(self._parameters.cdf_expr))
        elif equation == 'pdf':
            print(pretty(self._parameters.pdf_expr))
        else:
            raise ValueError("Equation must be 'cdf' or 'pdf'.")
        
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
            return self._parameters.cdf_numeric(u, v, *self.parameters)
        raise NotImplementedError("CDF not defined symbolically. Override get_cdf method.")

    def get_pdf(self, u, v):
        """
        Compute the probability density function c(u, v).

        Args:
            u (float or np.ndarray): First marginal (in [0, 1]).
            v (float or np.ndarray): Second marginal (in [0, 1]).
            param (list or np.ndarray, optional): Copula parameters. Defaults to current parameters.

        Returns:
            float or np.ndarray: Value(s) of the PDF.
        """
        if self._parameters.pdf_numeric:
            return self._parameters.pdf_numeric(u, v, *self.parameters)
        raise NotImplementedError("PDF not defined symbolically. Override get_pdf method.")

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
    def sample(self, n, param=None):
        """
        Generate n random samples from the copula.

        Args:
            n (int): Number of samples to generate.
            param (list or np.ndarray, optional): Copula parameters. Defaults to current parameters.

        Returns:
            np.ndarray: Array of shape (n, 2) with samples from the copula.
        """
        pass
    
    def partial_derivative_C_wrt_u(self, u, v):
        if self._parameters.partial_u_numeric:
            return self._parameters.partial_u_numeric(u, v, *self.parameters)
        raise NotImplementedError("Partial derivative wrt u not defined symbolically.")

    def partial_derivative_C_wrt_v(self, u, v):
        if self._parameters.partial_v_numeric:
            return self._parameters.partial_v_numeric(u, v, *self.parameters)
        raise NotImplementedError("Partial derivative wrt v not defined symbolically.")

    def conditional_cdf_u_given_v(self, u, v):
        return self.partial_derivative_C_wrt_v(u, v)

    def conditional_cdf_v_given_u(self, u, v):
        return self.partial_derivative_C_wrt_u(u, v)