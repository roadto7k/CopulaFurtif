import numpy as np

class BaseCopula:
    """
    Base class for all bivariate copulas.
    Defines the common API: get_pdf, get_cdf, sample, etc.
    """

    def __init__(self):
        """
        Parameters:
            rotation (int): Must be one of {0, 90, 180, 270} degrees.
        """
        self.type = None
        self.name = None
        self.parameters = None
        self.bounds_param = None
        self.max_likelihood = None
        self.n_obs = None

    def get_pdf(self, u, v, params):
        """
        Computes the density of the copula at (u, v).
        This method should be overridden in the subclass.
        """
        raise NotImplementedError

    def get_cdf(self, u, v, params):
        """
        Computes the cumulative distribution function (CDF) of the copula at (u, v).
        This method should be overridden in the subclass.
        """
        raise NotImplementedError

    def kendall_tau(self, param):
        """
        Computes Kendall's tau for the given parameter.
        This method should be overridden in the subclass.
        """
        raise NotImplementedError

    def sample(self, n, params):
        """
        Generates n samples (u_i, v_i) from the copula with the given parameters.
        This method should be overridden in the subclass.
        """
        raise NotImplementedError
