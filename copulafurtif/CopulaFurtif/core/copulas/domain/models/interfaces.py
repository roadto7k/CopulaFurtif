from abc import ABC, abstractmethod
import numpy as np

class CopulaModel(ABC):
    def __init__(self):
        self._parameters = None
        self.bounds_param = None
        self.log_likelihood_ = None
        self.n_obs = None

    @abstractmethod
    def get_cdf(self, u, v, param=None): pass

    @abstractmethod
    def get_pdf(self, u, v, param=None): pass

    @abstractmethod
    def kendall_tau(self, param=None): pass

    @abstractmethod
    def sample(self, n, param=None): pass