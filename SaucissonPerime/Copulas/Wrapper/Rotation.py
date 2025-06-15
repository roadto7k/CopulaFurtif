import numpy as np

class RotatedCopulaWrapper:
    """
    Wraps an existing copula and applies a 0, 90, 180, or 270-degree rotation
    on the fly, without modifying the original copula class.
    """
    def __init__(self, base_copula, rotation=0):
        if rotation not in (0, 90, 180, 270):
            raise ValueError("Rotation must be 0, 90, 180, or 270.")
        self.base_copula = base_copula
        self.rotation = rotation

    def get_cdf(self, u, v, param):
        if self.rotation == 0:
            return self.base_copula.get_cdf(u, v, param)
        elif self.rotation == 180:
            return u + v - 1 + self.base_copula.get_cdf(1 - u, 1 - v, param)
        elif self.rotation == 90:
            return v - self.base_copula.get_cdf(1 - u, v, param)
        elif self.rotation == 270:
            return u - self.base_copula.get_cdf(u, 1 - v, param)

    def get_pdf(self, u, v, param):
        if self.rotation == 0:
            return self.base_copula.get_pdf(u, v, param)
        elif self.rotation == 180:
            return self.base_copula.get_pdf(1 - u, 1 - v, param)
        elif self.rotation == 90:
            return self.base_copula.get_pdf(1 - u, v, param)
        elif self.rotation == 270:
            return self.base_copula.get_pdf(u, 1 - v, param)

    def sample(self, n, param):
        samples = self.base_copula.sample(n, param)
        u, v = samples[:, 0], samples[:, 1]
        if self.rotation == 0:
            return samples
        elif self.rotation == 180:
            return np.column_stack((1 - u, 1 - v))
        elif self.rotation == 90:
            return np.column_stack((1 - u, v))
        elif self.rotation == 270:
            return np.column_stack((u, 1 - v))

    def kendall_tau(self, param):
        # Typically you'd need to define how tau changes under rotation.
        return self.base_copula.kendall_tau(param)

    def LTDC(self, param):
        if self.rotation == 0:
            return self.base_copula.LTDC(param)
        elif self.rotation == 180:
            return self.base_copula.UTDC(param)
        elif self.rotation in (90, 270):
            return 0.0

    def UTDC(self, param):
        if self.rotation == 0:
            return self.base_copula.UTDC(param)
        elif self.rotation == 180:
            return self.base_copula.LTDC(param)
        elif self.rotation in (90, 270):
            return 0.0
