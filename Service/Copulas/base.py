import numpy as np

class BaseCopula:
    """
    Classe de base pour toutes les copules bivariées.
    Définit l'API commune : get_pdf, get_cdf, sample, etc.
    """

    def __init__(self, rotation=0):
        """
        rotation : int ∈ {0, 90, 180, 270}
        """
        self.type = None
        self.name = None
        self.parameters = None
        self.bounds_param = None
        self.max_likelihood = None
        self.rotation = rotation
        if self.rotation not in {0, 90, 180, 270}:
            raise ValueError("Rotation must be 0, 90, 180, or 270 degrees.")


    def get_pdf(self, u, v, params):
        """
        Densité de la copule évaluée en (u, v).
        A override dans la classe fille.
        """
        raise NotImplementedError

    def get_cdf(self, u, v, params):
        """
        Fonction de répartition de la copule en (u, v).
        Idem, à override si on veut la CDF.
        """
        raise NotImplementedError

    def kendall_tau(self, param):

        raise NotImplementedError

    def sample(self, n, params):
        """
        Génération de n échantillons (u_i, v_i) ~ Copule(params).
        A override dans la classe fille.
        """
        raise NotImplementedError
