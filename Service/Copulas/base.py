import numpy as np

class BaseCopula:
    """
    Classe de base pour toutes les copules bivariées.
    Définit l'API commune : get_pdf, get_cdf, sample, etc.
    """

    def __init__(self):
        self.type = None  # type of copula
        self.name = None  # string for output/logging
        self.parameters = None  # initial guess for copula params
        self.bounds_param = None  # optimizer bounds
        self.max_likelihood = None  # function handle, assigned later


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
