.. _extending:

Étendre le pipeline : ajouter une copule
========================================

Cette section vous montre comment intégrer une nouvelle copule dans le pipeline CopulaFurtif selon l'architecture hexagonale.


🧱 Étapes pour ajouter une copule
----------------------------------

1. **Créer la classe de la copule**

   - Héritez de `CopulaModel` (et `ModelSelectionMixin`, `SupportsTailDependence` si applicable)
   - Implémentez les méthodes : `get_cdf`, `get_pdf`, `sample`, `kendall_tau`, etc.

.. code-block:: python

   from CopulaFurtif.core.copulas.domain.models.interfaces import CopulaModel

   class MyCopula(CopulaModel):
       def __init__(self):
           super().__init__()
           self.name = "My Copula"
           self.type = "mycopula"
           self.bounds_param = [(0.1, 5.0)]
           self._parameters = [1.0]

       def get_cdf(self, u, v, param=None):
           ...

       def get_pdf(self, u, v, param=None):
           ...

       def sample(self, n, param=None):
           ...

       def kendall_tau(self, param=None):
           ...

2. **Ajouter dans la factory**

.. code-block:: python

   from CopulaFurtif.core.copulas.domain.factories.copula_factory import CopulaFactory
   from CopulaFurtif.core.copulas.domain.models.archimedean.mycopula import MyCopula

   CopulaFactory.register("mycopula", MyCopula)


3. **Écrire un test unitaire**

   - Testez tous les comportements : paramètres, PDF, CDF, dérivées, etc.
   - Placez le fichier dans `tests/units/test_my_model.py`


4. **(Facultatif) Ajouter une visualisation**

   Si besoin, ajoutez une fonction dans `copula_viz_adapter.py`


🧪 Exemple complet
------------------

Un exemple d’intégration complète (copule Joe ou Gumbel) est disponible dans `tests/` et `domain/models/`.


📌 Bonnes pratiques
-------------------

- Utilisez `np.clip` pour les bornes (éviter les log(0), division par 0)
- Ajoutez `@property parameters` avec setter validant `bounds_param`
- Implémentez `__str__` si utile pour debug ou logs


📚 Voir aussi : `copula_factory.py`, `test_factory_and_archimedean.py`, `diagnostics_service.py`
