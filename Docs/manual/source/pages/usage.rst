.. _usage:

Utilisation de base
===================

Cette section vous guide dans l'utilisation du pipeline `CopulaFurtif` pour créer, manipuler et diagnostiquer des copules bivariées.


🧱 Création de copules
----------------------

Toutes les copules sont accessibles via la `CopulaFactory` :

.. code-block:: python

   from CopulaFurtif.core.copulas.domain.factories.copula_factory import CopulaFactory

   copula = CopulaFactory.create("gaussian")
   print(copula.name)  # Gaussian Copula

Copules disponibles : `gaussian`, `student`, `clayton`, `frank`, `joe`, `gumbel`, `amh`, `tawn3`, `galambos`, `plackett`, `fgm`, etc.


📊 Données d'entrée
-------------------

Le pipeline attend généralement :

- **Raw data** : données originales pour Kendall's tau (`[[X1, Y1], [X2, Y2], ...]`)
- **Pseudo-observations** : données uniformisées `u, v ∈ (0,1)` via les marges

Générer des pseudo-observations :

.. code-block:: python

   from CopulaFurtif.core.copulas.domain.estimation.utils import pseudo_obs

   u, v = pseudo_obs(data)  # data = [[X1, Y1], [X2, Y2], ...]


📈 Accès aux méthodes de base
-----------------------------

.. code-block:: python

   copula.parameters = [0.5]       # ou [rho, nu] pour Student
   print(copula.get_cdf(0.4, 0.8))
   print(copula.get_pdf(0.4, 0.8))
   print(copula.kendall_tau())

   samples = copula.sample(100)


🔬 Diagnostic
-------------

.. code-block:: python

   from CopulaFurtif.core.copulas.application.services.diagnostics_service import DiagnosticService

   diag = DiagnosticService()
   scores = diag.evaluate(data, copula)
   print(scores)

Résultat : dict avec `LogLik`, `AIC`, `BIC`, `Kendall Tau Error`, etc.


📌 À venir : fitting & visualisation
