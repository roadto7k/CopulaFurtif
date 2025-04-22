.. _fitting:

Ajustement de copules (fitting)
================================

Cette page décrit comment ajuster une copule à vos données à l'aide du pipeline CopulaFurtif.


🧪 Objectif
----------
Trouver les meilleurs paramètres de la copule qui maximisent la vraisemblance des données.


⚙️ Outils disponibles
---------------------

Trois méthodes principales d'ajustement sont supportées :

- `CMLE` : Canonical Maximum Likelihood Estimation (avec pseudo-observations)
- `MLE`  : Maximum Likelihood sur données brutes + marges
- `IFM`  : Inference Function for Margins (deux étapes)


🚀 Exemple : CMLE
------------------

.. code-block:: python

   from CopulaFurtif.core.copulas.domain.factories.copula_factory import CopulaFactory
   from CopulaFurtif.core.copulas.application.use_cases.fit_copula import FitCopulaUseCase

   copula = CopulaFactory.create("gumbel")
   data = [[0.2, 0.3], [0.5, 0.6], [0.9, 0.8], ...]  # liste de paires (X, Y)

   result = FitCopulaUseCase().fit_cmle(data, copula)
   print("Paramètres optimaux :", copula.parameters)
   print("Log-vraisemblance :", copula.log_likelihood_)


📦 Exemple : MLE avec marges
-----------------------------

.. code-block:: python

   from CopulaFurtif.core.copulas.application.use_cases.fit_copula import FitCopulaUseCase
   from CopulaFurtif.core.copulas.domain.estimation.marginals import fit_marginals

   marginals = fit_marginals(data, family="normal")
   result = FitCopulaUseCase().fit_mle(data, copula, marginals)


🔁 Exemple : IFM
----------------

.. code-block:: python

   from CopulaFurtif.core.copulas.application.use_cases.fit_copula import FitCopulaUseCase
   from CopulaFurtif.core.copulas.domain.estimation.marginals import fit_marginals

   marginals = fit_marginals(data, family="normal")
   result = FitCopulaUseCase().fit_ifm(data, copula, marginals)


🔍 Options d'optimisation
--------------------------

- Méthodes utilisées : `SLSQP`, `Powell`, `L-BFGS-B`, etc.
- Les options peuvent être passées via `FitCopulaUseCase(..., options={...})`


📌 Conseils
-----------

- Vérifiez que la copule est bien "fittée" (`copula.log_likelihood_` non nul)
- Adaptez le choix de la méthode selon la nature des données (brutes ou uniformisées)
- Vérifiez les bornes des paramètres (`copula.bounds_param`)