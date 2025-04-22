.. _visualization:

Visualisation
=============

CopulaFurtif propose plusieurs outils pour visualiser la qualité d'ajustement des copules.


🌡️ Heatmap des résidus (Empirical - Model)
------------------------------------------

.. code-block:: python

   from CopulaFurtif.core.copulas.infrastructure.visualization.copula_viz_adapter import plot_residual_heatmap

   u, v = pseudo_obs(data)
   plot_residual_heatmap(copula, u, v, bins=50)

Cela produit une carte des écarts entre la CDF empirique et celle du modèle.

.. image:: ../_static/heatmap_example.png
   :align: center
   :scale: 60 %


📈 Courbes conditionnelles
--------------------------

.. code-block:: python

   from CopulaFurtif.core.copulas.infrastructure.visualization.copula_viz_adapter import plot_conditional_curves

   plot_conditional_curves(copula, fixed_values=[0.25, 0.5, 0.75], kind="u_given_v")

.. image:: ../_static/conditional_curves.png
   :align: center
   :scale: 60 %


📊 Benchmark entre copules
--------------------------

.. code-block:: python

   from CopulaFurtif.core.copulas.infrastructure.visualization.copula_viz_adapter import plot_copula_comparison

   copulas = [CopulaFactory.create("gaussian"), CopulaFactory.create("gumbel")]
   for c in copulas:
       FitCopulaUseCase().fit_cmle(data, c)

   plot_copula_comparison(copulas, u, v)

.. image:: ../_static/comparison.png
   :align: center
   :scale: 60 %


🎯 Résumé visuel intégré
------------------------

.. code-block:: python

   from CopulaFurtif.core.copulas.infrastructure.visualization.copula_viz_adapter import full_copula_summary

   full_copula_summary(copula, data, bins=40)