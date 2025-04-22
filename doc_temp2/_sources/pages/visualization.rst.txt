.. _visualization:

Visualization
=============

CopulaFurtif provides several tools to visualize the quality of copula fitting.


🌡️ Residual Heatmap (Empirical - Model)
---------------------------------------

.. code-block:: python

   from CopulaFurtif.core.copulas.infrastructure.visualization.copula_viz_adapter import plot_residual_heatmap

   u, v = pseudo_obs(data)
   plot_residual_heatmap(copula, u, v, bins=50)

This produces a map of differences between the empirical CDF and the model CDF.

.. .. image:: ../_static/heatmap_example.png
..    :align: center
..    :scale: 60 %


📈 Conditional Curves
---------------------

.. code-block:: python

   from CopulaFurtif.core.copulas.infrastructure.visualization.copula_viz_adapter import plot_conditional_curves

   plot_conditional_curves(copula, fixed_values=[0.25, 0.5, 0.75], kind="u_given_v")

.. .. image:: ../_static/conditional_curves.png
..    :align: center
..    :scale: 60 %


📊 Copula Benchmarking
-----------------------

.. code-block:: python

   from CopulaFurtif.core.copulas.infrastructure.visualization.copula_viz_adapter import plot_copula_comparison

   copulas = [CopulaFactory.create("gaussian"), CopulaFactory.create("gumbel")]
   for c in copulas:
       FitCopulaUseCase().fit_cmle(data, c)

   plot_copula_comparison(copulas, u, v)

.. .. image:: ../_static/comparison.png
..    :align: center
..    :scale: 60 %


🎯 Integrated Visual Summary
----------------------------

.. code-block:: python

   from CopulaFurtif.core.copulas.infrastructure.visualization.copula_viz_adapter import full_copula_summary

   full_copula_summary(copula, data, bins=40)