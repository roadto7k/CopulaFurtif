.. CopulaFurtif documentation master file

Welcome to CopulaFurtif's documentation!
========================================

.. image:: images/saucisson.jpg
   :align: center
   :scale: 50 %

CopulaFurtif is a modular pipeline for modeling, fitting, and analyzing bivariate copulas.

---

📚 Available Sections:

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   pages/installation
   pages/usage
   pages/fitting
   pages/visualization
   pages/extending

.. toctree::
   :maxdepth: 1
   :caption: API References

.. toctree::
   :maxdepth: 2

   api/modules


---

🚀 Quickstart
-------------

Here's a quick example:

.. code-block:: python

   from CopulaFurtif import CopulaFactory
   from CopulaFurtif.core.copulas.application.use_cases.fit_copula import FitCopulaUseCase

   copula = CopulaFactory.create("gaussian")
   data = [[0.1, 0.2], [0.4, 0.5], [0.8, 0.9]]  # example

   fit_result = FitCopulaUseCase().fit_cmle(data, copula)
   print("Params:", copula.parameters)
