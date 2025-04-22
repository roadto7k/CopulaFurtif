.. _usage:

Basic Usage
===========

This section guides you through using the `CopulaFurtif` pipeline to create, manipulate, and diagnose bivariate copulas.


🧱 Creating Copulas
-------------------

All copulas are accessible via the `CopulaFactory`:

.. code-block:: python

   from CopulaFurtif.core.copulas.domain.factories.copula_factory import CopulaFactory

   copula = CopulaFactory.create("gaussian")
   print(copula.name)  # Gaussian Copula

Available copulas: `gaussian`, `student`, `clayton`, `frank`, `joe`, `gumbel`, `amh`, `tawn3`, `galambos`, `plackett`, `fgm`, etc.


📊 Input Data
-------------

The pipeline generally expects:

- **Raw data**: original data for Kendall's tau (`[[X1, Y1], [X2, Y2], ...]`)
- **Pseudo-observations**: data transformed to uniform scale `u, v ∈ (0,1)` using marginals

Generate pseudo-observations:

.. code-block:: python

   from CopulaFurtif.core.copulas.domain.estimation.utils import pseudo_obs

   u, v = pseudo_obs(data)  # data = [[X1, Y1], [X2, Y2], ...]


📈 Accessing Basic Methods
--------------------------

.. code-block:: python

   copula.parameters = [0.5]       # or [rho, nu] for Student
   print(copula.get_cdf(0.4, 0.8))
   print(copula.get_pdf(0.4, 0.8))
   print(copula.kendall_tau())

   samples = copula.sample(100)


🔬 Diagnostics
--------------

.. code-block:: python

   from CopulaFurtif.core.copulas.application.services.diagnostics_service import DiagnosticService

   diag = DiagnosticService()
   scores = diag.evaluate(data, copula)
   print(scores)

Result: a dict with `LogLik`, `AIC`, `BIC`, `Kendall Tau Error`, etc.


📌 Coming Soon: Fitting & Visualization