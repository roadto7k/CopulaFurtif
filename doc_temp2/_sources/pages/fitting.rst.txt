.. _fitting:

Copula Fitting
==============

This page describes how to fit a copula to your data using the CopulaFurtif pipeline.


🧪 Goal
-------

Find the best copula parameters that maximize the likelihood of the data.


⚙️ Available Tools
------------------

Three main fitting methods are supported:

- `CMLE`: Canonical Maximum Likelihood Estimation (with pseudo-observations)
- `MLE` : Maximum Likelihood on raw data + marginals
- `IFM` : Inference Function for Margins (two-step approach)


🚀 Example: CMLE
----------------

.. code-block:: python

   from CopulaFurtif.core.copulas.domain.factories.copula_factory import CopulaFactory
   from CopulaFurtif.core.copulas.application.use_cases.fit_copula import FitCopulaUseCase

   copula = CopulaFactory.create("gumbel")
   data = [[0.2, 0.3], [0.5, 0.6], [0.9, 0.8], ...]  # list of (X, Y) pairs

   result = FitCopulaUseCase().fit_cmle(data, copula)
   print("Optimal parameters:", copula.parameters)
   print("Log-likelihood:", copula.log_likelihood_)


📦 Example: MLE with marginals
------------------------------

.. code-block:: python

   from CopulaFurtif.core.copulas.application.use_cases.fit_copula import FitCopulaUseCase
   from CopulaFurtif.core.copulas.domain.estimation.marginals import fit_marginals

   marginals = fit_marginals(data, family="normal")
   result = FitCopulaUseCase().fit_mle(data, copula, marginals)


🔁 Example: IFM
---------------

.. code-block:: python

   from CopulaFurtif.core.copulas.application.use_cases.fit_copula import FitCopulaUseCase
   from CopulaFurtif.core.copulas.domain.estimation.marginals import fit_marginals

   marginals = fit_marginals(data, family="normal")
   result = FitCopulaUseCase().fit_ifm(data, copula, marginals)


🔍 Optimization Options
------------------------

- Supported methods: `SLSQP`, `Powell`, `L-BFGS-B`, etc.
- Options can be passed via `FitCopulaUseCase(..., options={...})`


📌 Tips
-------

- Make sure the copula is properly fitted (`copula.log_likelihood_` is not zero)
- Choose the fitting method based on the nature of your data (raw or transformed to uniform)
- Check the parameter bounds (`copula.bounds_param`)