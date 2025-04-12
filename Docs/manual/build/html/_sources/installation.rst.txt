.. index:: Installation and Deployment

Installation and Deployment
===========================

Standard Installation
---------------------

**CopulaFurtif** is a Python library requiring Python >= 3.10. It depends on the following *external* Python libraries:

.. include:: dependencies.rst

.. In addition, the Event Generation module requires:

.. - `Pythia 8.3 <https://arxiv.org/abs/1410.3012>`_ (requires a C++ compiler)
.. - `MadGraph5_aMC@NLO <https://arxiv.org/abs/1405.0301>`_ (optional, for advanced event generation)
.. - `HepMC3 <https://hepmc.web.cern.ch/hepmc>`_ (used by Pythia)

Installation Methods
--------------------

You can install all the necessary python packages and the librairy using pip installation method.

To install the software, use:

.. code-block:: bash

    pip install -e CopulaFurtif




Deployment Notes
----------------

Ensure that you have the necessary compilers installed:

- Python installed on your computer

Once the installation is complete, the software will be ready for event generation and other analyses.
