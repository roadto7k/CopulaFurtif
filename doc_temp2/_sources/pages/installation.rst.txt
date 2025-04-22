.. _installation:

Installation
============

Here's how to install and configure CopulaFurtif in your local environment.


⚙️ Prerequisites
----------------

- Python >= 3.9
- `poetry` (recommended) or `pip`
- Unix-like OS recommended (Linux/macOS)


📦 Installation with Poetry (recommended)
-----------------------------------------

.. code-block:: bash

   git clone https://github.com/roadto7k/CopulaFurtif.git
   cd CopulaFurtif
   poetry install
   poetry shell


📦 Installation with pip (alternative)
--------------------------------------

.. code-block:: bash

   git clone https://github.com/roadto7k/CopulaFurtif.git
   cd CopulaFurtif
   pip install -r requirements.txt


💡 (Optional): install `pre-commit`
-----------------------------------

.. code-block:: bash

   pre-commit install

This enables automatic linting before each commit (PEP8, isort, black, etc).


🧪 Run the tests
----------------

.. code-block:: bash

   make test         # or: pytest tests/
   make coverage-html  # then open htmlcov/index.html


📚 Generate the documentation
-----------------------------

.. code-block:: bash

   cd docs
   make html

The documentation will then be available in `docs/_build/html/index.html`


📂 Simplified folder structure
------------------------------

.. code-block:: text

   CopulaFurtif/
   ├── core/
   │   └── copulas/
   │       ├── domain/
   │       ├── application/
   │       └── infrastructure/
   ├── tests/
   ├── docs/
   └── pyproject.toml


✅ And that’s it — you're ready to explore the fascinating world of copulas!