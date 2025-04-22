.. _installation:

Installation
============

Voici comment installer et configurer CopulaFurtif dans votre environnement local.


⚙️ Prérequis
------------

- Python >= 3.9
- `poetry` (recommandé) ou `pip`
- Unix-like OS recommandé (Linux/macOS)


📦 Installation avec Poetry (recommandée)
-----------------------------------------

.. code-block:: bash

   git clone https://github.com/roadto7k/CopulaFurtif.git
   cd CopulaFurtif
   poetry install
   poetry shell


📦 Installation avec pip (alternatif)
-------------------------------------

.. code-block:: bash

   git clone https://github.com/roadto7k/CopulaFurtif.git
   cd CopulaFurtif
   pip install -r requirements.txt


💡 (Optionnel) : installer `pre-commit`
----------------------------------------

.. code-block:: bash

   pre-commit install

Cela active le lint automatique avant chaque commit (PEP8, isort, black, etc).


🧪 Lancer les tests
-------------------

.. code-block:: bash

   make test         # ou: pytest tests/
   make coverage-html  # et ouvrir htmlcov/index.html


📚 Générer la documentation
----------------------------

.. code-block:: bash

   cd docs
   make html

La documentation est ensuite accessible dans `docs/_build/html/index.html`


📂 Arborescence simplifiée
--------------------------

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


✅ Et voilà, vous êtes prêt à explorer le monde fascinant des copules !