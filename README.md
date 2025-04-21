[![codecov](https://codecov.io/gh/roadto7k/CopulaFurtif/branch/main/graph/badge.svg)](https://codecov.io/gh/roadto7k/CopulaFurtif)
# CopulaFurtif

Un projet modulaire de copules bivariées suivant l'architecture hexagonale avec support complet du fitting, des diagnostics, des visualisations et des tests.

## 📦 Fonctionnalités principales

- ✅ Création dynamique de copules via `CopulaFactory`
- 🧠 Algorithmes d’estimation : `CMLE`, `MLE`, `IFM`
- 📊 Scores et métriques : AIC, BIC, Kendall Tau error
- 📈 Visualisations : PDF/CDF, heatmaps, arbitrage frontiers
- 🧪 Tests unitaires et d’intégration
- 🧱 SOLID + Clean architecture hexagonale

## 📁 Structure (simplifiée)

```
core/
├── domain/
│   └── models/, estimation/, factories/
├── application/
│   └── use_cases/, services/
├── infra/
│   └── registry/, visualization/
example/
├── example_gaussian_refactored.py
├── example_viz_refactored.py
├── example_Residual_heatmap_refactored.py
```

---

## 🚀 Lancer un exemple

```bash
python example/example_gaussian_refactored.py
```

---

## 🧪 Tester le projet

```bash
# Tous les tests + couverture
make test

# Tests séparés
make unit
make integration

# Couverture HTML
make coverage-html
```

---

## ✅ Requis

- Python 3.9+ recommandé
- Dépendances : voir `requirements.txt`

```bash
pip install -r requirements.txt
```

---

## 📊 Badge couverture (local)

Généré via :
```bash
make coverage-html
```
→ ouvre `htmlcov/index.html`

Pour badge GitHub : `codecov` ou `coveralls` à intégrer dans CI.

---

## 📌 À venir
- [ ] Support complet pour de nouvelles copules, etc.
- [ ] API FastAPI ou Streamlit pour interface
- [ ] Ajout d'un comparateur automatique

---

## 🤝 Contribuer
- Architecture claire et extensible
- Nouvelles copules : hériter de `CopulaModel` et enregistrer dans `CopulaFactory`
- Tests → placer dans `tests/unit/` ou `tests/integration/`