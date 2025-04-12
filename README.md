# CopulaFurtif

CopulaFurtif est un projet permettant d’effectuer des analyses statistiques et du trading basé sur des **copulas** et, plus largement, de l’analyse de données financières. 

## Table des matières

- [Fonctionnalités](#fonctionnalités)
- [Architecture](#architecture)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Tests](#tests)
- [Contribuer](#contribuer)
- [License](#license)

### Fonctionnalités

- **Fit de copulas** (Archimedean, Elliptical, etc.)  
- **Calcul de metrics** (AIC, BIC, IAD-score, AD-score, etc.)  
- **Analyse du tail dependence**  
- **Module Trading** (work in progress) : signaux, gestion de portefeuille, backtests (futurs développements)  
- **Module ML** (work in progress) : feature engineering, modèles ML sur données financières 

### Architecture

Le projet suit une approche inspirée de l’architecture hexagonale (Ports & Adapters).  
- Le **dossier `domain/`** contient la logique métier : [copulas](copulafurtif/domain/copulas/), [metrics](copulafurtif/domain/copulas/metrics/), un futur [module ML](copulafurtif/domain/ml/), etc.  
- Le **dossier `application/`** gère les cas d’usage concrets (use cases) et définit des interfaces (ports).  
- Le **dossier `infrastructure/`** contient les DAO et connecteurs concrets (Yahoo Finance, API Binance, etc.).  
- Le **dossier `interface/`** propose des interfaces de présentation (CLI, Web).  
- Les **tests** sont dans le répertoire [tests/](tests/).

### Installation

1. Cloner le repo :
   ```bash
   git clone https://github.com/roadto7k/CopulaFurtif.git
   cd CopulaFurtif
