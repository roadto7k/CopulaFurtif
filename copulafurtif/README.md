[![codecov](https://codecov.io/gh/roadto7k/CopulaFurtif/graph/badge.svg?token=R2DEQCUUB1)](https://codecov.io/gh/roadto7k/CopulaFurtif)

# CopulaFurtif

CopulaFurtif is a modular bivariate copula library designed following the hexagonal architecture, providing comprehensive support for estimation, diagnostics, visualization, and testing.

---

## Features

- Dynamic copula creation via `CopulaFactory`
- Estimation methods: `CMLE`, `MLE`, `IFM`
- Model selection metrics: AIC, BIC, Kendall tau error
- Visualizations: PDF, CDF, residual heatmaps, arbitrage frontiers
- Unit and integration testing
- SOLID principles and hexagonal architecture

---

## Installation

To install the project in editable/development mode:

```bash
pip install -e copulafurtif
```

Dependencies are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
copulafurtif/
├── CopulaFurtif/├── core/
                     ├── copulas/ ├── domain/, infrastructure/, adapters/, ports/
                     └── common/
│                ├── examples/
│                └── tests/
                     ├── unit/
                     ├── integration/

```

---

## Quick Start

### Example: Gaussian copula fitting

```python
from copulafurtif.copulas import CopulaFactory, CopulaType, CopulaFitter
from copulafurtif.CopulaFurtif.core.DAO.generate_data_beta_lognorm import generate_data_beta_lognorm
from copulafurtif.copulas import CopulaDiagnostics
from copulafurtif.visualization import MatplotlibCopulaVisualizer
from copulafurtif.copula_utils import pseudo_obs
from scipy.stats import beta, lognorm
import numpy as np

np.random.seed(42)
data = generate_data_beta_lognorm(n=5000, rho=0.7)
copula = CopulaFactory.create(CopulaType.GAUSSIAN)

# Fit via CMLE
fitter = CopulaFitter()
params_cmle = fitter.fit_cmle(data, copula)

# Fit via MLE with known marginals
marginals = [
    {"distribution": beta, "a": 2, "b": 5, "loc": 0, "scale": 1},
    {"distribution": lognorm, "s": 0.5, "loc": 0, "scale": np.exp(1)},
]
params_mle, ll = fitter._fit_mle(data, copula, marginals, known_parameters=True)

# Diagnostics
diagnostics = CopulaDiagnostics()
report = diagnostics.evaluate(data, copula)

# Residual heatmap
u, v = pseudo_obs(data)
MatplotlibCopulaVisualizer.plot_residual_heatmap(copula, u, v)
```

---

### Example: Copula visualization

```python
from copulafurtif.visualization import plot_pdf, plot_cdf, plot_mpdf, plot_arbitrage_frontiers, Plot_type
from copulafurtif.copulas import CopulaFactory, CopulaType
from scipy.stats import norm
import numpy as np

copula = CopulaFactory.create(CopulaType.BB8)

plot_cdf(copula, plot_type=Plot_type.DIM3, Nsplit=100)
plot_pdf(copula, plot_type=Plot_type.CONTOUR, log_scale=True)
plot_mpdf(copula, [{'distribution': norm}, {'distribution': norm}], plot_type=Plot_type.CONTOUR)
plot_arbitrage_frontiers(copula, alpha_low=0.05, alpha_high=0.95)
```

---

## Testing

To run tests:

```bash
make test           # All tests
make unit           # Unit tests
make integration    # Integration tests
make coverage-html  # HTML coverage report
```

The coverage report will be generated at `htmlcov/index.html`.

---

## Code Coverage (CI/CD)

This project integrates with Codecov:

[![codecov](https://codecov.io/gh/roadto7k/CopulaFurtif/graph/badge.svg?token=R2DEQCUUB1)](https://codecov.io/gh/roadto7k/CopulaFurtif)

---

## Roadmap

- Full support for new copula families
- Web interface (FastAPI or Streamlit)
- Automatic model comparator

---

## Contributing

The codebase follows a clear and extensible architecture.

To add a new copula:

1. Inherit from the `CopulaModel` base class
2. Register it via `CopulaFactory.register(...)`
3. Write tests in `tests/unit` or `tests/integration`
