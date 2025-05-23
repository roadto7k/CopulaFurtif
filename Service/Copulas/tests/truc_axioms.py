import numpy as np
# === Manual import of copulas ===
from Service.Copulas.archimedean.AMH import AMHCopula

from Service.Copulas.archimedean.BB1 import BB1Copula
from Service.Copulas.archimedean.BB2 import BB2Copula
from Service.Copulas.archimedean.BB3 import BB3Copula
from Service.Copulas.archimedean.BB4 import BB4Copula
from Service.Copulas.archimedean.BB5 import BB5Copula
from Service.Copulas.archimedean.BB6 import BB6Copula
from Service.Copulas.archimedean.BB7 import BB7Copula
from Service.Copulas.archimedean.BB8 import BB8Copula
from Service.Copulas.archimedean.TawnT1 import TawnT1Copula
from Service.Copulas.archimedean.TawnT2 import TawnT2Copula
from Service.Copulas.archimedean.clayton import ClaytonCopula
from Service.Copulas.archimedean.fgm import FGMCopula
from Service.Copulas.archimedean.frank import FrankCopula
from Service.Copulas.archimedean.galambos import GalambosCopula
from Service.Copulas.archimedean.gumbel import GumbelCopula
from Service.Copulas.archimedean.joe import JoeCopula
from Service.Copulas.archimedean.plackett import PlackettCopula
from Service.Copulas.elliptical.gaussian import GaussianCopula
from Service.Copulas.elliptical.student import StudentCopula
from Service.Copulas.exotic.husle_reiss import HuslerReissCopula
from Service.Copulas.exotic.marshall_olkin import MarshallOlkinCopula

# === List of copulas to test with their parameters ===
copulas_to_test = [
    (AMHCopula, [1.0]),
    (MarshallOlkinCopula, [0.5, 0.5]),
    (HuslerReissCopula, [1.0]),
    (BB1Copula, [2.0, 3.0]),
    (BB2Copula, [2.0, 3.0]),
    (BB3Copula, [2.0, 3.0]),
    (BB4Copula, [2.0, 3.0]),
    (BB5Copula, [2.0, 3.0]),
    (BB6Copula, [2.0, 3.0]),
    (BB7Copula, [2.0, 3.0]),
    (BB8Copula, [2.0, 0.5]),
    (TawnT1Copula, [2.0, 0.5]),
    (TawnT2Copula, [2.0, 0.5]),
    (ClaytonCopula, [2.0]),
    (FGMCopula, [0.5]),
    (FrankCopula, [5.0]),
    (GalambosCopula, [2.0]),
    (GumbelCopula, [1.5]),
    (JoeCopula, [2.5]),
    (PlackettCopula, [3.0]),
    (GaussianCopula, [0.7]),
    (StudentCopula, [0.6, 4]),
]

def run_axiom_test(copula_class, params):
    cop = copula_class()
    cop.parameters = np.array(params)
    results = cop.check_axioms(verbose=False)

    all_ok = True
    for axiom, passed in results.items():
        if not passed:
            print(f"❌ {cop.name} | Param: {params} | Failed Axiom: {axiom}")
            all_ok = False

    if all_ok:
        print(f"✅ {cop.name} passed all axioms.")

if __name__ == "__main__":
    print("=== Running copula axiom checks ===\n")
    for copula_class, params in copulas_to_test:
        run_axiom_test(copula_class, params)
