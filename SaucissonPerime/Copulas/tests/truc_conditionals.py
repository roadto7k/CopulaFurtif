import numpy as np

from SaucissonPerime.Copulas.archimedean.AMH import AMHCopula
# === Manual import of copulas ===
from SaucissonPerime.Copulas.archimedean.BB1 import BB1Copula
from SaucissonPerime.Copulas.archimedean.BB2 import BB2Copula
from SaucissonPerime.Copulas.archimedean.BB3 import BB3Copula
from SaucissonPerime.Copulas.archimedean.BB4 import BB4Copula
from SaucissonPerime.Copulas.archimedean.BB5 import BB5Copula
from SaucissonPerime.Copulas.archimedean.BB6 import BB6Copula
from SaucissonPerime.Copulas.archimedean.BB7 import BB7Copula
from SaucissonPerime.Copulas.archimedean.BB8 import BB8Copula
from SaucissonPerime.Copulas.archimedean.Tawn import TawnCopula
from SaucissonPerime.Copulas.archimedean.TawnT1 import TawnT1Copula
from SaucissonPerime.Copulas.archimedean.TawnT2 import TawnT2Copula
from SaucissonPerime.Copulas.archimedean.clayton import ClaytonCopula
from SaucissonPerime.Copulas.archimedean.fgm import FGMCopula
from SaucissonPerime.Copulas.archimedean.frank import FrankCopula
from SaucissonPerime.Copulas.archimedean.galambos import GalambosCopula
from SaucissonPerime.Copulas.archimedean.gumbel import GumbelCopula
from SaucissonPerime.Copulas.archimedean.joe import JoeCopula
from SaucissonPerime.Copulas.archimedean.plackett import PlackettCopula
from SaucissonPerime.Copulas.elliptical.gaussian import GaussianCopula
from SaucissonPerime.Copulas.elliptical.student import StudentCopula
from SaucissonPerime.Copulas.exotic.husle_reiss import HuslerReissCopula
from SaucissonPerime.Copulas.exotic.marshall_olkin import MarshallOlkinCopula
from SaucissonPerime.Copulas.tests.validation import test_partial_derivative_C_wrt_u, test_partial_derivative_C_wrt_v, \
    test_partial_derivative_C_wrt_v_order2, test_partial_derivative_C_wrt_u_order2, \
    test_partial_derivative_C_wrt_v_order4, test_partial_derivative_C_wrt_u_order4

copulas_to_test = [
    (AMHCopula, [0.5]),
    (MarshallOlkinCopula, [0.5, 0.5]),
    (HuslerReissCopula, [1.0]),
    (BB1Copula, [2.0, 3.0]),
    (BB2Copula, [2.0, 3.0]),
    (BB3Copula, [2.0, 3.0]),
    (BB4Copula, [2.0, 3.0]),
    (BB5Copula, [2.0, 3.0]),
    (BB6Copula, [2.0, 3.0]),
    (BB7Copula, [2.0, 3.0]),
    (BB8Copula, [2.0, 0.7]),
    (TawnCopula, [2.0, 0.5, 0.5]),
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

def run_conditional_test(copula_class, param, tol=1e-4):
    cop = copula_class()
    cop.parameters = np.array(param)

    u_vals = np.linspace(0.1, 0.9, 5)
    v_vals = np.linspace(0.1, 0.9, 5)

    all_good = True
    for u in u_vals:
        for v in v_vals:
            try:
                ana1, num1, err1 = test_partial_derivative_C_wrt_v_order4(cop, u, v)
                ana2, num2, err2 = test_partial_derivative_C_wrt_u_order4(cop, u, v)
                if err1 > tol or err2 > tol:
                    print(f"❌ {cop.get_name()} | u={u:.2f}, v={v:.2f} | Δu={err1:.2e}, Δv={err2:.2e}")
                    all_good = False
            except Exception as e:
                print(f"❌ {cop.get_name()} | u={u:.2f}, v={v:.2f} | ERROR: {e}")
                all_good = False

    if all_good:
        print(f"✅ {cop.get_name()} passed conditional CDF test.")

if __name__ == "__main__":
    print("=== Running conditional CDF checks ===\n")
    for copula_class, params in copulas_to_test:
        run_conditional_test(copula_class, params)
