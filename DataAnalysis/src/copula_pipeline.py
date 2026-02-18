from CopulaFurtif.copulas import CopulaFactory, CopulaFitter, CopulaType
from CopulaFurtif.core.copulas.domain.estimation.estimation import pseudo_obs
from CopulaFurtif.core.copulas.domain.estimation.tail_dependance import compare_tail_dependence
from CopulaFurtif.visualization import MatplotlibCopulaVisualizer
from CopulaFurtif.copulas import CopulaFitter, CopulaDiagnostics
from DataAnalysis.config import COPULA_TYPE, SEED
import numpy as np
from scipy.stats import beta, lognorm, t as student_t

def run_copula_analysis(spread1, spread2):
    np.random.seed(SEED)
    data = [spread1.values, spread2.values]

    copula_enum = getattr(CopulaType, COPULA_TYPE.upper())
    copula = CopulaFactory.create(copula_enum)
    copula.set_parameters([0.5, 4.0])

    fitter = CopulaFitter()
    result = fitter.fit_cmle(data, copula)

    candidate_list = [CopulaFactory.create(CopulaType.STUDENT), CopulaFactory.create(CopulaType.FRANK), CopulaFactory.create(CopulaType.BB1), CopulaFactory.create(CopulaType.AMH)]
    for copula in candidate_list:
        try:
            true_rho, true_nu = 0.7, 7
            marginals_known = [
                {"distribution": beta, "a": 2, "b": 5, "loc": 0, "scale": 1},
                {"distribution": lognorm, "s": 0.5, "loc": 0, "scale": np.exp(1)}
            ]
                    
            CopulaFitter().fit_mle
            fitted_params, loglik = CopulaFitter().fit_mle(data, copula, marginals=marginals_known,
                                            known_parameters=True)
            copula.set_parameters(np.array(fitted_params))
            print(f"{copula.get_name()} fitted successfully:")
            print(f"  Fitted parameters: {fitted_params}")
            print(f"  Log-likelihood: {loglik:.4f}")
        except Exception as e:
            print(f"MLE fitting for {copula.get_name()} failed: {e}")

    
    result_df = compare_tail_dependence(data, candidate_list,
                                                 q_low=0.05, q_high=0.95, verbose=True)
    print(result_df)

    MatplotlibCopulaVisualizer.plot_tail_dependence(data, candidate_list, q_low=0.05, q_high=0.95)


    if result:
        print(result)
    else:
        print("Copula fitting failed.")
