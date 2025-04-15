def test_partial_derivative_C_wrt_v(copula, u, v, eps=1e-6):
    """
    Compare ∂C(u,v)/∂v (numerical) with partial_derivative_C_wrt_v (analytical).

    Parameters
    ----------
    copula : object with get_cdf() and partial_derivative_C_wrt_v()
    u : float
    v : float
    eps : float
        Finite difference step for numerical derivative

    Returns
    -------
    ana : float
        Analytical partial derivative ∂C/∂v
    num : float
        Numerical estimate of ∂C/∂v via finite differences
    error : float
        Absolute error |ana - num|
    """
    param = copula.parameters
    cdf1 = copula.get_cdf(u, v, param)
    cdf2 = copula.get_cdf(u, min(v + eps, 1 - 1e-10), param)
    num = (cdf2 - cdf1) / eps
    ana = copula.partial_derivative_C_wrt_v(u, v, param)
    error = abs(num - ana)
    return ana, num, error

def test_partial_derivative_C_wrt_u(copula, u, v, eps=1e-6):
    """
    Compare ∂C(u,v)/∂u (numerical) with partial_derivative_C_wrt_u (analytical).

    Parameters
    ----------
    copula : object with get_cdf() and partial_derivative_C_wrt_u()
    u : float
    v : float
    eps : float
        Finite difference step for numerical derivative

    Returns
    -------
    ana : float
        Analytical partial derivative ∂C/∂u
    num : float
        Numerical estimate of ∂C/∂u via finite differences
    error : float
        Absolute error |ana - num|
    """
    param = copula.parameters
    cdf1 = copula.get_cdf(u, v, param)
    cdf2 = copula.get_cdf(min(u + eps, 1 - 1e-10), v, param)
    num = (cdf2 - cdf1) / eps
    ana = copula.partial_derivative_C_wrt_u(u, v, param)
    error = abs(num - ana)
    return ana, num, error


def test_partial_derivative_C_wrt_v_order2(copula, u, v, eps=1e-4):
    """
    Différence centrée d'ordre 2 pour approximer ∂C(u,v)/∂v.

    Parameters
    ----------
    copula : objet avec get_cdf() et partial_derivative_C_wrt_v()
    u, v : float
        Point d'évaluation dans [0,1].
    eps : float
        Pas pour la différence finie (ici recommandé autour de 1e-4).

    Returns
    -------
    ana : float
        Dérivée partielle analytique ∂C/∂v.
    num : float
        Approximation numérique de ∂C/∂v.
    error : float
        Erreur absolue |ana - num|.
    """
    param = copula.parameters
    # S'assurer que v ± eps restent dans [0,1]
    v_plus = min(v + eps, 1 - 1e-10)
    v_minus = max(v - eps, 1e-10)
    h = v_plus - v_minus  # pas effectif
    cdf_plus = copula.get_cdf(u, v_plus, param)
    cdf_minus = copula.get_cdf(u, v_minus, param)
    num = (cdf_plus - cdf_minus) / h
    ana = copula.partial_derivative_C_wrt_v(u, v, param)
    error = abs(num - ana)
    return ana, num, error


def test_partial_derivative_C_wrt_u_order2(copula, u, v, eps=1e-4):
    """
    Différence centrée d'ordre 2 pour approximer ∂C(u,v)/∂u.

    Parameters
    ----------
    copula : objet avec get_cdf() et partial_derivative_C_wrt_u()
    u, v : float
        Point d'évaluation dans [0,1].
    eps : float
        Pas pour la différence finie.

    Returns
    -------
    ana : float
        Dérivée partielle analytique ∂C/∂u.
    num : float
        Approximation numérique de ∂C/∂u.
    error : float
        Erreur absolue |ana - num|.
    """
    param = copula.parameters
    # S'assurer que u ± eps restent dans [0,1]
    u_plus = min(u + eps, 1 - 1e-10)
    u_minus = max(u - eps, 1e-10)
    h = u_plus - u_minus
    f_plus = copula.get_cdf(u_plus, v, param)
    f_minus = copula.get_cdf(u_minus, v, param)
    num = (f_plus - f_minus) / h
    ana = copula.partial_derivative_C_wrt_u(u, v, param)
    error = abs(num - ana)
    return ana, num, error


def test_partial_derivative_C_wrt_v_order4(copula, u, v, eps=1e-4):
    """
    Différence centrée d'ordre 4 pour approximer ∂C(u,v)/∂v.

    Parameters
    ----------
    copula : objet avec get_cdf() et partial_derivative_C_wrt_v()
    u, v : float
        Point d'évaluation dans [0,1].
    eps : float
        Pas pour la différence finie (utilisé ici pour définir v ± eps et v ± 2eps).

    Returns
    -------
    ana : float
        Dérivée partielle analytique ∂C/∂v.
    num : float
        Approximation numérique de ∂C/∂v en utilisant le schéma d'ordre 4.
    error : float
        Erreur absolue |ana - num|.
    """
    param = copula.parameters
    # Définir les points en gérant les bornes
    v_plus2 = min(v + 2 * eps, 1 - 1e-10)
    v_plus = min(v + eps, 1 - 1e-10)
    v_minus = max(v - eps, 1e-10)
    v_minus2 = max(v - 2 * eps, 1e-10)

    f_plus2 = copula.get_cdf(u, v_plus2, param)
    f_plus = copula.get_cdf(u, v_plus, param)
    f_minus = copula.get_cdf(u, v_minus, param)
    f_minus2 = copula.get_cdf(u, v_minus2, param)

    num = (-f_plus2 + 8 * f_plus - 8 * f_minus + f_minus2) / (12 * eps)
    ana = copula.partial_derivative_C_wrt_v(u, v, param)
    error = abs(num - ana)
    return ana, num, error


def test_partial_derivative_C_wrt_u_order4(copula, u, v, eps=1e-4):
    """
    Différence centrée d'ordre 4 pour approximer ∂C(u,v)/∂u.

    Parameters
    ----------
    copula : objet avec get_cdf() et partial_derivative_C_wrt_u()
    u, v : float
        Point d'évaluation dans [0,1].
    eps : float
        Pas pour la différence finie (utilisé ici pour définir u ± eps et u ± 2eps).

    Returns
    -------
    ana : float
        Dérivée partielle analytique ∂C/∂u.
    num : float
        Approximation numérique de ∂C/∂u en utilisant le schéma d'ordre 4.
    error : float
        Erreur absolue |ana - num|.
    """
    param = copula.parameters
    # Définir les points en gérant les bornes
    u_plus2 = min(u + 2 * eps, 1 - 1e-10)
    u_plus = min(u + eps, 1 - 1e-10)
    u_minus = max(u - eps, 1e-10)
    u_minus2 = max(u - 2 * eps, 1e-10)

    f_plus2 = copula.get_cdf(u_plus2, v, param)
    f_plus = copula.get_cdf(u_plus, v, param)
    f_minus = copula.get_cdf(u_minus, v, param)
    f_minus2 = copula.get_cdf(u_minus2, v, param)

    num = (-f_plus2 + 8 * f_plus - 8 * f_minus + f_minus2) / (12 * eps)
    ana = copula.partial_derivative_C_wrt_u(u, v, param)
    error = abs(num - ana)
    return ana, num, error



