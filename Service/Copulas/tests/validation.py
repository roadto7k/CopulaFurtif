def test_conditional_cdf_u_given_v(copula, u, v, eps=1e-6):
    """
    Compare ∂C(u,v)/∂v (numerical) with conditional_cdf_u_given_v (analytical).
    """
    param = copula.parameters
    cdf1 = copula.get_cdf(u, v, param)
    cdf2 = copula.get_cdf(u, min(v + eps, 1 - 1e-10), param)
    num = (cdf2 - cdf1) / eps
    ana = copula.conditional_cdf_u_given_v(u, v, param)
    error = abs(num - ana)
    return ana, num, error

def test_conditional_cdf_v_given_u(copula, v, u, eps=1e-6):
    """
    Compare ∂C(u,v)/∂u (numerical) with conditional_cdf_v_given_u (analytical).
    """
    param = copula.parameters
    cdf1 = copula.get_cdf(u, v, param)
    cdf2 = copula.get_cdf(min(u + eps, 1 - 1e-10), v, param)
    num = (cdf2 - cdf1) / eps
    ana = copula.conditional_cdf_v_given_u(v, u, param)
    error = abs(num - ana)
    return ana, num, error


