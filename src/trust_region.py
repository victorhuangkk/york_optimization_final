def trust_region(obj_func, jac, hess, x0, delta_0, niter= 100):
    """
    the approximate method to solve the trust region sub problem
    
    :param obj_func: scalar function, the original objective function
    :param jac: first order derivative
    :param hess: second order derivative
    :param x0: the initial guess of x
    :param delta_0: the designated trust region. 
    :param niter: the maximum number of iterations
    :return: d(lambda_0), which is the designated return of this algo
    """

    delta_ = delta_0
    lambda_ = 0
    identy_matrix = np.eye(len(x0))
    d_lambda = np.array([0,0])
    hessian = hess(x0)
    gradient = jac(x0)
    L= np.linalg.cholesky(hessian)
    for i in range(niter):
        d_lambda = -np.linalg.inv(hessian + lambda_*identy_matrix) @ gradient
        d_lambda_norm = np.linalg.norm(d_lambda)
        print(d_lambda_norm)
        if 0.75*delta_0 <= d_lambda_norm <= 1.5*delta_0:
            break
        else:
            w= np.linalg.inv(L) @ d_lambda
            lambda_ += (1 - (d_lambda_norm)/(delta_))*(-d_lambda_norm**2/(w.T @ w))
    return d_lambda