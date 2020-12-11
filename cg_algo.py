def hscg(A, b, x0, niter=1000, tol=1e-6):
    """
    This is the Python implementation CG-Algorithm 
    via exact line search
    
    :param A: numpy array, the A matrix in 
              the quadratic decomposition
    :param b: numpy array, the b matrix in 
              the quadratic decomposition
    :param x0: numpy array, the starting point
    :param niter: maximum number of iterations, 
                 default is 1000
    :param tol: the tolerance criteria to 
                terminate the algo, default is 1e-6
    :return: the x value corresponding to the minimum point
    """

    x_ = x0
    gk_ = A @ x_ + b
    d_ = -gk_
    for iter_ in range(niter):
        lambda_ = (- gk_ @ d_) / (d_ @ A @ d_)
        if (np.abs(gk_) <= tol).all():
            return x_
        else:
            x_ += lambda_ * d_
            gk1_ = A @ x_ + b
            gamma_k1 = (gk1_ @ gk1_) / (gk_ @ gk_)
            d_ = -gk1_ + gamma_k1 * d_
            gk_ = gk1_
    return x_