def coordinate_descent(A, bx_0, x_0,
                       tol=1e-3, max_iter=1000):
    x_ = x_0
    res = []
    f_x = lambda x: x.T@A@x + b@x
    jac_x = lambda x: A@x + b
    for iteration in range(max_iter):
        
        for i in range(6):
            if i == 0:
                gk = jac_x(x_)[i]
                print(gk)
                gk_ = np.array([gk, 0, 0, 0, 0, 0])
                lambda_ = -0.02
                x_ = x_+ lambda_*gk_
            elif i == 1:
                gk = jac_x(x_)[i]
                print(gk)
                gk_ = np.array([ 0, gk, 0, 0, 0, 0])
                lambda_ = -0.02
                x_ = x_+ lambda_*gk_
            elif i == 2:
                gk = jac_x(x_)[i]
                print(gk)
                gk_ = np.array([ 0, 0, gk,0, 0, 0])
                lambda_ = -0.02
                x_ = x_+ lambda_*gk_
            elif i == 3:
                gk = jac_x(x_)[i]
                print(gk)
                gk_ = np.array([ 0, 0,0,gk,0, 0])
                lambda_ = -0.02
                x_ = x_+ lambda_*gk_
                
            elif i == 4:
                gk = jac_x(x_)[i]
                print(gk)
                gk_ = np.array([ 0, 0, 0, 0, gk,0])
                lambda_ = -0.02
                x_ = x_+ lambda_*gk_
                
            elif i == 5:
                gk = jac_x(x_)[i]
                print(gk)
                gk_ = np.array([ 0, 0, 0, 0, 0, gk])
                lambda_ = -0.02
                x_ = x_+ lambda_*gk_
                
        if (gk <= tol) :
            break
                

        if iteration % 100 ==0:
            res.append(x_.tolist())
