import numpy as np

def conjugate_gradient(grad, H, x0, tol=1e-12, max_iter=50):
    """
    Minimizes the given function

    grad : callable
        Gradient function grad(x).
    H : np.ndarray
        Hessian matrix of the quadratic function.
    x0 : np.ndarray
        Initial point.
    tol : float, optional
        Stopping tolerance based on gradient norm.
    max_iter : int, optional
        Maximum number of iterations.

    x : np.ndarray
        Final solution.
    iterations : int
        Number of iterations performed.
    path : np.ndarray
        Sequence of iterates.
    """
    x = x0.astype(float).copy()
    g = grad(x)
    d = -g
    path = [x.copy()]
    k = 0

    while np.linalg.norm(g) > tol and k < max_iter:
        denom = float(d @ H @ d)
        if abs(denom) < 1e-15:
            break

        alpha = -float(g @ d) / denom
        x_new = x + alpha * d
        g_new = grad(x_new)

        g_norm_sq = float(g @ g)
        if g_norm_sq < 1e-15:
            x = x_new
            path.append(x.copy())
            k += 1
            break

        beta = float(g_new @ g_new) / g_norm_sq
        d = -g_new + beta * d

        x, g = x_new, g_new
        path.append(x.copy())
        k += 1

    return x, k, np.array(path)