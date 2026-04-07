import numpy as np

def ensure_descent_direction(Hk, ck, mu0=1e-10, factor=10.0, max_tries=30):
    # Ensures a descent direction by modifying the Hessian if necessary.
    n = Hk.shape[0]
    I = np.eye(n)
    mu = 0.0

    for _ in range(max_tries):
        Htry = Hk + mu * I
        try:
            d = np.linalg.solve(Htry, -ck)
        except np.linalg.LinAlgError:
            mu = mu0 if mu == 0.0 else mu * factor
            continue

        if float(ck @ d) < 0.0:
            return d, mu

        mu = mu0 if mu == 0.0 else mu * factor

    return -ck, mu


def exact_alpha_quadratic(ck, dk, Hk):
    # Exact line search step size for quadratic functions.

    denom = float(dk @ (Hk @ dk))
    if denom <= 0.0:
        return 1.0
    return -float(ck @ dk) / denom


def modified_newton(f, grad, hess, x0, tol=1e-12, max_iter=50):
    """
    f : callable
        Objective function f(x).
    grad : callable
        Gradient function grad(x).
    hess : callable
        Hessian function hess(x).
    x0 : np.ndarray
        Initial point.
    tol : float, optional
        Stopping tolerance based on gradient norm.
    max_iter : int, optional
        Maximum number of iterations.

    x : np.ndarray
        Final solution.
    f_val : float
        Objective value at final solution.
    iterations : int
        Number of iterations.
    path : np.ndarray
        Sequence of iterates.
    """
    x = x0.astype(float).copy()
    path = [x.copy()]
    k = 0

    while np.linalg.norm(grad(x)) > tol and k < max_iter:
        ck = grad(x)
        Hk = hess(x)

        dk, _ = ensure_descent_direction(Hk, ck)
        alpha = exact_alpha_quadratic(ck, dk, Hk)

        x = x + alpha * dk
        path.append(x.copy())
        k += 1

    return x, f(x), k, np.array(path)