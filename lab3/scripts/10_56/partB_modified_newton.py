import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 10.56: f = 25x1^2 + 20x2^2 - 2x1 - x2 ; start (3,1)
H = np.array([[50.0, 0.0],
              [0.0,  40.0]])
b = np.array([2.0, 1.0])
x0 = np.array([3.0, 1.0])

def f(x):
    return 0.5 * float(x @ (H @ x)) - float(b @ x)

def grad(x):
    return H @ x - b

def hess(_x):
    return H

def ensure_descent_direction(Hk, ck, mu0=1e-10, factor=10.0, max_tries=30):
    n = Hk.shape[0]
    I = np.eye(n)
    mu = 0.0
    for _ in range(max_tries):
        Htry = Hk + mu * I
        d = np.linalg.solve(Htry, -ck)
        if float(ck @ d) < 0.0:
            return d, mu
        mu = mu0 if mu == 0.0 else mu * factor
    return -ck, mu

def exact_alpha_quadratic(ck, dk, Hk):
    denom = float(dk @ (Hk @ dk))
    if denom <= 0.0:
        return 1.0
    return -float(ck @ dk) / denom

def modified_newton(x_start, eps=1e-12, max_iter=50):
    x = x_start.copy()
    path = [x.copy()]
    k = 0
    while np.linalg.norm(grad(x)) > eps and k < max_iter:
        ck = grad(x)
        Hk = hess(x)
        dk, _mu = ensure_descent_direction(Hk, ck)
        alpha = exact_alpha_quadratic(ck, dk, Hk)
        x = x + alpha * dk
        path.append(x.copy())
        k += 1
    return x, f(x), k, np.array(path)

def save_figure(path):
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    out_dir = project_root / "figures" / "10_56"
    out_dir.mkdir(parents=True, exist_ok=True)

    X1, X2 = np.meshgrid(np.linspace(-0.1, 3.2, 220),
                         np.linspace(-0.1, 1.2, 220))
    h11, h12, h22 = float(H[0,0]), float(H[0,1]), float(H[1,1])
    b1, b2 = float(b[0]), float(b[1])
    Z = 0.5 * (h11*X1**2 + 2*h12*X1*X2 + h22*X2**2) - (b1*X1 + b2*X2)

    x_opt = np.linalg.solve(H, b)

    plt.figure(figsize=(6.6, 5.6))
    cs = plt.contour(X1, X2, Z, levels=25)
    plt.clabel(cs, inline=True, fontsize=8)

    plt.plot(path[:,0], path[:,1], "o-", label="Modified Newton path")
    plt.scatter(path[0,0], path[0,1], s=90, label=r"$x^{(0)}$")
    plt.scatter(x_opt[0], x_opt[1], s=140, marker="*", label=r"$x^*$")

    plt.title("Problem 10.56 – Modified Newton")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.grid(True)
    plt.legend()

    out_path = out_dir / "modified_newton.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Figure saved:", out_path)

if __name__ == "__main__":
    x_opt = np.linalg.solve(H, b)
    print("Analytic check: x* =", x_opt, " f(x*) =", f(x_opt))

    x_star, f_star, iters, path = modified_newton(x0)
    print("Modified Newton: x* =", x_star, " f(x*) =", f_star, " iterations =", iters)

    save_figure(path)