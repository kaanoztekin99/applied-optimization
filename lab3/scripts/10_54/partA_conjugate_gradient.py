import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# f = 6.983 x1^2 + 12.415 x2^2 - x1 ; start (2,1)
# Quadratic form: f=0.5 x^T H x - b^T x
# H = diag([2*6.983, 2*12.415]) = diag([13.966, 24.83]), b=[1,0]

H = np.array([[13.966, 0.0],
              [0.0,   24.83]])
b = np.array([1.0, 0.0])
x0 = np.array([2.0, 1.0])

def f_quad(x):
    return 0.5 * float(x @ (H @ x)) - float(b @ x)

def grad(x):
    return H @ x - b

def cg_fr_exact(x_start, tol=1e-12, max_iter=50):
    x = x_start.copy()
    g = grad(x)
    d = -g
    path = [x.copy()]
    k = 0

    while np.linalg.norm(g) > tol and k < max_iter:
        alpha = -(g @ d) / (d @ H @ d)
        x_new = x + alpha * d
        g_new = grad(x_new)

        beta = (g_new @ g_new) / (g @ g)
        d = -g_new + beta * d
        x, g = x_new, g_new

        path.append(x.copy())
        k += 1

    return x, f_quad(x), k, np.array(path)

def save_figure(path):
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    out_dir = project_root / "figures" / "10_54"
    out_dir.mkdir(parents=True, exist_ok=True)

    # tighter window makes contours nicer here
    X1, X2 = np.meshgrid(np.linspace(-0.1, 2.2, 220),
                         np.linspace(-0.1, 1.2, 220))
    h11, h12, h22 = float(H[0,0]), float(H[0,1]), float(H[1,1])
    b1, b2 = float(b[0]), float(b[1])
    Z = 0.5 * (h11*X1**2 + 2*h12*X1*X2 + h22*X2**2) - (b1*X1 + b2*X2)

    x_opt = np.linalg.solve(H, b)

    plt.figure(figsize=(6.6, 5.6))
    cs = plt.contour(X1, X2, Z, levels=25)
    plt.clabel(cs, inline=True, fontsize=8)

    plt.plot(path[:,0], path[:,1], "o-", label="CG path")
    plt.scatter(path[0,0], path[0,1], s=90, label=r"$x^{(0)}$")
    plt.scatter(x_opt[0], x_opt[1], s=140, marker="*", label=r"$x^*$")

    plt.title("Problem 10.54 – Conjugate Gradient (FR)")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.grid(True)
    plt.legend()

    out_path = out_dir / "cg_path.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Figure saved:", out_path)

if __name__ == "__main__":
    x_opt = np.linalg.solve(H, b)
    print("Analytic check: x* =", x_opt, " f(x*) =", f_quad(x_opt))

    x_star, f_star, iters, path = cg_fr_exact(x0)
    print("CG result:     x* =", x_star, " f(x*) =", f_star, " iterations =", iters)

    save_figure(path)