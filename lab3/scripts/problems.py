import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from conjugate_gradient import conjugate_gradient
from modified_newton import modified_newton

def get_problem_10_52():
    # f(x) = x1^2 + 2x2^2 - 4x1 - 2x1x2
    H = np.array([[2.0, -2.0],
                  [-2.0, 4.0]])
    b = np.array([4.0, 0.0])
    x0 = np.array([1.0, 1.0])

    def f(x):
        return 0.5 * float(x @ (H @ x)) - float(b @ x)

    def grad(x):
        return H @ x - b

    def hess(_x):
        return H

    return "10.52", f, grad, hess, H, b, x0


def get_problem_10_54():
    # f = 6.983 x1^2 + 12.415 x2^2 - x1 ; start (2,1)
    # Quadratic form: f=0.5 x^T H x - b^T x
    # H = diag([2*6.983, 2*12.415]) = diag([13.966, 24.83]), b=[1,0]
    H = np.array([[13.966, 0.0],
                  [0.0, 24.83]])
    b = np.array([1.0, 0.0])
    x0 = np.array([2.0, 1.0])

    def f(x):
        return 0.5 * float(x @ (H @ x)) - float(b @ x)

    def grad(x):
        return H @ x - b

    def hess(_x):
        return H

    return "10.54", f, grad, hess, H, b, x0


def get_problem_10_56():
    # f = 25 x1^2 + 20 x2^2 - 2 x1 - x2 ; start (3,1)
    # Quadratic form: f=0.5 x^T H x - b^T x
    # H = diag([50,40]), b=[2,1]
    H = np.array([[50.0, 0.0],
                  [0.0, 40.0]])
    b = np.array([2.0, 1.0])
    x0 = np.array([3.0, 1.0])

    def f(x):
        return 0.5 * float(x @ (H @ x)) - float(b @ x)

    def grad(x):
        return H @ x - b

    def hess(_x):
        return H

    return "10.56", f, grad, hess, H, b, x0


def save_contour_plot(problem_name, f, H, b, path_cg, path_mn):
    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    x_opt = np.linalg.solve(H, b)

    x1_min = min(path_cg[:, 0].min(), path_mn[:, 0].min(), x_opt[0]) - 1.0
    x1_max = max(path_cg[:, 0].max(), path_mn[:, 0].max(), x_opt[0]) + 1.0
    x2_min = min(path_cg[:, 1].min(), path_mn[:, 1].min(), x_opt[1]) - 1.0
    x2_max = max(path_cg[:, 1].max(), path_mn[:, 1].max(), x_opt[1]) + 1.0

    X1, X2 = np.meshgrid(
        np.linspace(x1_min, x1_max, 250),
        np.linspace(x2_min, x2_max, 250)
    )

    h11, h12, h22 = float(H[0, 0]), float(H[0, 1]), float(H[1, 1])
    b1, b2 = float(b[0]), float(b[1])
    Z = 0.5 * (h11 * X1**2 + 2 * h12 * X1 * X2 + h22 * X2**2) - (b1 * X1 + b2 * X2)

    plt.figure(figsize=(7, 6))
    cs = plt.contour(X1, X2, Z, levels=25)
    plt.clabel(cs, inline=True, fontsize=8)

    plt.plot(path_cg[:, 0], path_cg[:, 1], "o-", label="Conjugate Gradient")
    plt.plot(path_mn[:, 0], path_mn[:, 1], "s--", label="Modified Newton")

    plt.scatter(path_cg[0, 0], path_cg[0, 1], s=100, label=r"Start point $x^{(0)}$")
    plt.scatter(x_opt[0], x_opt[1], s=150, marker="*", label=r"Optimal point $x^*$")

    plt.title(f"Problem {problem_name} – Optimization Paths")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.grid(True)
    plt.legend()

    out_path = out_dir / f"problem_{problem_name.replace('.', '_')}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Figure saved: {out_path}")


def run_problem(problem_getter):
    problem_name, f, grad, hess, H, b, x0 = problem_getter()

    print(f"\n===== Problem {problem_name} =====")
    x_true = np.linalg.solve(H, b)
    print("Analytical solution:")
    print("x* =", x_true)
    print("f(x*) =", f(x_true))

    x_cg, it_cg, path_cg = conjugate_gradient(grad, H, x0)
    print("\nConjugate Gradient result:")
    print("x* =", x_cg)
    print("f(x*) =", f(x_cg))
    print("iterations =", it_cg)

    x_mn, f_mn, it_mn, path_mn = modified_newton(f, grad, hess, x0)
    print("\nModified Newton result:")
    print("x* =", x_mn)
    print("f(x*) =", f_mn)
    print("iterations =", it_mn)

    save_contour_plot(problem_name, f, H, b, path_cg, path_mn)


if __name__ == "__main__":
    run_problem(get_problem_10_52)
    run_problem(get_problem_10_54)
    run_problem(get_problem_10_56)