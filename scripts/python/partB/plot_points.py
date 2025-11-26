import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import os


#  CLASSIFICATION
def classify_stationary_point(H, f_xx):
    D = H.det()

    if D > 0 and f_xx > 0:
        return "Local Minimum"
    elif D > 0 and f_xx < 0:
        return "Local Maximum"
    elif D < 0:
        return "Saddle Point"
    else:
        return "Degenerate / Inflection"

#  SYMBOLIC ANALYSIS
def analyze_function(f, x1, x2):
    """Analytically find stationary points and classify them."""
    print("========================================")
    print("Function:", f)
    print("========================================")

    # Gradient
    f_x1 = sp.diff(f, x1)
    f_x2 = sp.diff(f, x2)

    # Solve ∇f = 0
    stationary_points = sp.solve([f_x1, f_x2], (x1, x2), dict=True)

    print("\nStationary points:")
    for p in stationary_points:
        print("  →", p)

    # Hessian
    f_x1x1 = sp.diff(f_x1, x1)
    f_x1x2 = sp.diff(f_x1, x2)
    f_x2x2 = sp.diff(f_x2, x2)

    H = sp.Matrix([[f_x1x1, f_x1x2],
                   [f_x1x2, f_x2x2]])

    print("\nHessian matrix:")
    sp.pprint(H)
    print("det(H) =", H.det())

    # Classification
    results = []
    for p in stationary_points:
        subs_H = H.subs({x1: p[x1], x2: p[x2]})
        fxx_val = f_x1x1.subs({x1: p[x1], x2: p[x2]})
        classification = classify_stationary_point(subs_H, fxx_val)
        results.append((p[x1], p[x2], classification))

        print(f"\nAt {p}:")
        print("H =", subs_H)
        print("classification =", classification)

    return results


#  NUMERIC FUNCTIONS FOR PLOTTING
def f_numpy(func, X, Y, x1, x2):
    """Convert symbolic f(x1,x2) to a numpy-evaluable lambda."""
    f_lambda = sp.lambdify((x1, x2), func, "numpy")
    return f_lambda(X, Y)


def plot_contour_and_point(f, stationary_list, save_path, x1, x2, title):
    x = np.linspace(-5, 5, 500)
    y = np.linspace(-5, 5, 500)
    X, Y = np.meshgrid(x, y)
    F = f_numpy(f, X, Y, x1, x2)

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, F, 30, linewidths=1.0)
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)

    # stationary points
    for (px, py, cls) in stationary_list:
        plt.scatter(float(px), float(py), s=100, label=f"{cls}: ({px},{py})")

    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


def plot_surface_and_point(f, stationary_list, save_path, x1, x2, title):
    x = np.linspace(-5, 5, 300)
    y = np.linspace(-5, 5, 300)
    X, Y = np.meshgrid(x, y)
    F = f_numpy(f, X, Y, x1, x2)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, F, cmap="viridis", edgecolor="none", alpha=0.85)

    for (px, py, cls) in stationary_list:
        z = float(f_numpy(f, np.array([[px]]), np.array([[py]]), x1, x2))
        ax.scatter(float(px), float(py), z, s=60, label=cls)

    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x1,x2)")
    ax.legend()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


#  WRAPPERS FOR FUNC1 AND FUNC2
def run_func1(out_dir):
    x1, x2 = sp.symbols("x1 x2", real=True)
    f1 = 3*x1**2 + 2*x1*x2 + 2*x2**2 + 7

    results = analyze_function(f1, x1, x2)

    plot_contour_and_point(
        f1, results,
        save_path=os.path.join(out_dir, "func1_contour.png"),
        x1=x1, x2=x2,
        title="FUNC1 - Contour + Stationary Point"
    )

    plot_surface_and_point(
        f1, results,
        save_path=os.path.join(out_dir, "func1_surface.png"),
        x1=x1, x2=x2,
        title="FUNC1 - Surface + Stationary Point"
    )


def run_func2(out_dir):
    x1, x2 = sp.symbols("x1 x2", real=True)
    f2 = x1**2 + 4*x1*x2 + x2**2 + 3

    results = analyze_function(f2, x1, x2)

    plot_contour_and_point(
        f2, results,
        save_path=os.path.join(out_dir, "func2_contour.png"),
        x1=x1, x2=x2,
        title="FUNC2 - Contour + Stationary Point"
    )

    plot_surface_and_point(
        f2, results,
        save_path=os.path.join(out_dir, "func2_surface.png"),
        x1=x1, x2=x2,
        title="FUNC2 - Surface + Stationary Point"
    )
def main():
    out_dir = "./stationary_plots"
    os.makedirs(out_dir, exist_ok=True)

    print("\n===== RUNNING FUNC1 =====")
    run_func1(out_dir)

    print("\n===== RUNNING FUNC2 =====")
    run_func2(out_dir)


if __name__ == "__main__":
    main()