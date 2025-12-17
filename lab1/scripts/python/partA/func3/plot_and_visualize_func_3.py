import os
import numpy as np
import matplotlib.pyplot as plt

# Objective function FUNC3
# f(x, y) = 9x^2 + 13y^2 + 18xy - 4
def objective_function(X, Y):
    return 9*X**2 + 13*Y**2 + 18*X*Y - 4


# Constraint: (x+1)^2 + y^2 = 17
def constraint(X, Y, tol=0.05):
    h1 = (X + 1)**2 + Y**2 - 17
    feasible = np.abs(h1) < tol
    return h1, feasible


# Contour plot with constraint circle
def plot_contour(X, Y, F, feasible, save_path):
    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, F, 40, linewidths=1.2)
    plt.title("Contour Plot of f(x, y) with Constraint (FUNC3)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)

    # Constraint circle
    t = np.linspace(0, 2*np.pi, 400)
    xc = -1 + np.sqrt(17) * np.cos(t)
    yc = np.sqrt(17) * np.sin(t)
    plt.plot(xc, yc, 'r', linewidth=2, label="(x+1)² + y² = 17")
    plt.scatter(X[feasible], Y[feasible],
                s=6, c='green', alpha=0.35)

    plt.legend()
    plt.axis("equal")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


# Surface plot with constraint curve
def plot_surface(X, Y, F, save_path):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, F, cmap='turbo', edgecolor='none', alpha=0.95)

    # constraint curve on surface
    t = np.linspace(0, 2*np.pi, 400)
    xc = -1 + np.sqrt(17) * np.cos(t)
    yc = np.sqrt(17) * np.sin(t)
    zc = objective_function(xc, yc)
    ax.plot3D(xc, yc, zc, 'k', linewidth=2)

    ax.set_title("Surface Plot of f(x,y) with Constraint (FUNC3)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y)")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


def main():
    out_dir = "../../../../figures/python/func3"
    os.makedirs(out_dir, exist_ok=True)

    # grid
    x = np.linspace(-10, 10, 600)
    y = np.linspace(-10, 10, 600)
    X, Y = np.meshgrid(x, y)

    F = objective_function(X, Y)

    h1, feasible = constraint(X, Y)

    plot_contour(X, Y, F, feasible, os.path.join(out_dir, "fig_contour.png"))
    plot_surface(X, Y, F, os.path.join(out_dir, "fig_surface.png"))


if __name__ == "__main__":
    main()