import os
import numpy as np
import matplotlib.pyplot as plt

# Objective function for FUNC2:
# f(x, y) = 4x^2 + 3y^2 - 5xy - 8x
def objective_function(X, Y):
    return 4*X**2 + 3*Y**2 - 5*X*Y - 8*X


# Constraint: x + y = 4
def constraint(X, Y, tol=0.05):
    h1 = X + Y - 4
    feasible = np.abs(h1) < tol
    return h1, feasible


# Contour plot with constraint
def plot_contour(X, Y, F, feasible, save_path):
    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, F, 30, linewidths=1.2)
    plt.title("Contour Plot of f(x, y) with Constraint x + y = 4 (FUNC2)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)

    # Constraint line: x + y = 4  ->  y = 4 - x
    x_line = np.linspace(-2, 6, 400)
    y_line = 4 - x_line
    plt.plot(x_line, y_line, 'r', linewidth=2, label="x + y = 4")

    # Feasible on grid (just for visualization)
    plt.scatter(
        X[feasible], Y[feasible],
        s=6, c='green', alpha=0.35, label="Points close to x + y = 4"
    )

    plt.legend()
    plt.axis("equal")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


# Surface plot with constraint line
def plot_surface(X, Y, F, save_path):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, F, cmap='turbo', edgecolor='none', alpha=0.95)

    # Constraint line on the surface: x + y = 4  -> parametrize in x
    x_line = np.linspace(-2, 6, 400)
    y_line = 4 - x_line
    z_line = objective_function(x_line, y_line)
    ax.plot3D(x_line, y_line, z_line, 'k', linewidth=2)

    ax.set_title("Surface Plot of f(x, y) with Constraint x + y = 4 (FUNC2)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y)")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


def main():
    out_dir = "../../../../figures/python/func2"
    os.makedirs(out_dir, exist_ok=True)

    # Grid (choose a reasonable window)
    x = np.linspace(-2, 6, 400)
    y = np.linspace(-2, 6, 400)
    X, Y = np.meshgrid(x, y)
    F = objective_function(X, Y)

    # Constraint mask
    h1, feasible = constraint(X, Y)

    # Plots
    plot_contour(X, Y, F, feasible, os.path.join(out_dir, "fig_contour.png"))
    plot_surface(X, Y, F, os.path.join(out_dir, "fig_surface.png"))


if __name__ == "__main__":
    main()