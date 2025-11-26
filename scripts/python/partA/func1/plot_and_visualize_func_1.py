import os
import numpy as np
import matplotlib.pyplot as plt

# Objective function
# f(x,y) = 2x^2 + y^2 - 2xy - 3x - 2y

def objective_function(X, Y):
    return 2*X**2 + Y**2 - 2*X*Y - 3*X - 2*Y


# Constraints
# y - x <= 0        (g1)
# x^2 + y^2 - 1 = 0 (circle boundary)

def constraints(X, Y):
    g1 = Y - X
    h1 = X**2 + Y**2 - 1
    feasible = (g1 <= 0) & (np.abs(h1) < 0.02)
    return g1, h1, feasible


# Contour plot with constraints
def plot_contour(X, Y, F, feasible, save_path):
    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, F, 30, linewidths=1.2)
    plt.title("Contour Plot of f(x, y) with Constraints")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)

    # Constraint 1: y = x
    plt.plot(X[0], X[0], 'r', linewidth=2, label="y = x")

    # Constraint 2: unit circle
    theta = np.linspace(0, 2*np.pi, 300)
    xc = np.cos(theta)
    yc = np.sin(theta)
    plt.plot(xc, yc, 'b', linewidth=2, label="x² + y² = 1")

    # Feasible region shading
    plt.scatter(
        X[feasible], Y[feasible],
        s=6, c='green', alpha=0.35, label="Feasible region"
    )

    plt.legend()
    plt.axis("equal")

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


# Surface plot with constraints
def plot_surface(X, Y, F, save_path):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, F, cmap='turbo', edgecolor='none', alpha=0.95)

    # Circle on surface
    theta = np.linspace(0, 2*np.pi, 300)
    xc = np.cos(theta)
    yc = np.sin(theta)
    zc = objective_function(xc, yc)
    ax.plot3D(xc, yc, zc, 'k', linewidth=2)

    ax.set_title("Surface Plot of f(x, y) with Constraints")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y)")

    # Save
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


def main():
    out_dir = "../../../../figures/python/func1"
    os.makedirs(out_dir, exist_ok=True)
    # Grid
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-2, 2, 400)
    X, Y = np.meshgrid(x, y)
    F = objective_function(X, Y)

    # Constraints
    g1, h1, feasible = constraints(X, Y)

    # Plots
    plot_contour(X, Y, F, feasible, os.path.join(out_dir, "fig_contour.png"))
    plot_surface(X, Y, F, os.path.join(out_dir, "fig_surface.png"))

if __name__ == "__main__":
    main()