import os
import numpy as np
import matplotlib.pyplot as plt

# Objective function
def f(x, y):
    return 2*x**2 + y**2 - 2*x*y - 3*x - 2*y


# Feasible region
def feasible_mask(X, Y):
    g1 = Y - X                # inequality y <= x
    h1 = X**2 + Y**2 - 1      # equality x^2 + y^2 = 1
    return (g1 <= 0) & (np.abs(h1) < 0.02)


# Plot contour + optimal points
def plot_contour_with_points(X, Y, save_path, min_point, max_point):
    F = f(X, Y)
    feasible = feasible_mask(X, Y)

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, F, 30, linewidths=1.2)
    plt.plot(X[0], X[0], 'r', linewidth=2, label="y = x")

    theta = np.linspace(0, 2*np.pi, 300)
    xc, yc = np.cos(theta), np.sin(theta)
    plt.plot(xc, yc, 'b', linewidth=2, label="x² + y² = 1")

    plt.scatter(X[feasible], Y[feasible], s=6, c='green', alpha=0.35)

    # Optimal points
    plt.scatter(min_point[0], min_point[1], color='purple', s=80, label="Min point")
    plt.scatter(max_point[0], max_point[1], color='orange', s=80, label="Max point")

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Contour Plot with Constraints and Optimum Points")
    plt.axis('equal')
    plt.grid(True)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


# Plot surface + optimal points
def plot_surface_with_points(X, Y, save_path, min_point, max_point):
    F = f(X, Y)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, F, cmap='turbo', edgecolor='none', alpha=0.92)

    # plot feasible curve (circle)
    theta = np.linspace(0, 2*np.pi, 300)
    xc, yc = np.cos(theta), np.sin(theta)
    ax.plot3D(xc, yc, f(xc, yc), 'k', linewidth=2)

    # plot optimum points
    ax.scatter(min_point[0], min_point[1], f(*min_point), s=60, c='purple')
    ax.scatter(max_point[0], max_point[1], f(*max_point), s=60, c='orange')

    ax.set_title("Surface Plot with Constraints and Optimum Points")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y)")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")

def main():
    out_dir = "../../../../figures/python/func1"
    os.makedirs(out_dir, exist_ok=True)

    x = np.linspace(-2, 2, 500)
    y = np.linspace(-2, 2, 500)
    X, Y = np.meshgrid(x, y)

    min_point = (1/np.sqrt(2), 1/np.sqrt(2))
    max_point = (-1/np.sqrt(2), -1/np.sqrt(2))

    plot_contour_with_points(X, Y, f"{out_dir}/fig_contour_optimal.png",
                             min_point, max_point)

    plot_surface_with_points(X, Y, f"{out_dir}/fig_surface_optimal.png",
                             min_point, max_point)


if __name__ == "__main__":
    main()