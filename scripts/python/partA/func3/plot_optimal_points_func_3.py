import os
import numpy as np
import matplotlib.pyplot as plt


# FUNC3 objective function
# f(x, y) = 9x^2 + 13y^2 + 18xy - 4
def f(x, y):
    return 9*x**2 + 13*y**2 + 18*x*y - 4


# Parametric feasible curve (the circle)
# (x+1)^2 + y^2 = 17
def feasible_curve(num_points=20000):
    t = np.linspace(0, 2*np.pi, num_points)
    R = np.sqrt(17)

    x = -1 + R * np.cos(t)
    y = R * np.sin(t)
    return x, y

def find_optimal_points():
    x, y = feasible_curve()
    vals = f(x, y)

    i_min = np.argmin(vals)
    i_max = np.argmax(vals)

    min_pt = (x[i_min], y[i_min], vals[i_min])
    max_pt = (x[i_max], y[i_max], vals[i_max])

    return min_pt, max_pt

# Plot contour with min/max
def plot_contour_with_points(X, Y, save_path, min_pt, max_pt):
    F = f(X, Y)

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, F, 40, linewidths=1.2)

    # constraint circle
    xc, yc = feasible_curve()
    plt.plot(xc, yc, 'r', linewidth=2, label="Constraint circle")

    # min & max
    plt.scatter(min_pt[0], min_pt[1], s=80, c='purple', label='Min point')
    plt.scatter(max_pt[0], max_pt[1], s=80, c='orange', label='Max point')

    plt.legend()
    plt.title("Contour Plot with Optimal Points (FUNC3)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


# Surface with min/max
def plot_surface_with_points(X, Y, save_path, min_pt, max_pt):
    F = f(X, Y)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, F, cmap='turbo', edgecolor='none', alpha=0.92)

    # constraint curve on surface
    xc, yc = feasible_curve()
    zc = f(xc, yc)
    ax.plot3D(xc, yc, zc, 'k', linewidth=2)

    # optimal points
    ax.scatter(min_pt[0], min_pt[1], min_pt[2], s=80, c='purple')
    ax.scatter(max_pt[0], max_pt[1], max_pt[2], s=80, c='orange')

    ax.set_title("Surface Plot with Optimal Points (FUNC3)")
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

    # Compute optimal points numerically
    min_pt, max_pt = find_optimal_points()

    print("\n=== FUNC3 Optimal Points ===")
    print("Min point:", min_pt)
    print("Max point:", max_pt)

    # Draw plots
    plot_contour_with_points(
        X, Y,
        os.path.join(out_dir, "fig_contour_optimal.png"),
        min_pt, max_pt
    )

    plot_surface_with_points(
        X, Y,
        os.path.join(out_dir, "fig_surface_optimal.png"),
        min_pt, max_pt
    )


if __name__ == "__main__":
    main()