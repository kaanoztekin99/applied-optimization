import os
import numpy as np
import matplotlib.pyplot as plt

# Objective function (FUNC2)
# f(x, y) = 4x^2 + 3y^2 - 5xy - 8x
def f(x, y):
    return 4*x**2 + 3*y**2 - 5*x*y - 8*x


# Feasible mask for visualization: points near x + y = 4
def feasible_mask(X, Y, tol=0.05):
    h1 = X + Y - 4
    return np.abs(h1) < tol


# We parametrize y = 4 - x, fit a quadratic to f(x, 4-x), and use vertex formula x* = -b / (2a).
def find_min_on_constraint(x_min=-10.0, x_max=10.0, num_samples=2000):
    xs = np.linspace(x_min, x_max, num_samples)
    ys = 4 - xs
    vals = f(xs, ys)

    # Fit a quadratic: vals ≈ a x^2 + b x + c
    coeffs = np.polyfit(xs, vals, 2)
    a, b, c = coeffs

    # If a > 0 -> convex -> global minimum
    x_star = -b / (2.0 * a)
    y_star = 4 - x_star
    f_star = f(x_star, y_star)

    return (x_star, y_star, f_star), a


# Plot contour + optimal point
def plot_contour_with_points(X, Y, save_path, min_point):
    F = f(X, Y)
    feasible = feasible_mask(X, Y)

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, F, 30, linewidths=1.2)

    # Constraint line
    x_line = np.linspace(-2, 6, 400)
    y_line = 4 - x_line
    plt.plot(x_line, y_line, 'r', linewidth=2, label="x + y = 4")

    # Feasible tube
    plt.scatter(X[feasible], Y[feasible], s=6, c='green',
                alpha=0.35, label="Near x + y = 4")

    # Optimal point (minimum)
    plt.scatter(min_point[0], min_point[1],
                color='purple', s=80, label="Min point")

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Contour Plot with Constraint and Minimum (FUNC2)")
    plt.axis('equal')
    plt.grid(True)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


# Plot surface + optimal point
def plot_surface_with_points(X, Y, save_path, min_point):
    F = f(X, Y)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, F, cmap='turbo', edgecolor='none', alpha=0.9)

    # Constraint line on the surface
    x_line = np.linspace(-2, 6, 400)
    y_line = 4 - x_line
    z_line = f(x_line, y_line)
    ax.plot3D(x_line, y_line, z_line, 'k', linewidth=2)

    # Minimum point
    ax.scatter(min_point[0], min_point[1], min_point[2],
               s=60, c='purple')

    ax.set_title("Surface Plot with Constraint and Minimum (FUNC2)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y)")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


def main():
    out_dir = "../../../../figures/python/func2"
    os.makedirs(out_dir, exist_ok=True)

    # Grid for visualization
    x = np.linspace(-2, 6, 500)
    y = np.linspace(-2, 6, 500)
    X, Y = np.meshgrid(x, y)

    min_point, a = find_min_on_constraint()

    print("Quadratic coefficient a =", a)
    if a > 0:
        print("Function along the line is convex -> global minimum exists.")
        print("Global minimum (on x + y = 4):")
        print("  x* = {:.6f}, y* = {:.6f}, f* = {:.6f}".format(
            min_point[0], min_point[1], min_point[2]))
        print("No finite global maximum (problem unbounded above along the line).")
    elif a < 0:
        print("Function along the line is concave -> global maximum would exist,")
        print("but minimum would be unbounded below.")
    else:
        print("Degenerate case: the quadratic term vanished.")

    plot_contour_with_points(
        X, Y,
        os.path.join(out_dir, "fig_contour_optimal.png"),
        min_point
    )

    plot_surface_with_points(
        X, Y,
        os.path.join(out_dir, "fig_surface_optimal.png"),
        min_point
    )


if __name__ == "__main__":
    main()