import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import os


def main():
    # minimize f(x) = x^4 - 2x^2 (unconstrained)
    x = sp.Symbol('x', real=True)
    f = x**4 - 2*x**2
    fp = sp.diff(f, x)
    fpp = sp.diff(fp, x)

    crit = sp.solve(sp.Eq(fp, 0), x)  # [ -1, 0, 1 ] (order may vary)
    crit_sorted = sorted([sp.N(c) for c in crit], key=float)

    print("f(x) =", f)
    print("f'(x) =", fp)
    print("f''(x) =", fpp)
    print("\nStationary points:", crit_sorted)

    # Classify using 2nd derivative test
    print("\nClassification:")
    for c in crit_sorted:
        c_val = float(c)
        f_val = float(sp.N(f.subs(x, c)))
        fpp_val = float(sp.N(fpp.subs(x, c)))

        if fpp_val > 1e-9:
            typ = "local minimum"
        elif fpp_val < -1e-9:
            typ = "local maximum"
        else:
            typ = "inconclusive (f''=0)"

        print(f"  x = {c_val: .6g} | f(x) = {f_val: .6g} | f''(x) = {fpp_val: .6g} -> {typ}")

    f_vals = [(float(c), float(sp.N(f.subs(x, c)))) for c in crit_sorted]
    min_val = min(v for _, v in f_vals)
    argmins = [c for c, v in f_vals if abs(v - min_val) < 1e-9]
    print(f"\nGlobal minimum value among stationary points: {min_val} at x = {argmins}")

    # Plot and save
    xs = np.linspace(-2.5, 2.5, 800)
    f_np = sp.lambdify(x, f, 'numpy')
    ys = f_np(xs)

    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys)
    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)
    plt.grid(True)
    plt.title("Problem 6: f(x)=x^4-2x^2 and stationary points")
    plt.xlabel("x")
    plt.ylabel("f(x)")

    # mark stationary points
    for c, v in f_vals:
        plt.scatter([c], [v], s=80)
        plt.annotate(f"({c:.2g}, {v:.2g})", (c, v), textcoords="offset points", xytext=(8, 8))

    figures_dir = os.path.join("..", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    fig_path = os.path.join(figures_dir, "problem6.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved to: {fig_path}")

    plt.show()


if __name__ == "__main__":
    main()