import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import product
from mpl_toolkits.mplot3d import Axes3D


def solve_problem5_kkt():
    x1, x2, x3 = sp.symbols("x1 x2 x3", real=True)
    f = (x1 - 2)**2 + (x2 - 1)**2 + (x3 - 3)**2
    g1 = 6 - x1 - x2 - x3           # x1+x2+x3 >= 6
    g2 = x1 - 2*x2 - 1              # x1 - 2x2 <= 1
    g3 = 1 - x3                     # x3 >= 1
    g_list = [g1, g2, g3]

    # Multipliers
    l1, l2, l3 = sp.symbols("l1 l2 l3", real=True)
    l_syms = [l1, l2, l3]

    gradf = sp.Matrix([sp.diff(f, v) for v in (x1, x2, x3)])
    gradg = [sp.Matrix([sp.diff(g, v) for v in (x1, x2, x3)]) for g in g_list]

    candidates = []

    # Active-set enumeration for 3 inequalities
    for mask in product([0, 1], repeat=3):
        # Stationarity: grad f + sum lambda_i grad g_i = 0
        station = gradf + l1*gradg[0] + l2*gradg[1] + l3*gradg[2]

        eqs = [sp.Eq(station[0], 0), sp.Eq(station[1], 0), sp.Eq(station[2], 0)]
        subs_fixed = {}

        # Active/inactive handling
        # active -> g_i = 0
        # inactive -> lambda_i = 0
        for i, active in enumerate(mask):
            if active:
                eqs.append(sp.Eq(g_list[i], 0))
            else:
                subs_fixed[l_syms[i]] = 0

        eqs = [e.subs(subs_fixed) for e in eqs]

        unknowns = [x1, x2, x3] + [l_syms[i] for i, a in enumerate(mask) if a]
        sol_list = sp.solve(eqs, unknowns, dict=True)

        for sol in sol_list:
            sol_full = {x1: sol[x1], x2: sol[x2], x3: sol[x3]}
            for lam in l_syms:
                sol_full[lam] = subs_fixed.get(lam, sol.get(lam, 0))

            # KKT feasibility checks
            g_vals = [float(sp.N(g.subs(sol_full))) for g in g_list]
            l_vals = [float(sp.N(sol_full[lam])) for lam in l_syms]

            primal_ok = all(gv <= 1e-9 for gv in g_vals)
            dual_ok = all(lv >= -1e-9 for lv in l_vals)
            comp_ok = all(abs(l_vals[i]*g_vals[i]) <= 1e-6 for i in range(3))

            if primal_ok and dual_ok and comp_ok:
                f_val = float(sp.N(f.subs(sol_full)))
                candidates.append((mask, sol_full, f_val, g_vals, l_vals))

    # Picking the best
    candidates.sort(key=lambda t: t[2])
    best = candidates[0] if candidates else None

    return (x1, x2, x3, f, g_list, l_syms, candidates, best)


def sample_feasible_points(n=3000, seed=0):
    rng = np.random.default_rng(seed)

    # a simple bounding box for visualization
    X1 = rng.uniform(-2, 6, n)
    X2 = rng.uniform(-2, 6, n)
    X3 = rng.uniform(0, 8, n)

    # constraints:
    # x1+x2+x3 >= 6
    # x1 - 2x2 <= 1
    # x3 >= 1
    mask = (X1 + X2 + X3 >= 6) & (X1 - 2*X2 <= 1) & (X3 >= 1)
    return X1[mask], X2[mask], X3[mask]


def main():
    x1, x2, x3, f, g_list, l_syms, candidates, best = solve_problem5_kkt()

    print("\n============================================================")
    print("PROBLEM 5 (KKT via active-set enumeration)")
    print("============================================================")
    print("Objective f(x1,x2,x3) =", f)
    print("Constraints (g<=0):")
    for i, g in enumerate(g_list, start=1):
        print(f"  g{i}(x) = {g} <= 0")

    if best is None:
        print("\nNo feasible KKT candidate found (unexpected for this convex problem).")
        return

    mask, sol_full, f_val, g_vals, l_vals = best

    x_star = (float(sp.N(sol_full[x1])), float(sp.N(sol_full[x2])), float(sp.N(sol_full[x3])))
    print("\nBest KKT solution:")
    print("  Active-set mask (g1,g2,g3) =", mask, " (1=active, 0=inactive)")
    print("  x* =", x_star)
    print("  f(x*) =", f_val)

    # multipliers + slack
    print("\nMultipliers and slacks:")
    for i in range(3):
        slack = max(0.0, -g_vals[i])  # s = -g(x)
        print(f"  lambda{i+1} = {l_vals[i]:.6g},  g{i+1}(x*) = {g_vals[i]:.6g},  s{i+1} = {slack:.6g},  s{i+1}^2 = {slack**2:.6g}")

    X1, X2, X3 = sample_feasible_points(n=25000, seed=1)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X1, X2, X3, s=2, alpha=0.15)
    ax.scatter([x_star[0]], [x_star[1]], [x_star[2]], s=120, marker="*", label="KKT optimum")

    ax.set_title("Problem 5: Feasible region samples and KKT optimum")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    ax.legend()

    figures_dir = os.path.join("..", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    fig_path = os.path.join(figures_dir, "problem5_kkt.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved to: {fig_path}")
    plt.show()


if __name__ == "__main__":
    main()