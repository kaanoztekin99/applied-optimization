import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import product


def solve_kkt_active_sets():
    x1, x2 = sp.symbols("x1 x2", real=True)
    #(minimize)
    f = (x1 - 3)**2 + (x2 - 2)**2

    # Constraints in standard form g(x) <= 0
    # x1 >= 2  -> g1 = 2 - x1 <= 0
    # x2 >= 1  -> g2 = 1 - x2 <= 0
    # x1 + x2 <= 4 -> g3 = x1 + x2 - 4 <= 0
    g1 = 2 - x1
    g2 = 1 - x2
    g3 = x1 + x2 - 4
    g_list = [g1, g2, g3]

    l1, l2, l3 = sp.symbols("l1 l2 l3", real=True)
    l_syms = [l1, l2, l3]
    L = f + l1*g1 + l2*g2 + l3*g3
    eq_base = [
        sp.Eq(sp.diff(L, x1), 0),
        sp.Eq(sp.diff(L, x2), 0),
    ]

    candidates = []

    # Enumerate active sets
    for mask in product([0, 1], repeat=3):
        eqs = list(eq_base)
        subs_fixed = {}

        # active -> add gi=0
        # inactive -> set lambda_i=0
        for i, active in enumerate(mask):
            if active:
                eqs.append(sp.Eq(g_list[i], 0))
            else:
                subs_fixed[l_syms[i]] = 0

        eqs = [e.subs(subs_fixed) for e in eqs]

        unknowns = [x1, x2] + [l_syms[i] for i, a in enumerate(mask) if a]
        sol_list = sp.solve(eqs, unknowns, dict=True)

        for sol in sol_list:
            sol_full = {x1: sol[x1], x2: sol[x2]}
            for lam in l_syms:
                sol_full[lam] = subs_fixed.get(lam, sol.get(lam, 0))

            # KKT checks: primal (g<=0), dual (lambda>=0), comp slackness
            subs_xy = {x1: sol_full[x1], x2: sol_full[x2]}
            g_vals = [float(sp.N(g.subs(subs_xy))) for g in g_list]
            l_vals = [float(sp.N(sol_full[lam])) for lam in l_syms]

            primal_ok = all(gv <= 1e-9 for gv in g_vals)
            dual_ok = all(lv >= -1e-9 for lv in l_vals)
            comp_ok = all(abs(g_vals[i]*l_vals[i]) <= 1e-6 for i in range(3))

            if primal_ok and dual_ok and comp_ok:
                f_val = float(sp.N(f.subs(subs_xy)))
                candidates.append((mask, sol_full, f_val, g_vals, l_vals))

    candidates.sort(key=lambda t: t[2])
    best = candidates[0] if candidates else None
    return x1, x2, f, g_list, l_syms, candidates, best


def plot_and_save(x1, x2, f, g_list, best, out_name="problem7_kkt.png"):
    f_func = sp.lambdify((x1, x2), f, "numpy")
    x1_vals = np.linspace(1.5, 4.5, 500)
    x2_vals = np.linspace(0.5, 3.5, 500)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = f_func(X1, X2)

    plt.figure(figsize=(8, 6))
    cs = plt.contour(X1, X2, Z, levels=18)
    plt.clabel(cs, inline=True, fontsize=8)

    # Constraint lines
    plt.plot([2, 2], [x2_vals.min(), x2_vals.max()], linewidth=2)
    plt.plot([x1_vals.min(), x1_vals.max()], [1, 1], linewidth=2)
    x_line = np.linspace(x1_vals.min(), x1_vals.max(), 400)
    plt.plot(x_line, 4 - x_line, linewidth=2)

    # Feasible region mask
    mask = (X1 >= 2) & (X2 >= 1) & (X1 + X2 <= 4)
    plt.imshow(
        mask.astype(float),
        extent=[x1_vals.min(), x1_vals.max(), x2_vals.min(), x2_vals.max()],
        origin="lower",
        alpha=0.15,
        aspect="auto",
    )

    # optimum
    if best is not None:
        _, sol_full, f_val, g_vals, l_vals = best
        x_star = float(sp.N(sol_full[x1]))
        y_star = float(sp.N(sol_full[x2]))
        plt.scatter([x_star], [y_star], s=120)
        plt.annotate(
            f"optimum\n({x_star:.2f},{y_star:.2f})",
            (x_star, y_star),
            textcoords="offset points",
            xytext=(10, 10),
        )

    plt.title("Problem 7: Level curves and feasible region")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(script_dir, "..", "figures")
    os.makedirs(figures_dir, exist_ok=True)

    fig_path = os.path.join(figures_dir, out_name)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved to: {fig_path}")
    plt.show()


def main():
    x1, x2, f, g_list, l_syms, candidates, best = solve_kkt_active_sets()

    print("\n============================================================")
    print("PROBLEM 7 (KKT via active-set enumeration)")
    print("============================================================")
    print("Objective:", f)
    print("Constraints (g<=0):")
    print("  g1 = 2 - x1 <= 0   (x1 >= 2)")
    print("  g2 = 1 - x2 <= 0   (x2 >= 1)")
    print("  g3 = x1 + x2 - 4 <= 0   (x1 + x2 <= 4)")

    if best is None:
        print("\nNo feasible KKT candidate found.")
        return

    mask, sol_full, f_val, g_vals, l_vals = best
    x_star = float(sp.N(sol_full[x1]))
    y_star = float(sp.N(sol_full[x2]))

    print("\nBest KKT solution:")
    print("  Active-set mask (g1,g2,g3) =", mask, "(1=active, 0=inactive)")
    print(f"  x* = ({x_star:.6g}, {y_star:.6g})")
    print(f"  f(x*) = {f_val:.6g}")

    print("\nFinal report (constraints, multipliers, slack):")
    for i, (gv, lv) in enumerate(zip(g_vals, l_vals), start=1):
        slack = max(0.0, -gv)
        active = abs(gv) <= 1e-8
        print(
            f"  g{i} value = {gv:.6g} | active? {active} | "
            f"lambda{i} = {lv:.6g} | slack s{i} = {slack:.6g} | s{i}^2 = {slack**2:.6g}"
        )

    plot_and_save(x1, x2, f, g_list, best, out_name="problem7_kkt.png")


if __name__ == "__main__":
    main()