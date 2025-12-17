import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import os

# ERROR HANDLING HERE: float conversion
def safe_float(expr):
    expr_s = sp.simplify(expr)
    if len(expr_s.free_symbols) == 0:
        return float(sp.N(expr_s))
    return expr_s


def evaluate_candidate(F, g_list, x1, x2, sol_dict, lam_syms):
    subs = {x1: sol_dict[x1], x2: sol_dict[x2]}
    Fv = safe_float(F.subs(subs))

    out = {
        "x1": safe_float(sol_dict[x1]),
        "x2": safe_float(sol_dict[x2]),
        "F(x)": Fv
    }

    # evaluate all inequalities
    for i, g in enumerate(g_list):
        gv = safe_float(g.subs(subs))
        lamv = safe_float(sol_dict.get(lam_syms[i], 0))
        slack = None
        if isinstance(gv, (int, float)):
            slack = max(0.0, -gv)
        out[f"g{i}(x)"] = gv
        out[f"lambda{i}"] = lamv
        out[f"slack{i}"] = slack
    return out


#     Returns list of candidate solutions (dicts) using simple active-set enumeration
#     for inequalities (each inequality can be active or inactive).
#     For this lab we only need up to 1 inequality in given problems, but code supports many.

def solve_kkt(F, g_list, h_list, is_maximize, x1, x2):

    # convert objective to minimization form if needed
    f = -F if is_maximize else F

    lam_syms = sp.symbols(f'lam0:{len(g_list)}', real=True)
    mu_syms  = sp.symbols(f'mu0:{len(h_list)}', real=True)

    candidates = []

    # enumerate active sets for inequalities
    m = len(g_list)
    for mask in range(2**m):
        L = f
        for i in range(m):
            L += lam_syms[i] * g_list[i]
        for j in range(len(h_list)):
            L += mu_syms[j] * h_list[j]

        eqs = [
            sp.Eq(sp.diff(L, x1), 0),
            sp.Eq(sp.diff(L, x2), 0),
        ]

        for h in h_list:
            eqs.append(sp.Eq(h, 0))

        extra_subs = {}
        for i in range(m):
            is_active = (mask >> i) & 1
            if is_active:
                eqs.append(sp.Eq(g_list[i], 0))     # active => g=0
            else:
                extra_subs[lam_syms[i]] = 0         # inactive => lambda=0

        # apply inactive lambda=0 substitutions to equations
        eqs_subbed = [e.subs(extra_subs) for e in eqs]

        # unknowns to solve for
        unknowns = [x1, x2]
        # keep only multipliers not fixed
        unknowns += [lam_syms[i] for i in range(m) if lam_syms[i] not in extra_subs]
        unknowns += list(mu_syms)

        sol_list = sp.solve(eqs_subbed, unknowns, dict=True)

        # KKT feasibility checks:
        # 1) primal: g<=0
        # 2) dual: lambda>=0
        # (Complementary slackness is satisfied by construction via active/inactive cases)

        for sol in sol_list:
            sol_full = dict(sol)
            for k, v in extra_subs.items():
                sol_full[k] = v

            subs = {x1: sol_full[x1], x2: sol_full[x2]}
            primal_ok = True
            dual_ok = True
            for i in range(m):
                gv = sp.N(g_list[i].subs(subs))
                lv = sp.N(sol_full[lam_syms[i]])
                # tolerance
                if float(gv) > 1e-10:
                    primal_ok = False
                if float(lv) < -1e-10:
                    dual_ok = False

            if primal_ok and dual_ok:
                candidates.append((sol_full, lam_syms))

    return candidates


def plot_and_save(problem_id, F, g_list, h_list, candidates, x1, x2):
    x1_vals = np.linspace(-6, 6, 400)
    x2_vals = np.linspace(-6, 6, 400)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    F_func = sp.lambdify((x1, x2), F, 'numpy')
    Z = F_func(X1, X2)

    plt.figure(figsize=(8, 6))
    cs = plt.contour(X1, X2, Z, levels=15)
    plt.clabel(cs, inline=True, fontsize=8)

    # Plot inequality boundary
    if len(g_list) == 1:
        g = sp.simplify(g_list[0])
        # If g = x1+x2-4 <=0  -> boundary x2 = 4-x1
        if sp.simplify(g - (x1 + x2 - 4)) == 0:
            x_line = np.linspace(-6, 6, 200)
            plt.plot(x_line, 4 - x_line, linewidth=2)
            mask = (X1 + X2 <= 4)
            plt.imshow(mask.astype(float),
                       extent=[x1_vals.min(), x1_vals.max(), x2_vals.min(), x2_vals.max()],
                       origin='lower', alpha=0.15, aspect='auto')
        # If g = 4 - x1 - x2 <=0  -> x1+x2 >=4 boundary same line
        elif sp.simplify(g - (4 - x1 - x2)) == 0:
            x_line = np.linspace(-6, 6, 200)
            plt.plot(x_line, 4 - x_line, linewidth=2)
            mask = (X1 + X2 >= 4)
            plt.imshow(mask.astype(float),
                       extent=[x1_vals.min(), x1_vals.max(), x2_vals.min(), x2_vals.max()],
                       origin='lower', alpha=0.15, aspect='auto')

    # Plot equality line(s) roughly (only for 2D, 1 equality)
    if len(h_list) == 1:
        h = sp.simplify(h_list[0])
        # if h = x1 - x2 - 2 = 0 -> x2 = x1 - 2
        if sp.simplify(h - (x1 - x2 - 2)) == 0:
            x_line = np.linspace(-6, 6, 200)
            plt.plot(x_line, x_line - 2, linewidth=2)

        # if h = x1 + x2 - 4 = 0 -> x2 = 4 - x1
        if sp.simplify(h - (x1 + x2 - 4)) == 0:
            x_line = np.linspace(-6, 6, 200)
            plt.plot(x_line, 4 - x_line, linewidth=2)

    for (sol, lam_syms) in candidates:
        x1v = float(sp.N(sol[x1]))
        x2v = float(sp.N(sol[x2]))
        plt.scatter([x1v], [x2v], s=80)

    plt.title(f"Problem {problem_id}: Level curves and constraints")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)

    figures_dir = os.path.join("..", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    fig_path = os.path.join(figures_dir, f"problem{problem_id}_kkt.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved to: {fig_path}")

    plt.show()


def run_problem(problem_id, F, g_list, h_list, is_maximize, x1, x2):
    print("\n" + "="*60)
    print(f"PROBLEM {problem_id}")
    print("="*60)

    # Solve KKT properly (active/inactive)
    candidates = solve_kkt(F, g_list, h_list, is_maximize, x1, x2)

    print("\nKKT candidates (feasible + dual-feasible):")
    for sol, lam_syms in candidates:
        cand = evaluate_candidate(F, g_list, x1, x2, sol, lam_syms)
        print(cand)

    # Plot + save
    plot_and_save(problem_id, F, g_list, h_list, candidates, x1, x2)


def main():
    x1, x2 = sp.symbols('x1 x2', real=True)

    # Problem 1
    # F = 4x1^2 + 3x2^2 - 5x1x2 - 8
    # subject to x1 + x2 <= 4  -> g = x1 + x2 - 4 <= 0
    F1 = 4*x1**2 + 3*x2**2 - 5*x1*x2 - 8
    g1 = [x1 + x2 - 4]
    run_problem(1, F1, g1, [], is_maximize=True, x1=x1, x2=x2)

    # Problem 2
    # F = 4x1^2 + 3x2^2 - 5x1x2 - 8x1
    # subject to x1 + x2 <= 4
    F2 = 4*x1**2 + 3*x2**2 - 5*x1*x2 - 8*x1
    g2 = [x1 + x2 - 4]
    run_problem(2, F2, g2, [], is_maximize=True, x1=x1, x2=x2)

    # Problem 3
    # f = (x1-1)^2 + (x2-1)^2
    # subject to x1 + x2 >= 4  -> g = 4 - x1 - x2 <= 0
    #      x1 - x2 - 2 = 0 -> h=0
    F3 = (x1 - 1)**2 + (x2 - 1)**2
    g3 = [4 - x1 - x2]
    h3 = [x1 - x2 - 2]
    run_problem(3, F3, g3, h3, is_maximize=False, x1=x1, x2=x2)

    # Problem 4
    # f = 4x1^2 + 3x2^2 - 5x1x2 - 8x1
    # subject to x1 + x2 = 4  -> h=0
    F4 = 4*x1**2 + 3*x2**2 - 5*x1*x2 - 8*x1
    h4 = [x1 + x2 - 4]
    run_problem(4, F4, [], h4, is_maximize=False, x1=x1, x2=x2)

if __name__ == "__main__":
    main()