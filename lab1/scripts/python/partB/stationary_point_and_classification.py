import sympy as sp

#  GENERAL STATIONARY POINT SOLVER + HESSIAN CLASSIFIER
def classify_stationary_point(H, f_xx):
    D = H.det()

    if D > 0 and f_xx > 0:
        return "Local Minimum"
    elif D > 0 and f_xx < 0:
        return "Local Maximum"
    elif D < 0:
        return "Saddle Point"
    else:
        return "Degenerate / Inflection (D = 0)"


def analyze_function(f, x1, x2):
    print("==============================================")
    print("Function:", f)
    print("==============================================")

    # Gradient
    f_x1 = sp.diff(f, x1)
    f_x2 = sp.diff(f, x2)

    print("\nGradient components:")
    print("f_x1 =", f_x1)
    print("f_x2 =", f_x2)

    # Solve system: gradient = 0
    stationary_points = sp.solve([f_x1, f_x2], (x1, x2), dict=True)

    print("\nStationary points found:")
    for p in stationary_points:
        print(" →", p)

    # Hessian
    f_x1x1 = sp.diff(f_x1, x1)
    f_x1x2 = sp.diff(f_x1, x2)
    f_x2x2 = sp.diff(f_x2, x2)

    H = sp.Matrix([[f_x1x1, f_x1x2],
                   [f_x1x2, f_x2x2]])

    print("\nHessian matrix:")
    sp.pprint(H)
    print("det(H) =", H.det())

    # Classification
    print("\nClassification:")
    for p in stationary_points:
        subs_H = H.subs({x1: p[x1], x2: p[x2]})
        fxx_val = f_x1x1.subs({x1: p[x1], x2: p[x2]})
        det_val = subs_H.det()
        classification = classify_stationary_point(subs_H, fxx_val)

        print(f"At point {p}:")
        print(f" • Hessian = {subs_H}")
        print(f" • det(H) = {det_val}")
        print(f" → {classification}")

    print("\n")


def run_func1():
    x1, x2 = sp.symbols('x1 x2', real=True)
    f1 = 3*x1**2 + 2*x1*x2 + 2*x2**2 + 7
    analyze_function(f1, x1, x2)

def run_func2():
    x1, x2 = sp.symbols('x1 x2', real=True)
    f2 = x1**2 + 4*x1*x2 + x2**2 + 3
    analyze_function(f2, x1, x2)

def main():
    print("\n===== ANALYSIS OF FUNCTION 1 =====")
    run_func1()

    print("\n===== ANALYSIS OF FUNCTION 2 =====")
    run_func2()


if __name__ == "__main__":
    main()