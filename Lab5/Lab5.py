import sympy as sp


def lagrange_polynomial(array_x, array_y):
    x = sp.symbols('x')
    p = []
    for i in range(len(array_x)):
        p.append(1)
        ost = 1
        for j in range(len(array_x)):
            if j != i:
                p[i] *= (x - array_x[j])
                ost *= (array_x[i] - array_x[j])
        p[i] /= ost
    L = 0
    for i in range(len(array_x)):
        L += p[i] * array_y[i]
    return L


def newton_divided_differences(array_x, array_y):
    x = sp.symbols('x')
    finite_differences = [array_y]
    for i in range(len(array_x) - 1):
        finite_differences.append([])
        for j in range(len(array_x) - (i + 1)):
            diff = finite_differences[i][j + 1] - finite_differences[i][j]
            finite_differences[i + 1].append(diff)
    divided_differences = [array_y]
    h = 0
    for i in range(len(array_x) - 1):
        h += 1
        divided_differences.append([])
        for j in range(len(array_x) - (i + 1)):
            diff = (divided_differences[i][j + 1] - divided_differences[i][j]) / (array_x[j + h] - array_x[j])
            divided_differences[i + 1].append(diff)

    Newton = 0
    for i in range(len(array_x)):
        term = divided_differences[i][0]
        composition = 1
        for j in range(i):
            composition *= x - array_x[j]
        Newton += term * composition
    return Newton


def linear_spline(array_x, array_y):
    a, b, x = sp.symbols('a, b, x')
    systems = []
    for i in range(len(array_x) - 1):
        systems.append(sp.Matrix((
            (array_x[i], 1, array_y[i]),
            (array_x[i + 1], 1, array_y[i + 1])
        )))
    solves = []
    for i in range(len(systems)):
        solves.append(sp.solve_linear_system(systems[i], a, b))
    fi = []
    for i in range(len(solves)):
        eq = solves[i][a] * x + solves[i][b]
        fi.append(eq)
    return fi


def quadratic_spline(array_x, array_y):
    a, b, c, x = sp.symbols('a, b, c, x')
    systems = []
    r = [0, 2]
    for i in r:
        systems.append(sp.Matrix((
            (array_x[i] ** 2, array_x[i], 1, array_y[i]),
            (array_x[i + 1] ** 2, array_x[i + 1], 1, array_y[i + 1]),
            (array_x[i + 2] ** 2, array_x[i + 2], 1, array_y[i + 2])
        )))
    solves = []
    for i in range(len(systems)):
        solves.append(sp.solve_linear_system(systems[i], a, b, c))
    fi = []
    for i in range(len(solves)):
        eq = solves[i][a] * x ** 2 + solves[i][b] * x + solves[i][c]
        fi.append(eq)
    return fi


def main():
    # array_x = [0.351, 0.867, 3.315, 5.013, 6.432]
    # array_y = [-0.572, -2.015, -3.342, -5.752, -6.911]
    array_x = [0.135, 0.876, 1.336, 2.301, 2.642]
    array_y = [-2.132, -2.113, -1.613, -0.842, 1.204]
    x = sp.symbols('x')
    L = lagrange_polynomial(array_x, array_y)
    L_lambd = sp.lambdify(x, L)
    print("L = {0}".format(L))
    print("L(x1 + x2) = {0}".format(L_lambd(array_x[1] + array_x[2])))

    Newton = newton_divided_differences(array_x, array_y)
    Newton_lambda = sp.lambdify(x, Newton)
    print("N = {0}".format(Newton))
    print("N(x1 + x2) = {0}".format(Newton_lambda(array_x[1] + array_x[2])))

    a, b, c = sp.symbols('a, b, c')

    linear = linear_spline(array_x, array_y)
    quadratic = quadratic_spline(array_x, array_y)

    quadratic_lambda = []
    for i in quadratic:
        quadratic_lambda.append(sp.lambdify(x, i))

    diff1 = (array_x[2] - array_x[0]) / 4
    diff2 = (array_x[4] - array_x[2]) / 4

    points_y = []
    h = array_x[0]
    points_x = []
    for i in range(4):
        points_x.append(h)
        points_y.append(quadratic_lambda[0](h))
        h += diff1
    h = array_x[2]
    for i in range(4):
        points_x.append(h)
        points_y.append(quadratic_lambda[1](h))
        h += diff2


main()
