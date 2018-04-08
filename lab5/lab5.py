import numpy as np
import sympy as sp


def newton(F, J, x0, eps):
    diffLog = [0]
    x0 = np.resize(x0, 2)
    log = [x0]
    x1 = doIteration(F, J, x0)
    diff = norm(x1 - x0)
    diffLog.append(diff)
    log.append(x1)
    while diff > eps:
        x0 = x1
        x1 = doIteration(F, J, x1)
        diff = norm(x1 - x0)
        log.append(x1)
        diffLog.append(diff)
    return log, diffLog


def doIteration(F, J, x0):
    J_num = J(x0[0], x0[1])
    J_reverse = np.linalg.inv(J_num)
    F_num = F(x0[0], x0[1])
    x1 = x0 - np.dot(J_reverse, F_num)
    return x1


def norm(x):
    return max(abs(i) for i in x)


def printNewton(log, diffLog):
    for k in range(len(log)):
        print("{0}\t{1:.6f}\t{2:.6f}".format(k, log[k][0], diffLog[k]))
        print(" \t{0:.6f}".format(log[k][1]))


def gradient_descent(G, x, eps):
    F = G[0] ** 2 + G[1] ** 2
    F = sp.expand(F)
    JF = [sp.diff(F, x[0]), sp.diff(F, x[1])]
    JF_calc = sp.lambdify(x, JF)

    x0 = [0, 1]
    alpha = 0.01
    diff_log = [0]
    log = [norm(x0)]

    x1 = x0 - np.dot(alpha, JF_calc(x0[0], x0[1]))

    norm_diff = norm(x1 - x0)

    diff_log.append(norm_diff)
    log.append(x1)

    while norm_diff > eps:
        x0 = x1
        x1 = x1 - np.dot(alpha, JF_calc(x1[0], x1[1]))
        norm_diff = norm(x1 - x0)
        diff_log.append(norm_diff)
        log.append(x1)

    print()


def main():
    unknownsCount = 2
    x = sp.symbols('x1, x2')
    system = [
        sp.sin(x[0] + x[1]) - 1.5 * x[0] - 0.1,
        3 * x[0] ** 2 + x[1] ** 2 - 1
    ]
    # system = [
    #     sp.sin(x[0] + 1.5) - x[1] + 2.9,
    #     sp.cos(x[1] - 2) + x[0]
    # ]
    J = [[0] * unknownsCount for i in range(unknownsCount)]
    for i in range(unknownsCount):
        for j in range(unknownsCount):
            J[i][j] = sp.diff(system[i], x[j], 1)
    J = sp.lambdify(x, J)
    F = sp.lambdify(x, system)
    x0 = np.array([
        [0],
        [4]
    ])
    log, diffLog = newton(F, J, x0, 1e-6)
    #printNewton(log, diffLog)
    gradient_descent(system, x, 1e-6)


main()
