import numpy as np
import sympy as sp


def newton(F, J, x0, eps):
    log = [x0]
    diffLog = [0]
    x0 = np.resize(x0, 2)
    # J_num = J(x0[0], x0[1])
    # J_reverse = np.linalg.inv(J_num)
    # F_num = F(x0[0], x0[1])
    # x1 = x0 - np.dot(J_reverse, F_num)
    # log.append(np.resize(x1, (2, 1)))
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
    # x0 = np.resize(x0, 2).tolist()
    J_num = J(x0[0], x0[1])
    J_reverse = np.linalg.inv(J_num)
    F_num = F(x0[0], x0[1])
    x1 = x0 - np.dot(J_reverse, F_num)
    return x1


# def calcSystem(system, x):
#     symbols = ['x1', 'x2']
#     tmp = [[0] for i in range(len(x))]
#     for i in range(len(x)):
#         tmp[i][0] = list(system[i])
#         for j in range(len(x)):
#             tmp[i][0] = tmp[i][0].subs(symbols[j], x[j])
#     return tmp
#
#
# def calcJNumeric(J, x):
#     symbols = ['x1', 'x2']
#     tmp = [[0] * len(x) for i in range(len(x))]
#     for i in range(len(x)):
#         tmp[i] = list(J[i])
#         for j in range(len(x)):
#             for k in range(len(x)):
#                 tmp[i][j] = tmp[i][j].subs(symbols[k], x[k])
#     return np.array(tmp)


def norm(x):
    return max(abs(i) for i in x)


def main():
    unknownsCount = 2
    x = sp.symbols('x1, x2')
    # system = [sp.sin(x[0] + x[1]) - 1.5 * x[0] - 0.1, 3 * x[0] ** 2 + x[1] ** 2 - 1]
    system = [
        sp.sin(x[0] + 1.5) - x[1] + 2.9,
        sp.cos(x[1] - 2) + x[0]
    ]
    J = [[0] * unknownsCount for i in range(unknownsCount)]
    for i in range(unknownsCount):
        for j in range(unknownsCount):
            J[i][j] = sp.diff(system[i], x[j], 1)
    J = sp.lambdify(x, J)
    F = sp.lambdify(x, system)
    # print(J)
    x0 = np.array([
        [0],
        [4]
    ])
    log, diffLog = newton(F, J, x0, 1e-6)
    print()


main()
