import numpy as np
import math

def observError(A, b, absolute_error_b):
    nA = norm(A)
    nInvA = norm(np.linalg.inv(A))
    nB = norm(b)
    relative_error_b = absolute_error_b / nB
    absolute_error_solution = nInvA * absolute_error_b
    relative_error_solution = nA * nInvA * relative_error_b
    return absolute_error_solution, relative_error_solution

def calcK(c, B, eps):
    n_c = norm(np.array(c))
    n_B = norm(np.array(B))
    k = math.log((eps * (1 - n_B)) / n_c) / math.log(n_B)
    return k

def simpleIterationMethod(A, b, eps):
    B, c = makeRequiredForm(A, b)
    k = math.ceil(calcK(c, B, eps).real)
    c = np.array(c)
    B = np.array(B)
    if (norm(B) >= 1):
        return 0
    x = [np.array([0, 0, 0, 0])]
    x[0] = np.resize(x[0], (len(x[0]), 1))
    for i in range(1, k + 1):
        x.append(np.array(np.dot(B, x[i - 1]) + c))
    errors = itearationErrors(b, c, k, x[len(x) - 2], x[len(x) - 1])
    return x[len(x) - 1], errors

def makeRequiredForm(A, b):
    c = []
    B = []
    for i in range(len(A)):
        B.append([])
        c.append(b[i] / A[i][i])
        for j in range(len(A[i])):
            if (j == i):
                B[i].append(0.0)
            else:
                B[i].append(-A[i][j] / A[i][i])
    return B, c

def norm(A):
    arrOfSums = []
    if (len(A.shape) == 1):
        return abs(max(A))
    for i in A:
        arrOfSums.append(sum(abs(i)))
    return max(arrOfSums)

def itearationErrors(b, c, k, x1, x2):
    nDiff = norm(x2 - x1)
    result = norm(b) * nDiff / (1 - norm(b))
    return result

def seidel(A, b, eps):
    coverage = False
    n = len(A)
    x = [.0 for i in range(n)]
    while not coverage:
        x_new = np.copy(x)
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i][0] - s1 - s2) / A[i][i]
        coverage = norm(x_new - x) <= eps
        x = x_new
    return x

def main():
    A = np.array([
        [5.482, 0.358, 0.237, 0.409],
        [0.580, 4.953, 0.467, 0.028],
        [0.319, 0.372, 8.935, 0.520],
        [0.043, 0.459, 0.319, 4.778]
    ])
    b = np.array([0.416, 0.464, 0.979, 0.126])
    b = np.resize(b, (len(b), 1))
    err1, err2 = observError(A, b, 0.001)
    print("Абсолютная погрешность решения: {0:.5f}\nОтносительная погрешность решения: {1:.5f}".format(err1, err2))
    X_MPI, errors = simpleIterationMethod(A, b, 0.01)
    for i in range(len(X_MPI)):
        print("x{0} = {1:.3f}".format(i + 1, X_MPI[i][0]))
    print("Погрешность приближенного значения: {0}".format(round(errors, 5)))
    X_Seidel = seidel(A, b, 0.001)
    for i in range(len(X_Seidel)):
        print("x{0} = {1:.3f}".format(i + 1, X_Seidel[i]))

main()