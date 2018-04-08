import math


def fi(x):
    return x + math.pow(math.e, -0.5 * x) - 0.2 * math.pow(x, 2) + 1


def f(x):
    return math.pow(math.e, -0.5 * x) - 0.2 * math.pow(x, 2) + 1


def f1(x):
    return -0.5 * math.pow(math.e, -0.5 * x) - 0.4 * x


def newton(x0, eps, iteration):
    _f = f(x0)
    _f1 = f1(x0)
    x = x0 - _f / _f1
    diff = math.fabs(x - x0)
    print("{0}\t{1:0.6f}\t{2:0.6f}\t{3:0.6f}\t{4:0.6f}".format(iteration, x0, _f, _f1, diff))
    if diff > eps:
        return newton(x, eps, iteration + 1)
    else:
        return x0


def staffinson(x0, eps, iteration):
    _f = f(x0)
    x = x0 - math.pow(_f, 2) / (f(x0 + _f) - _f)
    diff = math.fabs(x - x0)
    print("{0}\t{1:0.6f}\t{2:0.6f}\t{3:0.6f}".format(iteration, x0, _f, diff))
    if diff > eps:
        return staffinson(x, eps, iteration + 1)
    else:
        return x0


def simp_iteration(x0, eps, iteration):
    x = fi(x0)
    diff = math.fabs(x - x0)
    print("{0}\t{1:0.4f}\t{2:0.4f}\t{3:0.4f}".format(iteration, x0, x, diff))
    if diff > eps:
        return simp_iteration(x, eps, iteration + 1)
    else:
        return x0


def main():
    print("Метод простых итераций")
    print("n\tx\t\tfi(x)\tdiff")
    print("x = {0:0.3f}".format(simp_iteration(3, 0.0004, 0)))
    print("\nМетод Ньютона")
    print("n\tx\t\t\tf(x)\t\tf1(x)\t\tdiff")
    print("x = {0:0.6f}".format(newton(30, 0.000001, 0)))
    print("\nМетод Стеффенсена")
    print("n\tx\t\t\tf(x)\t\tdiff")
    print("x = {0:0.6f}".format(staffinson(10, 0.000001, 0)))


main()
