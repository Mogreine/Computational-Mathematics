import math


def fi(x):
    return x + math.pow(math.e, -0.5 * x) - 0.2 * math.pow(x, 2) + 1


def f(x):
    return 2 * math.pow(x, 2) - math.pow(x, 3) - math.pow(math.e, x)


def f1(x):
    return 4 * x - 3 * math.pow(x, 2) - math.pow(math.e, x)


def newton(x0, eps, iteration):
    _f = f(x0)
    _f1 = f1(x0)
    x = x0 - _f / _f1
    diff = math.fabs(x - x0)
    print("{0}\t{1:0.6f}\t{2:0.6f}\t{3:0.6f}\t{4:0.6f}".format(iteration, x, _f, _f1, diff))
    if diff > eps:
        return newton(x, eps, iteration + 1)
    else:
        return x0


def staffinson(x0, eps, iteration):
    _f = f(x0)
    x = x0 - math.pow(_f, 2) / (f(x0 + _f) - _f)
    diff = math.fabs(x - x0)
    print("{0}\t{1:0.6f}\t{2:0.6f}\t{3:0.6f}".format(iteration, x, _f, diff))
    if diff > eps:
        return staffinson(x, eps, iteration + 1)
    else:
        return x0


def fi_sample(x):
    return (4 * x + 2 * math.pow(x, 2) - math.pow(x, 3) - math.pow(math.e, x)) / 4


def simp_iteration(x0, eps, iteration):
    x = fi_sample(x0)
    diff = math.fabs(x - x0)
    print("{0}\t{1:0.4f}\t{2:0.4f}\t{3:0.4f}".format(iteration, x0, x, diff))
    if diff > eps:
        return simp_iteration(x, eps, iteration + 1)
    else:
        return x0


def main():
    print("n\tx\t\tfi(x)\tdiff")
    print("x = {0:0.4f}".format(simp_iteration(-1, 0.0002, 0)))
    print("\nn\tx\t\t\tf(x)\t\tf1(x)\t\tdiff")
    print("x = {0:0.6f}".format(newton(-1, 0.000001, 0)))
    print("\nn\tx\t\t\tf(x)\t\tdiff")
    print("x = {0:0.6f}".format(staffinson(-1, 0.000001, 0)))

main()
