from autograd import grad
import autograd.numpy as np
from itertools import product
from math import factorial


def approx(D, x_0, x):
    y = D[0][0, 0]
    for D_n in D[1:]:
        y += float(approx_n(D_n, x - x_0))
    return y


def approx_n(D_n, x):
    y = 0
    n_axes = len(D_n.shape)
    x_int = np.copy(D_n)
    for i in range(n_axes):
        x_int = np.dot(x_int, x)
        x_int = x_int.swapaxes(i, -1)
    y += x_int
    return y


def get_params(f, n, m, x):
    D = [np.asarray([[f(*x)]])]
    for i in np.arange(n) + 1:
        D.append(D_n(f, i, m, x))
    return D


def D_n(f, n, m, x):
    size = n * [m]
    D_n = np.zeros(size)
    indexes = product(np.arange(m), repeat=n)
    for index in indexes:
        d_f = f
        for i in index:
            d_f = grad(d_f, i)
        D_n[index] = d_f(*x)
    return D_n / factorial(n)
