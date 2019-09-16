from importlib import import_module

from sympy import *

import torch
import pau_cuda
import numpy as np
import math


class ReAbs(Abs):
    def _eval_derivative(self, x):
        return Derivative(self.args[0], x, evaluate=True) * (
                self.args[0] / ReAbs(self.args[0]))  # sign(conjugate(self.args[0]))

    def _sympystr(self, settings=None):
        return "|%s|" % (self.args[0])


def getA(deg_a, deg_b):
    # P(X)/Q(X) = a_0 + a_1*X + a_2*X^2 + ... + a_n*X^n / 1 + |b_0|*|X| + |b_1|*|X^2| + ... + |b_{n-1}}|*|X^n|
    deg_a += 1

    a = symbols('a0:' + str(deg_a))
    b = symbols('b0:' + str(deg_b))
    x = symbols('x')

    Px = 0
    for i in range(deg_a):
        Px = Px + a[i] * x ** i

    absx = ReAbs(x)

    Qx = 1
    for i in range(deg_b):
        Qx = Qx + ReAbs(b[i]) * absx ** (i + 1)

    F = Px / Qx

    print(F)

    return F, a, b, x, deg_a - 1, deg_b, "A"

def getB(deg_a, deg_b):
    # // P(X)/Q(X) = a_0 + a_1*X + a_2*X^2 + ... + a_n*X^n / 1 + |b_0*X + b_1*X^2 + ... + b_{n-1}*X^n|
    deg_a += 1

    a = symbols('a0:' + str(deg_a))
    b = symbols('b0:' + str(deg_b))
    x = symbols('x')

    print(a, b)

    Px = 0
    for i in range(deg_a):
        Px = Px + a[i] * x ** i


    Qx = 0
    for i in range(deg_b):
        Qx = Qx + b[i] * x ** (i + 1)

    Qx = 1 + ReAbs(Qx)

    F = Px / Qx

    print(F)

    return F, a, b, x, deg_a - 1, deg_b, "B"



def getC(deg_a, deg_b):
    # P(X)/Q(X) = a_0 + a_1*X + a_2*X^2 + ... + a_n*X^n / eps + |b_0 + b_1*X + b_2*X^2 + ... + b_n*X^n|
    # eps = 0.0000001
    deg_a += 1

    a = symbols('a0:' + str(deg_a))
    b = symbols('b0:' + str(deg_b))
    x = symbols('x')

    print(a, b)

    Px = 0
    for i in range(deg_a):
        Px = Px + a[i] * x ** i


    Qx = 0
    for i in range(deg_b):
        Qx = Qx + b[i] * x ** (i)

    Qx = 0.0000001 + ReAbs(Qx)

    F = Px / Qx

    print(F)

    return F, a, b, x, deg_a - 1, deg_b, "C"

def get_diffs(F, a, b, x, deg_a, deg_b, version):
    dFdx = diff(F, x)
    dFda = list(map(lambda a_i: diff(F, a_i), a))
    dFdb = list(map(lambda b_i: diff(F, b_i), b))

    return F, a, b, x, dFdx, dFda, dFdb, deg_a, deg_b, version


def test(F, a, b, x, dFdx, dFda, dFdb, deg_a, deg_b, version):
    substitution_vals = []

    coef_a = []
    for i in range(len(a)):
        coef_a.append( math.pow(-1,i) * (i + 1) * 0.01)
        substitution_vals.append((a[i], coef_a[i]))

    coef_b = []
    for i in range(len(b)):
        coef_b.append(math.pow(-1,i) * (i + 1) * 0.04)
        substitution_vals.append((b[i], coef_b[i]))

    xval = [-1.0, -0.5, -0.1, 0.1, 0.5, 1.0]

    sympy_x = np.array(list(map(lambda xv: float(F.subs(substitution_vals + [(x, xv)])), xval)))

    print(deg_a, deg_b)

    pau_forward_cuda = getattr(pau_cuda, 'forward{version}_{deg_a}_{deg_b}'.format(version=version, deg_a=deg_a, deg_b=deg_b))
    pau_backward_cuda = getattr(pau_cuda, 'backward{version}_{deg_a}_{deg_b}'.format(version=version, deg_a=deg_a, deg_b=deg_b))

    go_torch = torch.from_numpy(np.ones_like(xval, dtype=np.float32)).cuda()
    xval_torch = torch.from_numpy(np.array(xval, dtype=np.float32)).cuda()
    coef_a_torch = torch.from_numpy(np.array(coef_a, dtype=np.float32)).cuda()
    coef_b_torch = torch.from_numpy(np.array(coef_b, dtype=np.float32)).cuda()


    x_torch = pau_forward_cuda(xval_torch, coef_a_torch, coef_b_torch).cpu().detach().numpy()

    print("sympy x", sympy_x)
    print("cuda  x", x_torch)

    assert np.all(np.isclose(x_torch - sympy_x, 0, atol=0.0000001))

    d_x_torch, d_weight_numerator_torch, d_weight_denominator_torch = pau_backward_cuda(go_torch, xval_torch, coef_a_torch, coef_b_torch)
    d_x = d_x_torch.cpu().detach().numpy()
    d_a = d_weight_numerator_torch.cpu().detach().numpy()
    d_b = d_weight_denominator_torch.cpu().detach().numpy()

    sympy_d_x = np.array(list(map(lambda xv: float(dFdx.subs(substitution_vals + [(x, xv)])), xval)))

    print("sympy d_x", sympy_d_x)
    print("cuda  d_x", d_x)

    assert np.all(np.isclose(d_x - sympy_d_x, 0, atol=0.000001))

    sympy_d_a = []
    for da in dFda:
        res = sum(map(lambda xv: float(da.subs(substitution_vals + [(x, xv)])), xval))
        sympy_d_a.append(res)

    print("sympy d_a", sympy_d_a)
    print("cuda  d_a", d_a)

    assert np.all(np.isclose(d_a - sympy_d_a, 0, atol=0.0001))

    sympy_d_b = []
    for db in dFdb:
        res = sum(map(lambda xv: float(db.subs(substitution_vals + [(x, xv)])), xval))
        sympy_d_b.append(res)

    print("sympy d_b", sympy_d_b)
    print("cuda  d_b", d_b)

    assert np.all(np.isclose(d_b - sympy_d_b, 0, atol=0.00001))

    print(deg_a, deg_b, version, "Passed!")


for degrees in [(3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (5, 4)]:
    test(*get_diffs(*getA(*degrees)))
    test(*get_diffs(*getB(*degrees)))
    test(*get_diffs(*getC(*degrees)))
