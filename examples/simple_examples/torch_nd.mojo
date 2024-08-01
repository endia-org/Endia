from endia import Array, sum, arange
from endia.autograd import grad
import endia.autograd.functional as F


def foo(x: Array) -> Array:
    return sum(x**2)


def main():
    x = arange(1.0, 4.0, requires_grad=True)

    y = foo(x)
    dy_dx = grad(outs=y, inputs=x)[0]
    d2y_dx2 = F.hessian(foo, x)

    print(y)
    print(dy_dx)
    print(d2y_dx2)
