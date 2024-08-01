from torch import Tensor, sum, arange
from torch.autograd import grad
import torch.autograd.functional as F


def foo(x: Tensor) -> Tensor:
    return sum(x**2)


def main():
    x = arange(1.0, 4.0, requires_grad=True)

    y = foo(x)
    dy_dx = grad(outputs=y, inputs=x)[0]
    dy2_dx2 = F.hessian(foo, x)

    print(y)
    print(dy_dx)
    print(dy2_dx2)


main()
