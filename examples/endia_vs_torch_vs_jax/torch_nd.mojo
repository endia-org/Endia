from endia import Tensor, sum, arange
from endia.autograd import grad
import endia.autograd.functional as F


def foo(x: Tensor) -> Tensor:
    return sum(x**2)


def example_torch_like():
    x = arange(1.0, 4.0, requires_grad=True)

    y = foo(x)
    dy_dx = grad(outs=y, inputs=x)[0]
    d2y_dx2 = F.hessian(foo, x)

    print(str(y))
    print(str(dy_dx))
    print(str(d2y_dx2))
