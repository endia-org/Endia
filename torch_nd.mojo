from endia import Array
import endia.autograd.functional as F


# Define the function
def foo(x: Array) -> Array:
    return endia.sum(x**2)


def main():
    # initialize input
    x = endia.arange(1.0, 4.0, requires_grad=True)  # [1.0, 2.0, 3.0]

    # Compute result, first and second order derivatives
    y = foo(x)
    dy_dx = F.grad(outs=y, inputs=x)[0]
    d2y_dx2 = F.hessian(foo, x)

    # Print results
    print(y)  # 14.0
    print(dy_dx)  # [2.0, 4.0, 6.0]
    print(d2y_dx2)  # [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
