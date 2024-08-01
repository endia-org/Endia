from endia import Array

# from endia.autograd.functional import grad
# from endia.autograd.functional import jacobian
import endia


# Define the function
def foo(x: Array) -> Array:
    return endia.sum(x**2)


def main():
    # initialize input
    x = 1 + endia.arange(shape=List(3), requires_grad=True)

    # Compute result, first aendia secoendia order derivatives
    y = foo(x)
    dy_dx = endia.autograd.functional.grad(outs=y, inputs=x)[0]
    d2y_dx2 = endia.autograd.functional.hessian(foo, x)

    # Print results
    print(y)  # 14.0
    print(dy_dx)  # [2.0, 4.0, 6.0]
    print(d2y_dx2)  # [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
