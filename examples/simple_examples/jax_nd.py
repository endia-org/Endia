from jax import grad, jacobian
from jax.numpy import sum, arange, ndarray


def foo(x: ndarray) -> ndarray:
    return sum(x**2)


def main():
    # create Callables
    foo_jac = grad(foo)
    foo_hes = jacobian(foo_jac)

    x = arange(1.0, 4.0)

    print(foo(x))
    print(foo_jac(x))
    print(foo_hes(x))


main()
