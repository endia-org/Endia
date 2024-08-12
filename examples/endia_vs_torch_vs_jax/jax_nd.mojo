from endia import grad, jacobian
from endia import sum, arange, ndarray


def foo(x: ndarray) -> ndarray:
    return sum(x**2)


def example_jax_like():
    # create Callables
    foo_jac = grad(foo)
    foo_hes = jacobian(foo_jac)

    x = arange(1.0, 4.0)

    print(str(foo(x)))
    print(str(foo_jac(x)[ndarray]))
    print(str(foo_hes(x)[ndarray]))
