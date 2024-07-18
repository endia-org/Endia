import endia as nd


# Define the function
def foo(x: nd.Array) -> nd.Array:
    return nd.sum(x**2)


def example1():
    print("Example 1 ###########################################")

    print("\nImperative grad computation:")
    # Initialize variable - requires_grad=True needed!
    x = nd.array("[1.0, 2.0, 3.0]", requires_grad=True)

    # Compute result, first and second order derivatives
    y = foo(x)
    y.backward(retain_graph=True)
    dy_dx = x.grad()
    d2y_dx2 = nd.grad(outs=dy_dx, inputs=x)[nd.Array]

    # Print results
    print(y)  # out: [14.0]
    print(dy_dx)  # out: [2.0, 4.0, 6.0]
    print(d2y_dx2)  # out: [2.0, 2.0, 2.0]

    print("\nFunctional grad computation:")

    # Create callables for the jacobian and hessian
    foo_jac = nd.grad(foo)
    foo_hes = nd.grad(foo_jac)

    # Initialize variable - no requires_grad=True needed
    x = nd.array("[1.0, 2.0, 3.0]")

    # Compute result and derivatives (with type hints)
    y = foo(x)
    dy_dx = foo_jac(x)[nd.Array]
    d2y_dx2 = foo_hes(x)[nd.Array]

    # Print results
    print(y)  # out: [14.0]
    print(dy_dx)  # out: [2.0, 4.0, 6.0]
    print(d2y_dx2)  # out: [2.0, 2.0, 2.0]
