from endia.utils import op_array, setup_shape_and_data
from endia import Array


def custom_mul_callable(inout curr: Array, args: List[Array]) -> None:
    """This operation defines what should happen when the operation is called.
    Since shapes are computed jsut as laziliy as the data, we need to make sure that the
    shape of the resulting array is set before we can set the data.
    """
    setup_shape_and_data(curr)
    for i in range(curr.size()):
        curr.store(i, args[0].load(i) * args[1].load(i))


def custom_mul_vjp(
    primals: List[Array], grad: Array, out: Array
) -> List[Array]:
    """The vector-Jacobian product of the custom multiplication operation.
    We use the vector-Jacobian to define what happens when the gradient is backpropagated.
    Reminder: d/dx (a*b) = a * d/dx(b) + b * d/dx(a).
    """
    return List(custom_mul(grad, primals[1]), custom_mul(grad, primals[0]))


def custom_mul(arg0: Array, arg1: Array) -> Array:
    """The forward pass of the custom multiplication operation.
    Here we register the the resulting array, its shape, its operation name and its
    parents i.e. arg0 and arg1. For simplicity this function does not support
    automatic broadcasting, therefore we add the constraint that the two input arrays
    must have the same shape.
    """
    if arg0.array_shape() != arg1.array_shape():
        raise "Warning in custom_mul: The two input arrays must have the same shape."

    # register the resulting array, its shape, its operation name, and its parents
    return op_array(
        array_shape=arg0.array_shape(),
        args=List(arg0, arg1),
        name="mul",
        callable=custom_mul_callable,
        vjp=custom_mul_vjp,
    )


def main():
    """Simple test of the custom operation.
    We create two arrays, multiply them and sum the result.
    Then we compute the derivative of the output with respect to the two input arrays.
    """
    a = Array("[1, 2, 3]", requires_grad=True)
    b = Array("[4, 5, 6]", requires_grad=True)
    c = endia.sum(custom_mul(a, b))
    print(c)  # expected output: [32]

    c.backward()
    print(a.grad())  # expected output: [4, 5, 6]
    print(b.grad())  # expected output: [1, 2, 3]
