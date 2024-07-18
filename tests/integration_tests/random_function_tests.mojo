import endia as nd
from python import Python


def foo(args: List[nd.Array]) -> nd.Array:
    a = args[0]
    b = args[1]
    c = args[2]
    return nd.sum(
        nd.mul(
            nd.cos(nd.sin(nd.cos(nd.cos(nd.add(nd.matmul(a, b), c))))),
            nd.matmul(a, b),
        )
    )


def foo_torch(args: List[PythonObject]) -> PythonObject:
    torch = Python.import_module("torch")
    a = args[0]
    b = args[1]
    c = args[2]
    return torch.sum(
        torch.mul(
            torch.cos(
                torch.sin(
                    torch.cos(torch.cos(torch.add(torch.matmul(a, b), c)))
                )
            ),
            torch.matmul(a, b),
        )
    )


def run_test_foo(msg: String = "foo"):
    # endia args initialization
    arg0 = nd.randn(List(2, 3, 4))
    arg1 = nd.randn(List(4, 5))
    arg2 = nd.randn(List(2, 3, 5))
    args = List(arg0, arg1, arg2)

    # torch args initialization
    arg0_torch = nd.utils.to_torch(arg0)
    arg1_torch = nd.utils.to_torch(arg1)
    arg2_torch = nd.utils.to_torch(arg2)
    args_torch = List(arg0_torch, arg1_torch, arg2_torch)

    # fucntional calls
    res = foo(args)
    res_torch = foo_torch(args_torch)

    # check if the results are close
    if not nd.utils.is_close(res, res_torch):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)


def run_test_foo_grad(msg: String = "foo_backward"):
    # endia args initialization
    a = nd.randn(List(3, 4), requires_grad=True)
    b = nd.randn(List(2, 4, 5), requires_grad=True)
    c = nd.randn(List(2, 3, 5), requires_grad=True)
    args = List(a, b, c)

    # torch args initialization
    a_torch = nd.utils.to_torch(a)
    b_torch = nd.utils.to_torch(b)
    c_torch = nd.utils.to_torch(c)
    args_torch = List(a_torch, b_torch, c_torch)

    # function calls
    res = foo(args)
    res_torch = foo_torch(args_torch)

    # backward pass
    res.backward()
    res_torch.backward()

    # get the gradients for nd and torch
    grad_a = a.grad()
    grad_b = b.grad()
    grad_c = c.grad()
    grad_a_torch = a_torch.grad
    grad_b_torch = b_torch.grad
    grad_c_torch = c_torch.grad

    # check if the results are close
    test_success = True
    if not nd.utils.is_close(res, res_torch):
        print("\033[31mTest failed\033[0m", msg)
        test_success = False
    if not nd.utils.is_close(grad_a, grad_a_torch):
        print("\033[31mTest failed\033[0m", msg, "grad_a")
        test_success = False
    if not nd.utils.is_close(grad_b, grad_b_torch):
        print("\033[31mTest failed\033[0m", msg, "grad_b")
        test_success = False
    if not nd.utils.is_close(grad_c, grad_c_torch):
        print("\033[31mTest failed\033[0m", msg, "grad_c")
        test_success = False
    if test_success:
        print("\033[32mTest passed\033[0m", msg)
