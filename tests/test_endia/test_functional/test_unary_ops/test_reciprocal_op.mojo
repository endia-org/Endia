import endia as nd
from python import Python


def run_test_reciprocal(msg: String = "reciprocal"):
    torch = Python.import_module("torch")
    arr = nd.arange(List(2, 30, 40))
    arr_torch = nd.utils.to_torch(arr)

    res = nd.reciprocal(arr)
    res_torch = torch.reciprocal(arr_torch)

    if not nd.utils.is_close(res, res_torch):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)


def run_test_reciprocal_grad(msg: String = "reciprocal_grad"):
    torch = Python.import_module("torch")
    arr = nd.arange(List(2, 30, 40), requires_grad=True)
    arr_torch = nd.utils.to_torch(arr)

    res = nd.sum(nd.reciprocal(arr))
    res_torch = torch.sum(torch.reciprocal(arr_torch))

    res.backward()
    res_torch.backward()

    grad = arr.grad()
    grad_torch = arr_torch.grad

    if not nd.utils.is_close(grad, grad_torch):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)


def run_test_reciprocal_complex(msg: String = "reciprocal_complex"):
    torch = Python.import_module("torch")
    arg0 = nd.randn_complex(List(2, 3, 4))
    arg0_torch = nd.utils.to_torch(arg0)

    res = nd.reciprocal(arg0)
    res_torch = torch.reciprocal(arg0_torch)

    if not nd.utils.is_close(res, res_torch):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)
