import endia as nd
from python import Python


def run_test_sinh(msg: String = "sinh"):
    torch = Python.import_module("torch")
    arr = nd.randn(List(2, 30, 40))
    arr_torch = nd.utils.to_torch(arr)

    res = nd.sinh(arr)
    res_torch = torch.sinh(arr_torch)

    if not nd.utils.is_close(res, res_torch):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)


def run_test_sinh_grad(msg: String = "sinh_grad"):
    torch = Python.import_module("torch")
    arr = nd.randn(List(2, 30, 40), requires_grad=True)
    arr_torch = nd.utils.to_torch(arr)

    res = nd.sum(nd.sinh(arr))
    res_torch = torch.sum(torch.sinh(arr_torch))

    res.backward()
    res_torch.backward()

    grad = arr.grad()
    grad_torch = arr_torch.grad

    if not nd.utils.is_close(grad, grad_torch):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)


def run_test_sinh_complex(msg: String = "sinh_complex"):
    torch = Python.import_module("torch")
    arg0 = nd.randn_complex(List(2, 3, 4))
    arg0_torch = nd.utils.to_torch(arg0)

    res = nd.sinh(arg0)
    res_torch = torch.sinh(arg0_torch)

    if not nd.utils.is_close(res, res_torch):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)
