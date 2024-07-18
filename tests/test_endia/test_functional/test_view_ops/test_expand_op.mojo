import endia as nd
from python import Python


def run_test_expand(msg: String = "expand"):
    torch = Python.import_module("torch")
    arr = nd.randn(List(2, 1, 4))
    arr_torch = nd.utils.to_torch(arr)

    shape = List(2, 2, 3, 4)
    shape_torch = [2, 2, 3, 4]

    res = nd.expand(arr, shape)
    res_torch = torch.broadcast_to(arr_torch, shape_torch)

    if not nd.utils.is_close(res, res_torch):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)


def run_test_expand_grad(msg: String = "expand_grad"):
    torch = Python.import_module("torch")
    arr = nd.arange(List(2, 3, 1), requires_grad=True)
    arr_torch = nd.utils.to_torch(arr)

    res = nd.sum(nd.sin(nd.expand(arr, List(2, 3, 4))))
    res_torch = torch.sum(torch.sin(arr_torch.broadcast_to((2, 3, 4))))

    res.backward()
    res_torch.backward()

    grad = arr.grad()
    grad_torch = arr_torch.grad

    if not nd.utils.is_close(grad, grad_torch):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)
