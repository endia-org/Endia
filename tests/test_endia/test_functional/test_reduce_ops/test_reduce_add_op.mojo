import endia as nd
from python import Python


def run_test_reduce_add(msg: String = "reduce_add"):
    torch = Python.import_module("torch")
    arr = nd.randn(List(2, 30, 40))
    arr_torch = nd.utils.to_torch(arr)

    axis = List(1)
    axis_torch = [1]

    res = nd.reduce_add(arr, axis)
    res_torch = torch.sum(arr_torch, axis_torch)

    if not nd.utils.is_close(res, res_torch):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)


def run_test_reduce_add_grad(msg: String = "reduce_add_grad"):
    torch = Python.import_module("torch")
    arr = nd.randn(List(2, 30, 40), requires_grad=True)
    arr_torch = nd.utils.to_torch(arr)

    axis = List(1)
    axis_torch = [1]

    res = nd.sum(nd.reduce_add(arr, axis))
    res_torch = torch.sum(torch.sum(arr_torch, axis_torch))

    res.backward()
    res_torch.backward()

    grad = arr.grad()
    grad_torch = arr_torch.grad

    if not nd.utils.is_close(grad, grad_torch):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)
