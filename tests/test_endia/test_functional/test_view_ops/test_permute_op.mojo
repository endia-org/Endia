import endia as nd
from python import Python


def run_test_permute(msg: String = "permute"):
    torch = Python.import_module("torch")
    arr = nd.randn(List(2, 30, 40))
    arr_torch = nd.utils.to_torch(arr)

    res = nd.permute(arr, List(2, 0, 1))
    res_torch = torch.permute(arr_torch, dims=(2, 0, 1))

    if not nd.utils.is_close(res, res_torch):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)


def run_test_permute_grad(msg: String = "permute_grad"):
    torch = Python.import_module("torch")
    arr = nd.randn(List(2, 30, 40), requires_grad=True)
    arr_torch = nd.utils.to_torch(arr)

    res = nd.sum((nd.permute(arr, List(2, 0, 1))))
    res_torch = torch.sum((arr_torch.permute(2, 0, 1)))

    res.backward()
    res_torch.backward()

    grad = arr.grad()
    grad_torch = arr_torch.grad

    if not nd.utils.is_close(grad, grad_torch):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)
