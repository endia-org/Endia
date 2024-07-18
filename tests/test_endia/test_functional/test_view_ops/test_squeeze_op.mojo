import endia as nd
from python import Python


def run_test_squeeze(msg: String = "squeeze"):
    torch = Python.import_module("torch")
    arr = nd.randn(List(2, 1, 4))
    arr_torch = nd.utils.to_torch(arr)

    res = nd.squeeze(arr)
    res_torch = torch.squeeze(arr_torch)

    if not nd.utils.is_close(res, res_torch):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)


def run_test_squeeze_grad(msg: String = "squeeze_grad"):
    torch = Python.import_module("torch")
    arr = nd.randn(List(2, 1, 4), requires_grad=True)
    arr_torch = nd.utils.to_torch(arr)

    res = nd.sum(nd.sin(nd.squeeze(arr)))
    res_torch = torch.sum(torch.sin(torch.squeeze(arr_torch)))

    res.backward()
    res_torch.backward()

    grad = arr.grad()
    grad_torch = arr_torch.grad

    if not nd.utils.is_close(grad, grad_torch):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)
