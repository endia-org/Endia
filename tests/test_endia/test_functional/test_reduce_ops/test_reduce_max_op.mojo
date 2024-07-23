import endia as nd
from python import Python


def run_test_reduce_max(msg: String = "reduce_arg_max"):
    torch = Python.import_module("torch")
    arr = nd.randu(List(2, 30, 40))
    arr_torch = nd.utils.to_torch(arr)

    axis = 1  # in PyTorch we can only call max and argmax along a single dimension at a time!

    res = nd.reduce_max(arr, axis)
    res_torch = torch.max(arr_torch, dim=axis)[
        0
    ]  # PyTorch always returns both the max and argmax as a Tuple

    if not nd.utils.is_close(res, res_torch):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)
