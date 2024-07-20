import endia as nd
from python import Python


def run_test_as_strided(msg: String = "as_strided"):
    torch = Python.import_module("torch")
    arr = nd.randn(List(2, 30, 40))  # 12,4,1
    arr_torch = nd.utils.to_torch(arr)

    res = nd.as_strided(
        arr,
        shape=List(1, 2, 40, 2, 2),
        stride=List(16, 8, 2, 2, 1),
        storage_offset=0,
    )
    res_torch = torch.as_strided(
        arr_torch,
        size=(1, 2, 40, 2, 2),
        stride=(16, 8, 2, 2, 1),
        storage_offset=0,
    )

    if not nd.utils.is_close(res, res_torch):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)


def run_test_as_strided_grad(msg: String = "as_strided_grad"):
    torch = Python.import_module("torch")
    arr = nd.arange(List(30, 30, 3), requires_grad=True)
    arr_torch = nd.utils.to_torch(arr)

    res = nd.sum(
        nd.sin(
            nd.as_strided(
                arr,
                shape=List(30, 30, 2),
                stride=List(9, 30, 2),
                storage_offset=0,
            )
        )
    )
    res_torch = torch.sum(
        torch.sin(
            torch.as_strided(
                arr_torch,
                size=(30, 30, 2),
                stride=(9, 30, 2),
                storage_offset=0,
            )
        )
    )

    res.backward()
    res_torch.backward()

    grad = arr.grad()
    grad_torch = arr_torch.grad

    if not nd.utils.is_close(grad, grad_torch):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)
