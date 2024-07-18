import endia as nd
from python import Python


def run_test_greater_equal(msg: String = "greater_equal"):
    arr = nd.randn(List(2, 30, 40))
    arr_torch = nd.utils.to_torch(arr)

    res = arr >= 0
    res_torch = (arr_torch >= 0).float()

    if not nd.utils.is_close(res, res_torch):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)
