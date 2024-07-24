import endia as nd
from python import Python


def run_test_avg_pool2d(msg: String = "avg_pool2d"):
    torch = Python.import_module("torch")

    kernel_size = (3, 3)
    stride = (2, 2)
    padding = (1, 1)
    dilation = (1, 1)

    input_tensor = nd.randu(List(2, 2, 10, 10))

    output = nd.avg_pool2d(
        input_tensor,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    input_tensor_torch = nd.utils.to_torch(input_tensor)
    output_torch = torch.nn.functional.avg_pool2d(
        input_tensor_torch,
        kernel_size=PythonObject(kernel_size),
        stride=PythonObject(stride),
        padding=PythonObject(padding),
        count_include_pad=False,
    )

    if not nd.utils.is_close(output, output_torch):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)
