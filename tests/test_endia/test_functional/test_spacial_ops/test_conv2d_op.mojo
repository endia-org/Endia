import endia as nd
from python import Python


def run_test_conv2d(msg: String = "conv2d"):
    torch = Python.import_module("torch")

    batch_size = 2
    in_channels = 2
    out_channels = 3
    kernel_width = 3
    kernel_height = 3
    elements = 6
    stride_width = 2
    stride_height = 2
    padding_width = 1
    padding_height = 1
    dilation_width = 1
    dilation_height = 1
    groups = 1

    a = nd.randu(shape=List(batch_size, in_channels, elements, elements))
    kernel = nd.randu(
        shape=List(out_channels, in_channels, kernel_width, kernel_height)
    )
    bias = nd.randu(shape=List(out_channels))

    a_torch = nd.utils.to_torch(a)
    kernel_torch = nd.utils.to_torch(kernel)
    bias_torch = nd.utils.to_torch(bias)

    res = nd.conv2d(
        a,
        kernel,
        bias,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=(kernel_width, kernel_height),
        stride=(stride_width, stride_height),
        padding=(padding_width, padding_height),
        dilation=(dilation_width, dilation_height),
        groups=groups,
    )
    res_torch = torch.nn.functional.conv2d(
        a_torch,
        kernel_torch,
        bias_torch,
        stride=(stride_height, stride_width),
        padding=(padding_height, padding_width),
        dilation=(dilation_height, dilation_width),
        groups=groups,
    )

    if not nd.utils.is_close(res, res_torch):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)
