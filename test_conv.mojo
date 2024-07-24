import endia as nd


def test_conv1d():
    # we test teh 1d convolution operation
    batch_size = 2
    in_channels = 2
    out_channels = 3
    kernel_size = 3
    elements = 6
    stride = 2
    padding = 1
    dilation = 1
    groups = 1

    a = nd.arange(shape=List(batch_size, in_channels, elements))
    kernel = nd.ones(shape=List(out_channels, in_channels, kernel_size))
    bias = nd.zeros(shape=List(out_channels))

    out = nd.conv1d(
        a,
        kernel,
        bias,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
    )
    print(out)


# PyTorch version:
# import torch
# import torch.nn as nn

# # Create input tensor: batch size 2, 3 channels, 5 elements each
# batch_size = 2
# in_channels = 2
# out_channels = 3
# kernel_size = 3
# elements = 6
# stride = 2
# padding = 1
# dilation = 1
# groups = 1
# num_elements = batch_size * in_channels * elements

# input = torch.arange(0, num_elements, dtype=torch.float32).reshape(batch_size, in_channels, elements)
# print("Input shape:", input.shape)
# print("Input:\n", input)

# # Define the convolution layer
# conv1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups)

# # Set kernel to ones and bias to zero
# with torch.no_grad():
#     conv1d.weight.fill_(1.0)
#     conv1d.bias.fill_(0.0)

# print("\nKernel (weight) shape:", conv1d.weight.shape)
# print("Kernel (weight):\n", conv1d.weight.data)
# print("Bias:\n", conv1d.bias.data)

# # Perform the convolution
# output = conv1d(input)

# print("\nOutput shape:", output.shape)
# print("Output:\n", output)


def test_conv2d():
    # we test teh 1d convolution operation
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

    a = nd.arange(shape=List(batch_size, in_channels, elements, elements))
    kernel = nd.ones(
        shape=List(out_channels, in_channels, kernel_width, kernel_height)
    )
    bias = nd.zeros(shape=List(out_channels))
    out = nd.conv2d(
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
    print(out)


def test_conv3d():
    # we test the 3d convolution operation
    batch_size = 2
    in_channels = 2
    out_channels = 3
    kernel_depth = 3
    kernel_height = 3
    kernel_width = 3
    depth = 6
    height = 6
    width = 6
    stride_depth = 2
    stride_height = 2
    stride_width = 2
    padding_depth = 1
    padding_height = 1
    padding_width = 1
    dilation_depth = 1
    dilation_height = 1
    dilation_width = 1
    groups = 1

    a = nd.arange(shape=List(batch_size, in_channels, depth, height, width))
    kernel = nd.ones(
        shape=List(
            out_channels, in_channels, kernel_depth, kernel_height, kernel_width
        )
    )
    bias = nd.zeros(shape=List(out_channels))

    out = nd.conv3d(
        a,
        kernel,
        bias,
        in_channels,
        out_channels,
        (kernel_depth, kernel_height, kernel_width),
        (stride_depth, stride_height, stride_width),
        (padding_depth, padding_height, padding_width),
        (dilation_depth, dilation_height, dilation_width),
        groups,
    )
    print(out)


def test_max_pool1d():
    # Parameters for the pooling operation
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1

    # Create an input tensor: batch size 2, channels 2, length 10
    input_tensor = nd.arange(List(2, 2, 10))

    # Define the 1D pooling layer
    output = nd.max_pool1d(
        input_tensor,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    # Print the output
    print("Output:\n", output)


def test_max_pool2d():
    # Parameters for the pooling operation
    kernel_size = (3, 3)
    stride = (2, 2)
    padding = (1, 1)
    dilation = (1, 1)

    # Create an input tensor: batch size 2, channels 2, height 10, width 10
    input_tensor = nd.arange(List(2, 2, 10, 10))

    # Define the 2D pooling layer
    output = nd.max_pool2d(
        input_tensor,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    # Print the output
    print("Output:\n", output)


def test_max_pool3d():
    # Parameters for the pooling operation
    kernel_size = (3, 3, 3)
    stride = (2, 2, 2)
    padding = (1, 1, 1)
    dilation = (1, 1, 1)

    # Create an input tensor: batch size 2, channels 2, depth 10, height 10, width 10
    input_tensor = nd.arange(List(2, 2, 10, 10, 10))

    # Define the 3D pooling layer
    output = nd.max_pool3d(
        input_tensor,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    # Print the output
    print("Output:\n", output)


def test_avg_pool1d():
    # Parameters for the pooling operation
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1

    # Create an input tensor: batch size 2, channels 2, length 10
    input_tensor = nd.arange(List(2, 2, 10))

    # Define the 1D average pooling layer
    output = nd.avg_pool1d(
        input_tensor,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    # Print the output
    print("Avg Pool 1D Output:\n", output)


def test_avg_pool2d():
    # Parameters for the pooling operation
    kernel_size = (3, 3)
    stride = (2, 2)
    padding = (1, 1)
    dilation = (1, 1)

    # Create an input tensor: batch size 2, channels 2, height 10, width 10
    input_tensor = nd.arange(List(2, 2, 10, 10))

    # Define the 2D average pooling layer
    output = nd.avg_pool2d(
        input_tensor,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    # Print the output
    print("Avg Pool 2D Output:\n", output)


def test_avg_pool3d():
    # Parameters for the pooling operation
    kernel_size = (3, 3, 3)
    stride = (2, 2, 2)
    padding = (1, 1, 1)
    dilation = (1, 1, 1)

    # Create an input tensor: batch size 2, channels 2, depth 10, height 10, width 10
    input_tensor = nd.arange(List(2, 2, 10, 10, 10))

    # Define the 3D average pooling layer
    output = nd.avg_pool3d(
        input_tensor,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    # Print the output
    print("Avg Pool 3D Output:\n", output)


def main():
    test_conv1d()
    test_conv2d()
    test_conv3d()

    test_max_pool1d()
    test_max_pool2d()
    test_max_pool3d()

    # Add tests for average pooling
    test_avg_pool1d()
    test_avg_pool2d()
    test_avg_pool3d()


# ... [rest of the code remains unchanged] ...
