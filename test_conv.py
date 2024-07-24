import torch
import torch.nn as nn
import torch.nn.functional as F


def test_conv1d():
    # Create input tensor: batch size 2, 3 channels, 5 elements each
    batch_size = 2
    in_channels = 2
    out_channels = 3
    kernel_size = 3
    elements = 6
    stride = 2
    padding = 1
    dilation = 1
    groups = 1
    num_elements = batch_size * in_channels * elements

    input = torch.arange(0, num_elements, dtype=torch.float32).reshape(
        batch_size, in_channels, elements
    )
    # print("Input shape:", input.shape)
    # print("Input:\n", input)

    # Define the convolution layer
    conv1d = nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    # Set kernel to ones and bias to zero
    with torch.no_grad():
        conv1d.weight.fill_(1.0)
        conv1d.bias.fill_(0.0)

    # print("\nKernel (weight) shape:", conv1d.weight.shape)
    # print("Kernel (weight):\n", conv1d.weight.data)
    # print("Bias:\n", conv1d.bias.data)

    # Perform the convolution
    output = conv1d(input)

    print("\nOutput shape:", output.shape)
    print("Output:\n", output)


def test_conv2d():
    # we test the 2d convolution operation
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
    num_elements = batch_size * in_channels * elements * elements

    input = torch.arange(0, num_elements, dtype=torch.float32).reshape(
        batch_size, in_channels, elements, elements
    )
    # print("Input shape:", input.shape)
    # print("Input:\n", input)

    # Define the convolution layer
    conv2d = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=(kernel_height, kernel_width),
        stride=(stride_height, stride_width),
        padding=(padding_height, padding_width),
        dilation=(dilation_height, dilation_width),
        groups=groups,
    )

    # Set kernel to ones and bias to zero
    with torch.no_grad():
        conv2d.weight.fill_(1.0)
        conv2d.bias.fill_(0.0)

    # print("\nKernel (weight) shape:", conv2d.weight.shape)
    # print("Kernel (weight):\n", conv2d.weight.data)
    # print("Bias:\n", conv2d.bias.data)

    # Perform the convolution
    output = conv2d(input)

    print("\nOutput shape:", output.shape)
    print("Output:\n", output)


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

    # Create input tensor
    input_tensor = torch.arange(
        batch_size * in_channels * depth * height * width, dtype=torch.float32
    ).reshape(batch_size, in_channels, depth, height, width)
    # print("Input shape:", input_tensor.shape)
    # print("Input:\n", input_tensor)

    # Define the convolution layer
    conv3d = nn.Conv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=(kernel_depth, kernel_height, kernel_width),
        stride=(stride_depth, stride_height, stride_width),
        padding=(padding_depth, padding_height, padding_width),
        dilation=(dilation_depth, dilation_height, dilation_width),
        groups=groups,
    )

    # Set kernel to ones and bias to zero
    with torch.no_grad():
        conv3d.weight.fill_(1.0)
        conv3d.bias.fill_(0.0)

    # print("\nKernel (weight) shape:", conv3d.weight.shape)
    # print("Kernel (weight):\n", conv3d.weight)
    # print("Bias:\n", conv3d.bias)

    # Perform the convolution
    output = conv3d(input_tensor)

    print("\nOutput shape:", output.shape)
    print("Output:\n", output)


def test_pool1d_pytorch():
    # Parameters for the pooling operation
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1

    # Create an input tensor: batch size 2, channels 2, length 10
    input_tensor = torch.arange(0, 40, dtype=torch.float32).reshape(2, 2, 10)

    # Define the 1D pooling layer
    pool1d = nn.MaxPool1d(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    # Apply the pooling operation
    output = pool1d(input_tensor)

    # Print the output
    print("Output:\n", output)


def test_max_pool2d_pytorch():
    # Parameters for the pooling operation
    kernel_size = (3, 3)
    stride = (2, 2)
    padding = (1, 1)
    dilation = (1, 1)

    # Create an input tensor: batch size 2, channels 2, height 10, width 10
    input_tensor = torch.arange(0, 400, dtype=torch.float32).reshape(
        2, 2, 10, 10
    )

    # Define the 2D pooling layer
    pool2d = nn.MaxPool2d(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    # Apply the pooling operation
    output = pool2d(input_tensor)

    # Print the output
    print("Output:\n", output)


def test_max_pool3d_pytorch():
    # Parameters for the pooling operation
    kernel_size = (3, 3, 3)
    stride = (2, 2, 2)
    padding = (1, 1, 1)
    dilation = (1, 1, 1)

    # Create an input tensor: batch size 2, channels 2, depth 10, height 10, width 10
    input_tensor = torch.arange(0, 4000, dtype=torch.float32).reshape(
        2, 2, 10, 10, 10
    )

    # Define the 3D pooling layer
    pool3d = nn.MaxPool3d(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    # Apply the pooling operation
    output = pool3d(input_tensor)

    # Print the output
    print("Output:\n", output)


def test_avg_pool1d_pytorch():
    # Parameters for the pooling operation
    kernel_size = 3
    stride = 2
    padding = 1

    # Create an input tensor: batch size 2, channels 2, length 10
    input_tensor = torch.arange(0, 40, dtype=torch.float32).reshape(2, 2, 10)

    # Define the 1D average pooling layer
    pool1d = nn.AvgPool1d(
        kernel_size=kernel_size, stride=stride, padding=padding
    )

    # Apply the pooling operation
    output = pool1d(input_tensor)

    # Print the output
    print("Avg Pool 1D Output:\n", output)


def test_avg_pool2d_pytorch():
    # Parameters for the pooling operation
    kernel_size = (3, 3)
    stride = (2, 2)
    padding = (1, 1)

    # Create an input tensor: batch size 2, channels 2, height 10, width 10
    input_tensor = torch.arange(0, 400, dtype=torch.float32).reshape(
        2, 2, 10, 10
    )

    # Define the 2D average pooling layer
    pool2d = nn.AvgPool2d(
        kernel_size=kernel_size, stride=stride, padding=padding
    )

    # Apply the pooling operation
    output = pool2d(input_tensor)

    # Print the output
    print("Avg Pool 2D Output:\n", output)


def test_avg_pool3d_pytorch():
    # Parameters for the pooling operation
    kernel_size = (3, 3, 3)
    stride = (2, 2, 2)
    padding = (1, 1, 1)

    # Create an input tensor: batch size 2, channels 2, depth 10, height 10, width 10
    input_tensor = torch.arange(0, 4000, dtype=torch.float32).reshape(
        2, 2, 10, 10, 10
    )

    # Define the 3D average pooling layer
    pool3d = nn.AvgPool3d(
        kernel_size=kernel_size, stride=stride, padding=padding
    )

    # Apply the pooling operation
    output = pool3d(input_tensor)

    # Print the output
    print("Avg Pool 3D Output:\n", output)


if __name__ == "__main__":
    test_conv1d()
    test_conv2d()
    test_conv3d()

    test_pool1d_pytorch()
    test_max_pool2d_pytorch()
    test_max_pool3d_pytorch()

    # Add tests for average pooling
    test_avg_pool1d_pytorch()
    test_avg_pool2d_pytorch()
    test_avg_pool3d_pytorch()
