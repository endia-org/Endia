import endia as nd


fn test_conv(ndim: Int) raises:
    print("Testing " + String(ndim) + "D convolution:")

    var in_channels = 2
    var out_channels = 3

    var input_shape: List[Int]
    var kernel_size: List[Int]
    var stride: List[Int]
    var padding: List[Int]
    var dilation: List[Int]

    var groups = 1

    if ndim == 1:
        input_shape = List[Int](in_channels, 8)
        kernel_size = List[Int](3)
        stride = List[Int](1)
        padding = List[Int](0)
        dilation = List[Int](0)

        var a = nd.arange(input_shape)
        var kernel = nd.ones(
            List[Int](out_channels, in_channels, kernel_size[0])
        )
        var bias = nd.zeros(List[Int](out_channels))

        # print("Input:")
        # print(a)

        # print("kernel:")
        # print(kernel)

        # print("bias:")
        # print(bias)

        var res = nd.conv1d(
            a,
            kernel,
            bias,
            in_channels,
            out_channels,
            kernel_size[0],
            stride[0],
            padding[0],
            dilation[0],
            groups,
        )

        print("out:")
        print(res)

        # print()
    # elif ndim == 2:
    #     input_shape = List[Int](in_channels, 32, 32)
    #     kernel_size = List[Int](30, 3)
    #     stride = List[Int](1, 1)
    #     padding = List[Int](1, 1)
    #     dilation = List[Int](1, 1)

    #     var a = nd.arange(input_shape)
    #     print("Input:", a)

    #     var res = nd.conv2d(a, in_channels, out_channels, kernel_size, stride, padding, dilation, groups)

    #     print("out:", res)
    # else:  # 3D
    #     input_shape = List[Int](in_channels, 16, 16, 16)
    #     kernel_size = List[Int](40, 40, 40)
    #     stride = List[Int](2,2,1)
    #     padding = List[Int](1, 1, 1)
    #     dilation = List[Int](1, 1, 1)

    #     var a = nd.arange(input_shape)
    #     print("Input:")
    #     print(a)

    #     var res = nd.conv3d(a, in_channels, out_channels, kernel_size, stride, padding, dilation, groups)

    #     print("out:")
    #     print(res)


fn test_conv() raises:
    test_conv(1)
    # test_conv(2)
    # test_conv(3)
