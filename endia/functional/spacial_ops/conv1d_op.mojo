from endia import Array
from endia.utils import array_shape_to_list, list_to_array_shape, concat_lists
from endia.utils.aliases import dtype, nelts, NA
from algorithm import vectorize, parallelize
import math
from endia.functional._utils import (
    setup_shape_and_data,
)
from endia.functional._utils import setup_array_shape, contiguous, op_array


struct Conv1d:
    """
    Namespace for 1D convolution operations.
    """

    @staticmethod
    fn compute_shape(inout curr: ArrayShape, args: List[ArrayShape]) raises:
        """
        Computes the shape of an array after a 1-dimensional convolution operation.
        """
        var arg = args[0]  # Input tensor
        var params = array_shape_to_list(args[1])  # Convolution parameters

        var input_shape = arg.shape_node[].shape
        var ndim = len(input_shape)
        if ndim != 3:
            raise "Input must be 3-dimensional (batch_size, in_channels, length) for 1D convolution!"

        var batch_size = input_shape[0]
        var in_channels = params[0]
        var out_channels = params[1]
        var kernel_size = params[2]
        var stride = params[3]
        var padding = params[4]
        var dilation = params[5]
        var groups = params[6]

        var new_shape = List[Int]()
        new_shape.append(batch_size)
        new_shape.append(out_channels)
        new_shape.append(
            (input_shape[2] + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        )
        curr.setup(new_shape)

    @staticmethod
    fn __call__(inout curr: Array, args: List[Array]) raises:
        var params = array_shape_to_list(curr.array_shape().args()[1])

        setup_shape_and_data(curr)

        var in_channels = params[0]
        var out_channels = params[1]
        var kernel_size = params[2]
        var stride = params[3]
        var padding = params[4]
        var dilation = params[5]
        var groups = params[6]

        var in_channels_per_group = in_channels // groups
        var out_channels_per_group = out_channels // groups

        var input = contiguous(args[0])
        var kernel = contiguous(args[1])
        var bias = contiguous(args[2])

        var out = curr
        var out_shape = out.shape()
        var out_data = out.data()
        var input_data = input.data()
        var kernel_data = kernel.data()
        var bias_data = bias.data()

        var out_stride = out.stride()
        var input_stride = input.stride()
        var kernel_stride = kernel.stride()
        var input_shape = input.shape()

        for batch in range(out_shape[0]):
            var base_input_idx_batch = batch * input_stride[0]

            for out_channel in range(out_channels):
                var base_kernel_idx_out_channel = out_channel * kernel_stride[0]

                for out_index in range(out_shape[2]):
                    var value = bias_data.load(out_channel)
                    var output_index = out_index * stride - padding

                    for group in range(groups):
                        var group_input_offset = group * in_channels_per_group
                        var group_kernel_offset = group * in_channels_per_group

                        for in_channel in range(in_channels_per_group):
                            var base_input_idx_channel = base_input_idx_batch + (group_input_offset + in_channel) * input_stride[1]
                            var base_kernel_idx_channel = base_kernel_idx_out_channel + (group_kernel_offset + in_channel) * kernel_stride[1]

                            for k in range(kernel_size):
                                var input_index = output_index + k * dilation

                                if input_index >= 0 and input_index < input_shape[2]:
                                    var final_input_idx = base_input_idx_channel + input_index * input_stride[2]
                                    var final_kernel_idx = base_kernel_idx_channel + k * kernel_stride[2]

                                    value += input_data.load(final_input_idx) * kernel_data.load(final_kernel_idx)

                    var out_idx = batch * out_stride[0] + out_channel * out_stride[1] + out_index * out_stride[2]
                    out_data.store(out_idx, value)

    @staticmethod
    fn vjp(primals: List[Array], grad: Array, out: Array) raises -> List[Array]:
        return default_vjp(primals, grad, out)

    @staticmethod
    fn jvp(primals: List[Array], tangents: List[Array]) raises -> Array:
        return default_jvp(primals, tangents)

    @staticmethod
    fn fwd(
        arg0: Array,
        kernel: Array,
        bias: Array,
        in_channels: Int,
        out_channels: Int,
        kernel_size: Int,
        stride: Int,
        padding: Int,
        dilation: Int,
        groups: Int,
    ) raises -> Array:
        var arr_shape = setup_array_shape(
            List(
                arg0.array_shape(),
                list_to_array_shape(
                    List(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride,
                        padding,
                        dilation,
                        groups,
                    )
                ),
            ),
            "conv1d_shape",
            Conv1d.compute_shape,
        )

        var args = List(arg0, kernel, bias)

        return op_array(arr_shape, args, NA, "conv1d", Conv1d.__call__, Conv1d.jvp, Conv1d.vjp, False)

fn conv1d(
    arg0: Array,
    kernel: Array,
    bias: Array,
    in_channels: Int,
    out_channels: Int,
    kernel_size: Int,
    stride: Int,
    padding: Int,
    dilation: Int,
    groups: Int,
) raises -> Array:
    """
    Applies a 1D convolution over an input signal composed of several input planes.

    Args:
        arg0: Input tensor of shape (batch_size, in_channels, length)
        kernel: Convolution kernel of shape (out_channels, in_channels // groups, kernel_size)
        bias: Bias tensor of shape (out_channels)
        in_channels: Number of channels in the input image
        out_channels: Number of channels produced by the convolution
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution
        padding: Zero-padding added to both sides of the input
        dilation: Spacing between kernel elements
        groups: Number of blocked connections from input channels to output channels

    Returns:
        Output tensor of shape (batch_size, out_channels, output_length)
    """
    return Conv1d.fwd(arg0, kernel, bias, in_channels, out_channels, kernel_size, stride, padding, dilation, groups)