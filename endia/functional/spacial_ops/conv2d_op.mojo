# ===----------------------------------------------------------------------=== #
# Endia 2024
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from endia import Array
from endia.utils import array_shape_to_list, list_to_array_shape, concat_lists
from endia.utils.aliases import dtype, nelts, NA
from algorithm import vectorize, parallelize
import math
from endia.functional._utils import (
    setup_shape_and_data,
)
from endia.functional._utils import setup_array_shape, contiguous, op_array


struct Conv2d:
    """
    Namespace for 2D convolution operations.
    """

    @staticmethod
    fn compute_shape(inout curr: ArrayShape, args: List[ArrayShape]) raises:
        """
        Computes the shape of an array after a 2-dimensional convolution operation.
        """
        var arg = args[0]  # Input tensor
        var params = array_shape_to_list(args[1])  # Convolution parameters

        var input_shape = arg.shape_node[].shape
        var ndim = len(input_shape)
        if ndim != 4:
            raise "Input must be 4-dimensional (batch_size, in_channels, height, width) for 2D convolution!"

        var batch_size = input_shape[0]
        var in_channels = params[0]
        var out_channels = params[1]
        var kernel_height = params[2]
        var kernel_width = params[3]
        var stride_height = params[4]
        var stride_width = params[5]
        var padding_height = params[6]
        var padding_width = params[7]
        var dilation_height = params[8]
        var dilation_width = params[9]
        var groups = params[10]

        var new_shape = List[Int]()
        new_shape.append(batch_size)
        new_shape.append(out_channels)
        new_shape.append(
            (
                input_shape[2]
                + 2 * padding_height
                - dilation_height * (kernel_height - 1)
                - 1
            )
            // stride_height
            + 1
        )
        new_shape.append(
            (
                input_shape[3]
                + 2 * padding_width
                - dilation_width * (kernel_width - 1)
                - 1
            )
            // stride_width
            + 1
        )
        curr.setup(new_shape)

    @staticmethod
    fn __call__(inout curr: Array, args: List[Array]) raises:
        var params = array_shape_to_list(curr.array_shape().args()[1])

        setup_shape_and_data(curr)

        var in_channels = params[0]
        var out_channels = params[1]
        var kernel_height = params[2]
        var kernel_width = params[3]
        var stride_height = params[4]
        var stride_width = params[5]
        var padding_height = params[6]
        var padding_width = params[7]
        var dilation_height = params[8]
        var dilation_width = params[9]
        var groups = params[10]

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

                for out_y in range(out_shape[2]):
                    var output_y_index = out_y * stride_height - padding_height

                    for out_x in range(out_shape[3]):
                        var value = SIMD[dtype, 1](0)
                        var output_x_index = out_x * stride_width - padding_width

                        for group in range(groups):
                            var group_input_offset = group * in_channels_per_group
                            var group_kernel_offset = group * in_channels_per_group

                            for in_channel in range(in_channels_per_group):
                                var base_input_idx_channel = base_input_idx_batch + (
                                    group_input_offset + in_channel
                                ) * input_stride[
                                    1
                                ]
                                var base_kernel_idx_channel = base_kernel_idx_out_channel + (
                                    group_kernel_offset + in_channel
                                ) * kernel_stride[
                                    1
                                ]

                                for ky in range(kernel_height):
                                    var input_y_index = output_y_index + ky * dilation_height

                                    if (
                                        input_y_index >= 0
                                        and input_y_index < input_shape[2]
                                    ):
                                        for kx in range(kernel_width):
                                            var input_x_index = output_x_index + kx * dilation_width

                                            if (
                                                input_x_index >= 0
                                                and input_x_index
                                                < input_shape[3]
                                            ):
                                                var final_input_idx = base_input_idx_channel + input_y_index * input_stride[
                                                    2
                                                ] + input_x_index * input_stride[
                                                    3
                                                ]
                                                var final_kernel_idx = base_kernel_idx_channel + ky * kernel_stride[
                                                    2
                                                ] + kx * kernel_stride[
                                                    3
                                                ]
                                                value += input_data.load(
                                                    final_input_idx
                                                ) * kernel_data.load(
                                                    final_kernel_idx
                                                )

                        var out_idx = batch * out_stride[
                            0
                        ] + out_channel * out_stride[1] + out_y * out_stride[
                            2
                        ] + out_x * out_stride[
                            3
                        ]
                        out_data.store(
                            out_idx, value + bias_data.load(out_channel)
                        )

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
        kernel_size: Tuple[Int, Int] = (1, 1),
        stride: Tuple[Int, Int] = (1, 1),
        padding: Tuple[Int, Int] = (0, 0),
        dilation: Tuple[Int, Int] = (1, 1),
        groups: Int = 1,
    ) raises -> Array:
        var arr_shape = setup_array_shape(
            List(
                arg0.array_shape(),
                list_to_array_shape(
                    concat_lists(
                        in_channels,
                        out_channels,
                        kernel_size[0],
                        kernel_size[1],
                        stride[0],
                        stride[1],
                        padding[0],
                        padding[1],
                        dilation[0],
                        dilation[1],
                        groups,
                    )
                ),
            ),
            "conv2d_shape",
            Conv2d.compute_shape,
        )

        var args = List(arg0, kernel, bias)

        return op_array(
            arr_shape,
            args,
            NA,
            "conv2d",
            Conv2d.__call__,
            Conv2d.jvp,
            Conv2d.vjp,
            False,
        )


fn conv2d(
    arg0: Array,
    kernel: Array,
    bias: Array,
    in_channels: Int,
    out_channels: Int,
    kernel_size: Tuple[Int, Int] = (1, 1),
    stride: Tuple[Int, Int] = (1, 1),
    padding: Tuple[Int, Int] = (0, 0),
    dilation: Tuple[Int, Int] = (1, 1),
    groups: Int = 1,
) raises -> Array:
    """
    Applies a 2D convolution over an input image composed of several input planes.

    Args:
        arg0: Input tensor of shape (batch_size, in_channels, height, width)
        kernel: Convolution kernel of shape (out_channels, in_channels // groups, kernel_height, kernel_width)
        bias: Bias tensor of shape (out_channels)
        in_channels: Number of channels in the input image
        out_channels: Number of channels produced by the convolution
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution
        padding: Zero-padding added to both sides of the input
        dilation: Spacing between kernel elements
        groups: Number of blocked connections from input channels to output channels

    Returns:
        Output tensor of shape (batch_size, out_channels, output_height, output_width)
    """
    return Conv2d.fwd(
        arg0,
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
