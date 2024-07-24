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


struct Conv3d:
    """
    Namespace for 3D convolution operations.
    """

    @staticmethod
    fn compute_shape(inout curr: ArrayShape, args: List[ArrayShape]) raises:
        """
        Computes the shape of an array after a 3-dimensional convolution operation.

        Args:
            curr: The ArrayShape to store the result of the computation.
            args: The input ArrayShape, and the convolution parameters encoded in an ArrayShape.
        """
        var arg = args[0]  # Input tensor
        var params = array_shape_to_list(args[1])  # Convolution parameters

        var input_shape = arg.shape_node[].shape
        var ndim = len(input_shape)
        if ndim != 5:
            raise "Input must be 5-dimensional (batch_size, in_channels, depth, height, width) for 3D convolution!"

        var batch_size = input_shape[0]
        var in_channels = params[0]
        var out_channels = params[1]
        var kernel_depth = params[2]
        var kernel_height = params[3]
        var kernel_width = params[4]
        var stride_depth = params[5]
        var stride_height = params[6]
        var stride_width = params[7]
        var padding_depth = params[8]
        var padding_height = params[9]
        var padding_width = params[10]
        var dilation_depth = params[11]
        var dilation_height = params[12]
        var dilation_width = params[13]
        var groups = params[14]

        var new_shape = List[Int]()
        new_shape.append(batch_size)
        new_shape.append(out_channels)
        new_shape.append(
            (
                input_shape[2]
                + 2 * padding_depth
                - dilation_depth * (kernel_depth - 1)
                - 1
            )
            // stride_depth
            + 1
        )
        new_shape.append(
            (
                input_shape[3]
                + 2 * padding_height
                - dilation_height * (kernel_height - 1)
                - 1
            )
            // stride_height
            + 1
        )
        new_shape.append(
            (
                input_shape[4]
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
        # Extract shape parameters
        var params = array_shape_to_list(curr.array_shape().args()[1])

        # Setup shape and data
        setup_shape_and_data(curr)

        var in_channels = params[0]
        var out_channels = params[1]
        var kernel_depth = params[2]
        var kernel_height = params[3]
        var kernel_width = params[4]
        var stride_depth = params[5]
        var stride_height = params[6]
        var stride_width = params[7]
        var padding_depth = params[8]
        var padding_height = params[9]
        var padding_width = params[10]
        var dilation_depth = params[11]
        var dilation_height = params[12]
        var dilation_width = params[13]
        var groups = params[14]

        var in_channels_per_group = in_channels // groups
        var out_channels_per_group = out_channels // groups

        # Make input, kernel, and bias contiguous
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

                for out_z in range(out_shape[2]):
                    var output_z_index = out_z * stride_depth - padding_depth

                    for out_y in range(out_shape[3]):
                        var output_y_index = out_y * stride_height - padding_height

                        for out_x in range(out_shape[4]):
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

                                    for kz in range(kernel_depth):
                                        var input_z_index = output_z_index + kz * dilation_depth

                                        if (
                                            input_z_index >= 0
                                            and input_z_index < input_shape[2]
                                        ):
                                            for ky in range(kernel_height):
                                                var input_y_index = output_y_index + ky * dilation_height

                                                if (
                                                    input_y_index >= 0
                                                    and input_y_index
                                                    < input_shape[3]
                                                ):
                                                    for kx in range(
                                                        kernel_width
                                                    ):
                                                        var input_x_index = output_x_index + kx * dilation_width

                                                        if (
                                                            input_x_index >= 0
                                                            and input_x_index
                                                            < input_shape[4]
                                                        ):
                                                            var final_input_idx = base_input_idx_channel + input_z_index * input_stride[
                                                                2
                                                            ] + input_y_index * input_stride[
                                                                3
                                                            ] + input_x_index * input_stride[
                                                                4
                                                            ]
                                                            var final_kernel_idx = base_kernel_idx_channel + kz * kernel_stride[
                                                                2
                                                            ] + ky * kernel_stride[
                                                                3
                                                            ] + kx * kernel_stride[
                                                                4
                                                            ]
                                                            value += input_data.load(
                                                                final_input_idx
                                                            ) * kernel_data.load(
                                                                final_kernel_idx
                                                            )

                            var out_idx = batch * out_stride[
                                0
                            ] + out_channel * out_stride[
                                1
                            ] + out_z * out_stride[
                                2
                            ] + out_y * out_stride[
                                3
                            ] + out_x * out_stride[
                                4
                            ]
                            out_data.store(
                                out_idx, value + bias_data.load(out_channel)
                            )

        _ = curr
        _ = kernel
        _ = bias
        _ = in_channels
        _ = out_channels
        _ = kernel_depth
        _ = kernel_height
        _ = kernel_width
        _ = stride_depth
        _ = stride_height
        _ = stride_width
        _ = padding_depth
        _ = padding_height
        _ = padding_width
        _ = dilation_depth
        _ = dilation_height
        _ = dilation_width
        _ = groups

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
        kernel_size: Tuple[Int, Int, Int] = (1, 1, 1),
        stride: Tuple[Int, Int, Int] = (1, 1, 1),
        padding: Tuple[Int, Int, Int] = (0, 0, 0),
        dilation: Tuple[Int, Int, Int] = (1, 1, 1),
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
                        kernel_size[2],
                        stride[0],
                        stride[1],
                        stride[2],
                        padding[0],
                        padding[1],
                        padding[2],
                        dilation[0],
                        dilation[1],
                        dilation[2],
                        groups,
                    )
                ),
            ),
            "conv3d_shape",
            # conv3d_shape,
            Conv3d.compute_shape,
        )

        var args = List(arg0, kernel, bias)

        # return op_array(arr_shape, args, NA, "conv3d", fwd, default_jvp, vjp, False)
        return op_array(
            arr_shape,
            args,
            NA,
            "conv3d",
            Conv3d.__call__,
            Conv3d.jvp,
            Conv3d.vjp,
            False,
        )


fn conv3d(
    arg0: Array,
    kernel: Array,
    bias: Array,
    in_channels: Int,
    out_channels: Int,
    kernel_size: Tuple[Int, Int, Int] = (1, 1, 1),
    stride: Tuple[Int, Int, Int] = (1, 1, 1),
    padding: Tuple[Int, Int, Int] = (0, 0, 0),
    dilation: Tuple[Int, Int, Int] = (1, 1, 1),
    groups: Int = 1,
) raises -> Array:
    """
    Applies a 3D convolution operation over an input array.

    Args:
        arg0: The input array.
        kernel: The convolution kernel.
        bias: The bias tensor.
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        kernel_size: The size of the kernel (depth, height, width).
        stride: The stride of the convolution operation. Defaults to (1, 1, 1).
        padding: The padding to apply to the input. Defaults to (0, 0, 0).
        dilation: The dilation to apply to the input. Defaults to (1, 1, 1).
        groups: The number of groups to split the input and output channels into. Defaults to 1.

    Returns:
        Array: The output array.
    """
    return Conv3d.fwd(
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
