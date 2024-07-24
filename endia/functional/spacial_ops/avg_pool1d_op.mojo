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


struct AvgPool1d:
    """
    Namespace for 1D average pooling operations.
    """

    @staticmethod
    fn compute_shape(inout curr: ArrayShape, args: List[ArrayShape]) raises:
        """
        Computes the shape of an array after a 1-dimensional average pooling operation with dilation.
        """
        var arg = args[0]  # Input tensor
        var params = array_shape_to_list(args[1])  # Pooling parameters

        var input_shape = arg.shape_node[].shape
        var ndim = len(input_shape)
        if ndim != 3:
            raise "Input must be 3-dimensional (batch_size, channels, length) for 1D pooling!"

        var batch_size = input_shape[0]
        var channels = input_shape[1]
        var kernel_size = params[0]
        var stride = params[1]
        var padding = params[2]
        var dilation = params[3]

        var new_shape = List[Int]()
        new_shape.append(batch_size)
        new_shape.append(channels)
        new_shape.append(
            (input_shape[2] + 2 * padding - dilation * (kernel_size - 1) - 1)
            // stride
            + 1
        )
        curr.setup(new_shape)

    @staticmethod
    fn __call__(inout curr: Array, args: List[Array]) raises:
        var params = array_shape_to_list(curr.array_shape().args()[1])

        setup_shape_and_data(curr)

        var kernel_size = params[0]
        var stride = params[1]
        var padding = params[2]
        var dilation = params[3]

        var input = contiguous(args[0])

        var out = curr
        var out_shape = out.shape()
        var out_data = out.data()
        var input_data = input.data()

        var out_stride = out.stride()
        var input_stride = input.stride()
        var input_shape = input.shape()

        for batch in range(out_shape[0]):
            for channel in range(out_shape[1]):
                for out_index in range(out_shape[2]):
                    var start = out_index * stride - padding
                    var sum_val = SIMD[dtype, 1](0)
                    var count = 0

                    for k in range(kernel_size):
                        var i = start + k * dilation
                        if i >= 0 and i < input_shape[2]:
                            var idx = batch * input_stride[
                                0
                            ] + channel * input_stride[1] + i * input_stride[2]
                            sum_val += input_data.load(idx)
                            count += 1

                    var out_idx = batch * out_stride[0] + channel * out_stride[
                        1
                    ] + out_index * out_stride[2]
                    out_data.store(
                        out_idx,
                        sum_val / count if count > 0 else SIMD[dtype, 1](0),
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
        kernel_size: Int,
        stride: Int = 1,
        padding: Int = 0,
        dilation: Int = 1,
    ) raises -> Array:
        var arr_shape = setup_array_shape(
            List(
                arg0.array_shape(),
                list_to_array_shape(
                    List(
                        kernel_size,
                        stride,
                        padding,
                        dilation,
                    )
                ),
            ),
            "avg_pool1d_shape",
            AvgPool1d.compute_shape,
        )

        var args = List(arg0)

        # return op_array(arr_shape, args, NA, "avg_pool1d", fwd, default_jvp, vjp, False)
        return op_array(
            arr_shape,
            args,
            NA,
            "avg_pool1d",
            AvgPool1d.__call__,
            AvgPool1d.jvp,
            AvgPool1d.vjp,
            False,
        )


fn avg_pool1d(
    arg0: Array,
    kernel_size: Int,
    stride: Int = 1,
    padding: Int = 0,
    dilation: Int = 1,
) raises -> Array:
    """
    Applies a 1D average pooling operation over an input array.

    Args:
        arg0: The input array.
        kernel_size: The size of the kernel.
        stride: The stride of the pooling operation. Defaults to 1.
        padding: The padding to apply to the input. Defaults to 0.
        dilation: The dilation to apply to the input. Defaults to 1.

    Returns:
        Array: The output array.
    """
    return AvgPool1d.fwd(arg0, kernel_size, stride, padding, dilation)
