from endia import Array
from endia.utils import array_shape_to_list, list_to_array_shape, concat_lists
from endia.utils.aliases import dtype, nelts, NA
from algorithm import vectorize, parallelize
import math
from endia.functional._utils import (
    setup_shape_and_data,
)
from endia.functional._utils import setup_array_shape, contiguous, op_array

# from ._utils import DifferentiableBinaryOp, execute_binary_op, binary_op_array


####--------------------------------------------------------------------------------------------------------------------####
#### Convolution
####--------------------------------------------------------------------------------------------------------------------####


# from basalt import Tensor, TensorShape
# from basalt.autograd.attributes import AttributeVector

# from algorithm import parallelize, vectorize, tile
# from utils.loop import unroll


# @always_inline
# fn get_result_shape(
#     input_shape: TensorShape,
#     kernel_shape: TensorShape,
#     padding: StaticIntTuple[2],
#     stride: StaticIntTuple[2],
#     dilation: StaticIntTuple[2],
# ) -> StaticIntTuple[2]:
#     """
#     Calculates the X and Y dimensions of the resulting convolution.
#     Dimensions X, Y are on the end of the shape (..., X, Y)
#         dimension X on index -2.
#         dimension Y on index -1.
#     """

#     var result_x_dim = (
#         (input_shape[-2] + (2 * padding[0]) - dilation[0] * (kernel_shape[-2] - 1) - 1)
#         // stride[0]
#     ) + 1
#     var result_y_dim = (
#         (input_shape[-1] + (2 * padding[1]) - dilation[1] * (kernel_shape[-1] - 1) - 1)
#         // stride[1]
#     ) + 1

#     return StaticIntTuple[2](result_x_dim, result_y_dim)


# struct CONV2D:
#     @staticmethod
#     fn result_shape(
#         input_shape: TensorShape,
#         kernel_shape: TensorShape,
#         bias_shape: TensorShape,
#         attributes: AttributeVector,
#     ) -> TensorShape:
#         # Output shape = [batch, out_channels, oX, oY]

#         var padding = attributes["padding"].value().to_static[2]()
#         var stride = attributes["stride"].value().to_static[2]()
#         var dilation = attributes["dilation"].value().to_static[2]()
#         var res = get_result_shape(input_shape, kernel_shape, padding, stride, dilation)

#         return TensorShape(input_shape[0], kernel_shape[0], res[0], res[1])

#     @staticmethod
#     fn forward[
#         input_shape: TensorShape,
#         kernel_shape: TensorShape,
#         bias_shape: TensorShape,
#         attributes: AttributeVector,
#     ](
#         inout outputs: Tensor[dtype],
#         inputs: Tensor[dtype],
#         kernel: Tensor[dtype],
#         bias: Tensor[dtype],
#     ):
#         """
#         Performs a 2D convolution on the input tensor using the kernel and bias.
#             inputs.shape     [batch, in_channels, iX, iY]
#             kernel.shape     [out_channels, in_channels, kX, kY] (or weights)
#             bias.shape       [out_channels].
#             output.shape     [batch, out_channels, oX, oY].
#         """
#         alias padding = attributes["padding"].value().to_static[2]()
#         alias stride = attributes["stride"].value().to_static[2]()
#         alias dilation = attributes["dilation"].value().to_static[2]()

#         alias padding_x = padding[0]
#         alias padding_y = padding[1]
#         alias stride_x = stride[0]
#         alias stride_y = stride[1]
#         alias dilation_x = dilation[0]
#         alias dilation_y = dilation[1]

#         alias batch_size = input_shape[0]
#         alias in_channels = input_shape[1]
#         alias in_x = input_shape[2]
#         alias in_y = input_shape[3]
#         alias out_channels = kernel_shape[0]
#         alias k_x = kernel_shape[2]
#         alias k_y = kernel_shape[3]
#         alias out_x = output_shape[2]
#         alias out_y = output_shape[3]
#         alias col_x = out_x
#         alias col_y = out_y

#         alias col_shape = TensorShape(
#             batch_size, col_x * col_y, in_channels * k_x * k_y
#         )  # [batch, colX * colY, in_channels * kX * kY]
#         alias output_shape = Self.result_shape(
#             input_shape, kernel_shape, bias_shape, attributes
#         )
#         alias col_shape_stripped = TensorShape(in_channels * k_x * k_y, col_x, col_y)

#         alias inputs_strides = input_shape.strides()
#         alias kernel_strides = kernel_shape.strides()
#         alias outputs_strides = output_shape.strides()
#         alias col_strides = col_shape.strides()

#         var col_ptr = DTypePointer[dtype].alloc(col_shape.num_elements())
#         memset_zero(col_ptr, col_shape.num_elements())

#         @parameter
#         fn im2col(batch: Int):
#             for ux in range(out_x):
#                 for uy in range(out_y):
#                     for in_ch in range(in_channels):
#                         for kx in range(k_x):
#                             for ky in range(k_y):
#                                 var ix = ux * stride_x - padding_x + kx * dilation_x
#                                 var iy = uy * stride_y - padding_y + ky * dilation_y

#                                 if ix < 0 or iy < 0 or ix >= in_x or iy >= in_y:
#                                     continue

#                                 var col_index = (
#                                     batch * col_strides[0]
#                                     + (ux * col_y + uy) * col_strides[1]
#                                     + (in_ch * k_x * k_y + kx * k_y + ky)
#                                 )

#                                 var input_index = (
#                                     batch * inputs_strides[0]
#                                     + in_ch * inputs_strides[1]
#                                     + ix * inputs_strides[2]
#                                     + iy
#                                 )

#                                 col_ptr[col_index] = inputs[input_index]

#         parallelize[im2col](batch_size)

#         @parameter
#         fn conv(batch: Int):
#             for out_ch in range(out_channels):
#                 for ux in range(out_x):
#                     for uy in range(out_y):
#                         var result: SIMD[dtype, nelts] = 0

#                         @parameter
#                         fn v_im2col[_nelts: Int](in_ch_kx_ky: Int):
#                             var col_index = (
#                                 batch * col_strides[0]
#                                 + (ux * col_y + uy) * col_strides[1]
#                                 + in_ch_kx_ky
#                             )

#                             var kernel_index = (
#                                 out_ch * kernel_strides[0] + in_ch_kx_ky
#                             )

#                             @parameter
#                             if _nelts == nelts:
#                                 result += col_ptr.load[width=nelts](
#                                     col_index
#                                 ) * kernel.load[nelts](kernel_index)
#                             else:
#                                 result[0] += (
#                                     col_ptr.load[width=_nelts](col_index)
#                                     * kernel.load[_nelts](kernel_index)
#                                 ).reduce_add()

#                         vectorize[v_im2col, nelts](in_channels * k_x * k_y)

#                         var output_index = (
#                             batch * outputs_strides[0]
#                             + out_ch * outputs_strides[1]
#                             + ux * outputs_strides[2]
#                             + uy
#                         )

#                         outputs[output_index] = result.reduce_add() + bias[out_ch]

#         parallelize[conv](batch_size)

#         col_ptr.free()

#     @staticmethod
#     fn backward[
#         tensor_id: Int,
#         ug_shape: TensorShape,
#         input_shape: TensorShape,
#         kernel_shape: TensorShape,
#         bias_shape: TensorShape,
#         attributes: AttributeVector,
#     ](
#         ug: Tensor[dtype],
#         inputs: Tensor[dtype],
#         kernel: Tensor[dtype],
#         bias: Tensor[dtype],
#     ) -> Tensor[dtype]:
#         """
#         Backward operation of 2D convolution.

#         Upper gradient of shape: [batch, out_channels, uX, uY].
#         """

#         alias padding = attributes["padding"].value().to_static[2]()
#         alias stride = attributes["stride"].value().to_static[2]()
#         alias dilation = attributes["dilation"].value().to_static[2]()
#         alias padding_0 = padding[0]
#         alias padding_1 = padding[1]
#         alias stride_0 = stride[0]
#         alias stride_1 = stride[1]
#         alias dilation_0 = dilation[0]
#         alias dilation_1 = dilation[1]

#         alias inputs_strides = input_shape.strides()
#         alias kernel_strides = kernel_shape.strides()
#         alias ug_strides = ug_shape.strides()
#         alias inputs_strides_0 = inputs_strides[0]
#         alias inputs_strides_1 = inputs_strides[1]
#         alias inputs_strides_2 = inputs_strides[2]
#         alias kernel_strides_0 = kernel_strides[0]
#         alias kernel_strides_1 = kernel_strides[1]
#         alias kernel_strides_2 = kernel_strides[2]
#         alias ug_strides_0 = ug_strides[0]
#         alias ug_strides_1 = ug_strides[1]
#         alias ug_strides_2 = ug_strides[2]

#         alias input_shape_0 = input_shape[0]
#         alias input_shape_1 = input_shape[1]
#         alias input_shape_2 = input_shape[2]
#         alias input_shape_3 = input_shape[3]
#         alias kernel_shape_2 = kernel_shape[2]
#         alias kernel_shape_3 = kernel_shape[3]
#         alias ug_shape_0 = ug_shape[0]
#         alias ug_shape_1 = ug_shape[1]
#         alias ug_shape_2 = ug_shape[2]
#         alias ug_shape_3 = ug_shape[3]

#         var res: Tensor[dtype]

#         @parameter
#         if tensor_id == 0:
#             # Inputs
#             # Sum of upper gradient over batch, X, Y dimensions

#             res = Tensor[dtype](input_shape)

#             @parameter
#             fn input_grad(batch: Int):
#                 for out_ch in range(ug_shape_1):
#                     for ux in range(ug_shape_2):
#                         for uy in range(ug_shape_3):  # For all the element of ug
#                             var ix_base = ux * stride_0 - padding_0
#                             var iy_base = uy * stride_1 - padding_1

#                             var ug_val = ug[
#                                 batch * ug_strides_0
#                                 + out_ch * ug_strides_1
#                                 + ux * ug_strides_2
#                                 + uy
#                             ]

#                             for in_ch in range(input_shape_1):
#                                 for kx in range(kernel_shape_2):
#                                     for ky in range(kernel_shape_3):
#                                         var ix = ix_base + kx * dilation_0
#                                         var iy = iy_base + ky * dilation_1

#                                         if (
#                                             ix < 0
#                                             or iy < 0
#                                             or ix >= input_shape_2
#                                             or iy >= input_shape_3
#                                         ):
#                                             continue

#                                         var kernel_index = (
#                                             out_ch * kernel_strides_0
#                                             + in_ch * kernel_strides_1
#                                             + kx * kernel_strides_2
#                                             + ky
#                                         )

#                                         var input_index = (
#                                             batch * inputs_strides_0
#                                             + in_ch * inputs_strides_1
#                                             + ix * inputs_strides_2
#                                             + iy
#                                         )
#                                         res[input_index] += (
#                                             kernel[kernel_index] * ug_val
#                                         )

#             parallelize[input_grad](input_shape_0)

#         elif tensor_id == 1:
#             # Kernel
#             # Sum of upper gradient over batch and X, Y dimensions
#             res = Tensor[dtype](kernel_shape)

#             @parameter
#             fn kernel_grad(out_ch: Int):
#                 var channel_offset = out_ch * kernel_strides_0
#                 for k in range(input_shape_1 * kernel_shape_2 * kernel_shape_3):
#                     var in_ch_kx_ky = divmod(k, kernel_shape_3)
#                     var in_ch = k // (kernel_shape_2 * kernel_shape_3)
#                     var kx = in_ch_kx_ky[0] % kernel_shape_2
#                     var ky = in_ch_kx_ky[1]

#                     # TODO: Cant vectorize since you are going different directions across input and upper grad
#                     # But theoretically could transpose or split somehow
#                     var result: Scalar[dtype] = 0
#                     for batch in range(input_shape_0):
#                         for ux in range(ug_shape_2):
#                             for uy in range(ug_shape_3):
#                                 var ix = ux * stride_0 - padding_0 + kx * dilation_0
#                                 var iy = uy * stride_1 - padding_1 + ky * dilation_1

#                                 if (
#                                     ix < 0
#                                     or iy < 0
#                                     or ix >= input_shape_2
#                                     or iy >= input_shape_3
#                                 ):
#                                     continue

#                                 var input_index = batch * inputs_strides_0 + in_ch * inputs_strides_1 + ix * inputs_strides_2 + iy
#                                 var ug_index = batch * ug_strides_0 + out_ch * ug_strides_1 + ux * ug_strides_2 + uy

#                                 result += inputs[input_index] * ug[ug_index]

#                     var kernel_index = channel_offset + k
#                     res[kernel_index] = result

#             parallelize[kernel_grad](ug_shape_1)

#         else:
#             # Bias
#             # Sum of upper gradient over batch and X, Y dimensions
#             # out_channels == ug_shape[1] == bias_shape[0]
#             res = Tensor[dtype](bias_shape)

#             # Psuedocode
#             # For every channel in the bias tensor,
#             # Iterate over the upper gradient across the batch
#             # For each batch, sum the upper gradient across X, Y dimensions
#             # Add the sum to the bias tensor

#             @parameter
#             fn bias_grad(out_ch: Int):
#                 var channel_offset = out_ch * ug_strides_1
#                 var sum: Scalar[dtype] = 0
#                 for batch in range(ug_shape_0):
#                     var batch_offset = batch * ug_strides_0 + channel_offset

#                     @parameter
#                     fn vec_sum[Nelts: Int](ux_uy: Int):
#                         sum += ug.load[Nelts](batch_offset + ux_uy).reduce_add()

#                     vectorize[vec_sum, nelts, size = ug_shape_2 * ug_shape_3]()

#                 res[out_ch] = sum

#             parallelize[bias_grad](ug_shape_1)

#         return res


fn convnd_shape(inout curr: ArrayShape, args: List[ArrayShape]) raises:
    """
    Computes the shape of an array after a n-dimensional convolution operation.

    Args:
        curr: The ArrayShape to store the result of the computation.
        args: The input ArrayShape, and the convolution parameters encoded in an ArrayShape.

    #### Constraints:
    - The number of dimensions of the input ArrayShape must be 2, 3, or 4 for 1D, 2D, or 3D convolution respectively.
    """
    var arg = args[0]  # Input tensor
    var params = array_shape_to_list(args[1])  # Convolution parameters

    var input_shape = arg.shape_node[].shape
    var ndim = len(input_shape) - 1
    if ndim < 1 or ndim > 3:
        raise Error(
            "Input must be 2D, 3D, or 4D for 1D, 2D, or 3D convolution"
            " respectively"
        )

    var in_channels = params[0]
    var out_channels = params[1]
    var groups = params[len(params) - 1] if len(params) > 2 + 4 * ndim else 1

    if input_shape[0] != in_channels:
        raise Error("Input channel dimension doesn't match in_channels")

    if in_channels % groups != 0 or out_channels % groups != 0:
        raise Error("in_channels and out_channels must be divisible by groups")

    fn create_list(size: Int, val: Int) -> List[Int]:
        var res = List[Int]()
        for _ in range(size):
            res.append(val)
        return res

    var kernel_size = params[2 : 2 + ndim]
    var stride = params[2 + ndim : 2 + 2 * ndim] if len(
        params
    ) > 2 + ndim else create_list(ndim, 1)
    var padding = params[2 + 2 * ndim : 2 + 3 * ndim] if len(
        params
    ) > 2 + 2 * ndim else create_list(ndim, 0)
    var dilation = params[2 + 3 * ndim : 2 + 4 * ndim] if len(
        params
    ) > 2 + 3 * ndim else create_list(ndim, 1)

    var new_shape = List[Int]()
    new_shape.append(out_channels)
    var out_channel_stride = 1
    for i in range(1, len(input_shape)):
        out_channel_stride *= input_shape[i]

    for i in range(ndim):
        var input_size = input_shape[i + 1]
        var out_size = (
            input_size + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1
        ) // stride[i] + 1
        if out_size <= 0:
            raise "Calculated out size is not positive for dimension" + str(i)
        new_shape.append(out_size)

    curr.setup(new_shape)


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
    fn fwd(inout curr: Array, args: List[Array]) raises:
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

        # lets do naive for loops ot compute the conv1d
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

        print("input:")
        print(input)
        print("kernel:")
        print(kernel)
        print("bias:")
        print(bias)

        for i in range(out_shape[0]):
            for j in range(out_shape[1]):
                var start = j * stride - padding
                var end = start + kernel_size * dilation
                var sum = SIMD[dtype, 1](0)
                for k in range(kernel_size):
                    var idx = i * input_stride[0] + (
                        start + k * dilation
                    ) * input_stride[1]
                    var kernel_idx = j * kernel_stride[0] + k * kernel_stride[1]
                    sum += input_data[idx] * kernel_data[kernel_idx]
                out_data[i * out_stride[0] + j * out_stride[1]] = (
                    sum + bias_data[j]
                )

    fn vjp(primals: List[Array], grad: Array, out: Array) raises -> List[Array]:
        return List(grad)

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
        "convnd_shape",
        convnd_shape,
    )

    var args = List(arg0, kernel, bias)

    return op_array(arr_shape, args, NA, "conv1d", fwd, default_jvp, vjp, False)


fn conv2d(
    arg0: Array,
    in_channels: Int,
    out_channels: Int,
    kernel_size: List[Int],
    stride: List[Int],
    padding: List[Int],
    dilation: List[Int],
    groups: Int,
) raises -> Array:
    fn fwd(inout curr: Array, args: List[Array]) raises:
        setup_shape_and_data(curr)

    fn vjp(primals: List[Array], grad: Array, out: Array) raises -> List[Array]:
        return List(grad)

    var arr_shape = setup_array_shape(
        List(
            arg0.array_shape(),
            list_to_array_shape(
                concat_lists(
                    List(in_channels),
                    List(out_channels),
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    groups,
                )
            ),
        ),
        "convnd_shape",
        convnd_shape,
    )

    return op_array(arr_shape, arg0, NA, "conv2d", fwd, default_jvp, vjp, False)


fn conv3d(
    arg0: Array,
    in_channels: Int,
    out_channels: Int,
    kernel_size: List[Int],
    stride: List[Int],
    padding: List[Int],
    dilation: List[Int],
    groups: Int,
) raises -> Array:
    fn fwd(inout curr: Array, args: List[Array]) raises:
        setup_shape_and_data(curr)

    fn vjp(primals: List[Array], grad: Array, out: Array) raises -> List[Array]:
        return List(grad)

    var arr_shape = setup_array_shape(
        List(
            arg0.array_shape(),
            list_to_array_shape(
                concat_lists(
                    List(in_channels),
                    List(out_channels),
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    groups,
                )
            ),
        ),
        "convnd_shape",
        convnd_shape,
    )

    return op_array(arr_shape, arg0, NA, "conv3d", fwd, default_jvp, vjp, False)
