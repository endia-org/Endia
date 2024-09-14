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
from endia.utils.aliases import dtype, nelts, NA
from algorithm import vectorize, parallelize
import math
from endia.functional._utils import (
    op_array,
    setup_shape_and_data,
    contiguous,
    compute_stride,
    compute_indeces_for_matmul,
    execute_copy_raw,
    setup_array_shape,
    compute_shape,
)

from endia.functional import permute, squeeze, expand

####--------------------------------------------------------------------------------------------------------------------####
#### Matrix Multiplication
####--------------------------------------------------------------------------------------------------------------------####


fn matmul_shape(inout curr: ArrayShape, args: List[ArrayShape]) raises:
    """
    Computes the shape of the result of a batched matrix multiplication operation. Given a lhs Array Shape (_,M,K) and a rhs Array Shape (_,K,N), the result will be (_,M,N).
    It also performs broadcasting on the two input shapes to make them compatible for matrix multiplication.

    Args:
        curr: The ArrayShape to store the result of the computation.
        args: Lhs ArrayShape, rhs ArrayShape.

    #### Constraints:
    - The number of dimensions of the lhs ArrayShape and rhs ArrayShape must be greater than or equal to 2.
    - The last dimension of the lhs ArrayShape must be equal to the second-to-last dimension of the rhs ArrayShape.
    """
    var arg0 = args[0]
    var arg1 = args[1]
    var shape0 = arg0.shape_node[].shape
    var shape1 = arg1.shape_node[].shape
    var ndim0 = arg0.shape_node[].ndim
    var ndim1 = arg1.shape_node[].ndim
    if ndim0 < 2 or ndim1 < 2:
        raise "Invalid shapes for matmul, i.e. ndim < 2"
    if shape0[ndim0 - 1] != shape1[ndim1 - 2]:
        raise "Invalid shapes for matmul, i.e. shapes are not compatible for matmul"
    var shape = List[Int]()
    var diff = len(shape0) - len(shape1)

    if diff > 0:
        # shape0 has more dimensions
        for i in range(diff):
            shape.append(shape0[i])
        for i in range(len(shape1) - 2):
            if shape0[i + diff] == shape1[i]:
                shape.append(shape0[i + diff])
            elif shape0[i + diff] == 1:
                shape.append(shape1[i])
            elif shape1[i] == 1:
                shape.append(shape0[i + diff])
            else:
                raise "Error: matmul Incompatible shapes for broadcasting"
    else:
        # shape1 has more dimensions
        for i in range(-diff):
            shape.append(shape1[i])
        for i in range(len(shape0) - 2):
            if shape1[i - diff] == shape0[i]:
                shape.append(shape1[i - diff])
            elif shape1[i - diff] == 1:
                shape.append(shape0[i])
            elif shape0[i] == 1:
                shape.append(shape1[i - diff])
            else:
                raise "Error: matmul Incompatible shapes for broadcasting"
    shape.append(shape0[ndim0 - 2])
    shape.append(shape1[ndim1 - 1])
    curr.setup(shape)


fn matmul_fwd(inout curr: Array, args: List[Array]) raises:
    """
    Perfomr batched matrix multiplication between two arrays and stores the result in the current array (curr). The function assumes that the shape and data of the args are already set up.

    Args:
        curr: The current array, must be mutable.
        args: The two arrays to multiply.

    Constraints:
        The shapes of the two arrays must be compatible for matrix multiplication, i.e. the last dimension of the first array must be equal to the second last dimension of the second array.
    """
    setup_shape_and_data(curr)

    # make sure that the first argument is contiguous
    var arg0 = contiguous(args[0])

    # the follwing code is to make arg1 to be a contiguous transposed version of args[1]
    # we do so much here since we can't call teh high level .T() inside a fwd fucntion
    var second_shape = args[1].shape()
    var second_stride = args[1].stride()
    var secod_storage_offset = args[1].storage_offset()
    second_shape[-2] = args[1].shape()[-1]
    second_shape[-1] = args[1].shape()[-2]
    second_stride[-2] = args[1].stride()[-1]
    second_stride[-1] = args[1].stride()[-2]
    var second_transposed_array_shape = ArrayShape(
        second_shape, second_stride, secod_storage_offset
    )
    var second_expected_stride = compute_stride(args[1].shape())
    var is_same = True
    for i in range(second_shape.size):
        if second_stride[i] != second_expected_stride[i]:
            is_same = False
            break
    var arg1 = args[1] if is_same else Array(
        second_shape, is_complex=args[1].is_complex()
    )
    if not is_same:
        execute_copy_raw(
            args[1].data(),
            arg1.data(),
            second_transposed_array_shape,
            args[1].is_complex(),
        )

    # define some helper variables
    var res_rows = curr.shape()[curr.ndim() - 2]
    var res_cols = curr.shape()[curr.ndim() - 1]
    var lhs_cols = arg0.shape()[arg0.ndim() - 1]
    var lhs_stride = arg0.stride()
    var rhs_stride = arg1.stride()
    var res_stride = curr.stride()
    var lhs_rank = arg0.ndim()
    var rhs_rank = arg1.ndim()
    var res_rank = curr.ndim()
    var lhs_data = arg0.data()
    var rhs_data = arg1.data()
    var res_data = curr.data()

    # cache some of teh often used shape values
    var k_end = lhs_cols - (lhs_cols % nelts[dtype]())
    var lhs_stride_min_1 = lhs_stride[lhs_rank - 1]
    var res_stride_min_1 = res_stride[res_rank - 1]
    var rhs_stride_min_1 = rhs_stride[rhs_rank - 1]
    var lhs_stride_min_2 = lhs_stride[lhs_rank - 2]
    var res_stride_min_2 = res_stride[res_rank - 2]
    var rhs_stride_min_2 = rhs_stride[rhs_rank - 2]

    # go through all matrix mbatches and compute the matmul respectively
    for i in range(0, curr.size(), res_rows * res_cols):
        var indeces = compute_indeces_for_matmul(i, curr, arg0, arg1)
        var lhs_idx_start = indeces[0]
        var rhs_idx_start = indeces[1]

        # perform the matmul
        # @parameter
        # fn matmul_par(m: Int):
        for m in range(res_rows):
            var res_idx_0 = i + m * res_stride_min_2
            var lhs_idx_0 = lhs_idx_start + m * lhs_stride_min_2

            for n in range(res_cols):
                var rhs_idx_0 = rhs_idx_start + n * rhs_stride_min_2
                var res_idx = res_idx_0 + n * res_stride_min_1

                if not curr.is_complex():
                    # we loop over the last dimension of the lhs and rhs matrices
                    # since we transposed the rhs, we can vectorize along both arrays
                    var sum = SIMD[dtype, nelts[dtype]()](0)
                    for k in range(0, k_end, nelts[dtype]()):
                        var lhs_idx = lhs_idx_0 + k * lhs_stride_min_1
                        var rhs_idx = rhs_idx_0 + k * rhs_stride_min_1
                        sum += lhs_data.load[width = nelts[dtype]()](
                            lhs_idx
                        ) * rhs_data.load[width = nelts[dtype]()](rhs_idx)

                    var sum_reduced = sum.reduce_add[1]()

                    # add the rest of the elements
                    for k in range(k_end, lhs_cols):
                        var lhs_idx = lhs_idx_0 + k * lhs_stride_min_1
                        var rhs_idx = rhs_idx_0 + k * rhs_stride_min_1
                        sum_reduced += lhs_data.load(lhs_idx) * rhs_data.load(
                            rhs_idx
                        )

                    res_data.store(res_idx, sum_reduced)
                else:
                    var sum_real = SIMD[dtype, 2 * nelts[dtype]() // 2](0)
                    var sum_imag = SIMD[dtype, 2 * nelts[dtype]() // 2](0)
                    for k in range(0, k_end, nelts[dtype]()):
                        var lhs_idx = lhs_idx_0 + k * lhs_stride_min_1
                        var rhs_idx = rhs_idx_0 + k * rhs_stride_min_1
                        var lhs = lhs_data.load[width = 2 * nelts[dtype]()](
                            2 * lhs_idx
                        ).deinterleave()
                        var rhs = rhs_data.load[width = 2 * nelts[dtype]()](
                            2 * rhs_idx
                        ).deinterleave()
                        var lhs_real = lhs[0]
                        var lhs_imag = lhs[1]
                        var rhs_real = rhs[0]
                        var rhs_imag = rhs[1]
                        sum_real += lhs_real * rhs_real - lhs_imag * rhs_imag
                        sum_imag += lhs_real * rhs_imag + lhs_imag * rhs_real

                    var sum_real_reduced = sum_real.reduce_add[1]()
                    var sum_imag_reduced = sum_imag.reduce_add[1]()

                    # add the rest of the elements
                    for k in range(k_end, lhs_cols):
                        var lhs_idx = lhs_idx_0 + k * lhs_stride_min_1
                        var rhs_idx = rhs_idx_0 + k * rhs_stride_min_1
                        var lhs_real = lhs_data.load(2 * lhs_idx)
                        var lhs_imag = lhs_data.load(2 * lhs_idx + 1)
                        var rhs_real = rhs_data.load(2 * rhs_idx)
                        var rhs_imag = rhs_data.load(2 * rhs_idx + 1)
                        sum_real_reduced += (
                            lhs_real * rhs_real - lhs_imag * rhs_imag
                        )
                        sum_imag_reduced += (
                            lhs_real * rhs_imag + lhs_imag * rhs_real
                        )

                    res_data.store(2 * res_idx, sum_real_reduced)
                    res_data.store(2 * res_idx + 1, sum_imag_reduced)

        # parallelize[matmul_par](res_rows, res_rows)

    _ = res_stride
    _ = lhs_stride
    _ = rhs_stride
    _ = lhs_rank
    _ = rhs_rank
    _ = rhs_stride_min_1
    _ = res_stride_min_1
    _ = lhs_stride_min_1
    _ = res_stride_min_2
    _ = lhs_stride_min_2
    _ = rhs_stride_min_2
    _ = arg0
    _ = arg1


fn matmul_vjp(
    primals: List[Array], grad: Array, out: Array
) raises -> List[Array]:
    """
    Compute vector-Jacobian product for batched matrix multiplication.

    Args:
        primals: Primal input arrays.
        grad: Gradient of the output with respect to some scalar function.
        out: The output of the forward pass.

    Returns:
        List[Array]: Gradients with respect to each input.

    #### Note:
    Implements reverse-mode automatic differentiation for batched matrix multiplication.
    Returns arrays with shape zero for inputs that do not require gradients.

    #### See Also:
    fwd: Forward-mode autodiff for batched matrix multiplication.
    """
    var lhs_grad = grad @ primals[1].T() if primals[
        0
    ].requires_grad() else Array(0)
    var rhs_grad = primals[0].T() @ grad if primals[
        1
    ].requires_grad() else Array(0)
    return List(lhs_grad, rhs_grad)


fn matmul_jvp(primals: List[Array], tangents: List[Array]) raises -> Array:
    """
    Compute Jacobian-vector product for batched matrix multiplication.

    Args:
        primals: Primal input arrays.
        tangents: Tangent vectors.

    Returns:
        Array: Jacobian-vector product.

    #### Note:
    Implements forward-mode automatic differentiation for batched matrix multiplication.
    The result represents how the output changes with respect to
    infinitesimal changes in the inputs along the directions specified by the tangents.

    #### See Also:
    vjp: Reverse-mode autodiff for batched matrix multiplication.
    """
    return tangents[0] @ primals[1].T() + primals[0].T() @ tangents[1]


fn matmul(arg0: Array, arg1: Array) raises -> Array:
    """
    Perform batched matrix multiplication between two arrays.

    Args:
        arg0: The first input array.
        arg1: The second input array.

    Returns:
        The result of the batched matrix multiplication.

    #### Examples:
    ```python
     a = Array([[1, 2], [3, 4]])
     b = Array([[5, 6], [7, 8]])
     result = matmul(a, b)
     print(result)
    ```

    #### Note:
    The shapes of the two arrays must be compatible for matrix multiplication, i.e. the last dimension of the first array must be equal to the second last dimension of the second array.
    """
    var arr_shape = setup_array_shape(
        List(arg0.array_shape(), arg1.array_shape()),
        "matmul_shape",
        matmul_shape,
    )

    if not arg0.has_fxgraph() and not arg1.has_fxgraph():
        compute_shape(arr_shape, arg0.requires_grad() or arg1.requires_grad())

    var args = List(
        expand(arg0, arr_shape, List(-2, -1)),
        expand(arg1, arr_shape, List(-2, -1)),
    )
    return op_array(
        arr_shape, args, NA, "matmul", matmul_fwd, matmul_jvp, matmul_vjp
    )
