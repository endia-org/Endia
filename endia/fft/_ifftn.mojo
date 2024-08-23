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

from endia import Array, permute, swapaxes
from endia.utils import compute_stride
from endia.functional._utils import is_contiguous, contiguous
from .utils import *
import math


def ifftn(
    input: Array,
    dims: List[Int] = List[Int](),
    norm: String = "backward",
    out: Optional[Array] = None,
) -> Array:
    """Compute the n-dimensional inverse FFT.

    Args:
        input: The input array.
        dims: The dimensions along which to compute the FFT.
        norm: The normalization mode.
        out: The output array (optional).

    Returns:
        The n-dimensional FFT of the input array.
    """
    var x: Array
    if not input.is_complex():
        x = complex(input, zeros_like(input))
    else:
        x = contiguous(input)

    if norm == "backward":
        x = x
    elif norm == "forward":
        x = x / x.size()
    elif "ortho":
        x = x / math.sqrt(x.size())
    else:
        raise "fftn: Invalid norm"

    # setup params
    var size = x.size()
    var shape = x.shape()
    var ndim = x.ndim()
    var axes = List[Int]()
    for i in range(ndim):
        axes.append(i)
    var fft_dims = dims if dims.size > 0 else axes
    if dims.size > 0:
        for i in range(fft_dims.size):
            if fft_dims[i] < 0:
                fft_dims[i] = ndim + fft_dims[i]
            if fft_dims[i] < 0 or fft_dims[i] >= ndim:
                raise "Invalid dimension"

    var res_data = UnsafePointer[Scalar[DType.float64]].alloc(size * 2)
    var data = UnsafePointer[Scalar[DType.float64]].alloc(size * 2)
    copy_complex_and_cast(data, x.data(), size, True)

    for dim in fft_dims:
        var divisions = size // shape[dim[]]
        var parallelize_threshold = 2**14
        var num_workers = num_physical_cores() if size >= parallelize_threshold else 1
        var workload = get_workload(size, divisions, num_workers)
        var h = (
            int(math.log2(Float32(size // workload)))
        ) if divisions == 1 else 0
        var number_subtasks = num_workers if divisions == 1 else divisions

        if dim[] != ndim - 1:
            x.stride_(list_swap(x.stride(), dim[], ndim - 1))
            x.shape_(list_swap(x.shape(), dim[], ndim - 1))

        if not is_contiguous(x.array_shape()):
            execute_copy_raw(data, res_data, x.array_shape(), x.is_complex())
        else:
            var tmp = data
            data = res_data
            res_data = tmp

        # Split the data into individual subarrays to perform #workload indipendent FFTs
        if h > 0:
            cooley_tukey_split(size, h, res_data)

        # Perform the Cooley-Tukey FFT on the subarrays in parallel
        var reordered_arr_data = UnsafePointer[Scalar[DType.uint32]].alloc(
            workload
        )
        bit_reversal(workload, reordered_arr_data)

        @parameter
        fn perform_cooley_tukey_sequencial(i: Int) capturing:
            cooley_tukey_with_bit_reversal(
                workload, res_data.offset(2 * i * workload), reordered_arr_data
            )

        parallelize[perform_cooley_tukey_sequencial](
            number_subtasks, num_workers
        )
        _ = workload, divisions
        reordered_arr_data.free()

        # Recombine the solutions of the subarrays
        if h > 0:
            cooley_tukey_recombine(size, h, res_data)

        # Swap the data pointers
        var tmp = data
        data = res_data
        res_data = tmp

        # Swap the strides and shapes
        x.stride_(compute_stride(x.shape()))
        if dim[] != ndim - 1:
            x.stride_(list_swap(x.stride(), dim[], ndim - 1))
            x.shape_(list_swap(x.shape(), dim[], ndim - 1))

    res_data.free()

    # Copy the data back to the output array
    var output: Array
    if out:
        output = out.value()
    else:
        output = Array(x.shape(), is_complex=True)
    var normalization_devisor = Float32(1.0)
    for dim in fft_dims:
        normalization_devisor *= shape[dim[]]
    copy_complex_and_cast(
        output.data(), data, size, True, normalization_devisor
    )
    data.free()

    return output
