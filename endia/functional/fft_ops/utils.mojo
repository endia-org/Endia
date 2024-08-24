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

import endia as nd
from endia.utils import compute_stride
from endia.functional._utils import is_contiguous, contiguous
import math
from algorithm import parallelize, vectorize
from memory import memcpy
from sys import num_physical_cores
from endia.utils import copy_shape, NA, list_to_array_shape, array_shape_to_list

alias pi = Float64(3.141592653589793)  # Maximum useful precision for Float64


fn reverse_bits_simd(
    x: SIMD[DType.uint32, nd.nelts[DType.uint32]()]
) -> SIMD[DType.uint32, nd.nelts[DType.uint32]()]:
    """
    Reverse the bits of a 32-bit integer.
    """
    var y = x
    y = ((y >> 1) & 0x55555555) | ((y & 0x55555555) << 1)
    y = ((y >> 2) & 0x33333333) | ((y & 0x33333333) << 2)
    y = ((y >> 4) & 0x0F0F0F0F) | ((y & 0x0F0F0F0F) << 4)
    y = ((y >> 8) & 0x00FF00FF) | ((y & 0x00FF00FF) << 8)
    return (y >> 16) | (y << 16)


@always_inline
fn bit_reversal(
    n: Int, reordered_arr_data: UnsafePointer[Scalar[DType.uint32]]
):
    """
    Generate a bit reversal permutation for integers from 0 to n-1.
    Works for any positive integer n.
    """
    var log2_n = int(math.ceil(math.log2(Float32(n))))
    var n_pow2 = 1 << log2_n

    var u = SIMD[DType.uint32, nd.nelts[DType.uint32]()]()
    for i in range(nd.nelts[DType.uint32]()):
        u[i] = SIMD[DType.uint32, 1](i)

    for i in range(0, n_pow2, nd.nelts[DType.uint32]()):
        var x = u + i
        var reversed = reverse_bits_simd(x) >> (32 - log2_n)

        # Only store if within original n
        if i < n:
            var to_store = reversed.cast[DType.uint32]()
            if i + 15 < n:
                reordered_arr_data.store[width = nd.nelts[DType.uint32]()](
                    i, to_store
                )
            else:
                for j in range(nd.nelts[DType.uint32]()):
                    if i + j < n:
                        reordered_arr_data.store(i + j, to_store[j])


@always_inline
fn cooley_tukey_split(
    n: Int, max_depth: Int, res_data: UnsafePointer[Scalar[DType.float64]]
):
    """
    Non-recursive Cooley-Tukey FFT splitting of the input data with a limited depth.
    """

    # Non-recursive Cooley-Tukey FFT splitting of the input data
    for iteration in range(max_depth):
        var subarray_size = n // (2**iteration)
        var num_subarrays = 2**iteration
        var temp = UnsafePointer[Scalar[DType.float64]].alloc(subarray_size * 2)

        # Split each subarray into even and odd indices
        for subarray in range(num_subarrays):
            var start = subarray * subarray_size
            var end = start + subarray_size
            var even_index = 0
            var odd_index = subarray_size // 2

            for i in range(start, end):
                if (i - start) % 2 == 0:
                    temp.store[width=2](
                        2 * even_index, res_data.load[width=2](2 * i)
                    )
                    even_index += 1
                else:
                    temp.store[width=2](
                        2 * odd_index, res_data.load[width=2](2 * i)
                    )
                    odd_index += 1

            for i in range(subarray_size):
                res_data.store(2 * (start + i), temp[2 * i])
                res_data.store(2 * (start + i) + 1, temp[2 * i + 1])

        # Free the temporary buffer
        temp.free()


@always_inline
fn cooley_tukey_recombine(
    n: Int, start_depth: Int, res_data: UnsafePointer[Scalar[DType.float64]]
):
    """
    Non-recursive Cooley-Tukey FFT recombination of the subsolutions. The recombination starts at a given depth.
    """
    # Allocate temporary buffers
    var temp = UnsafePointer[Scalar[DType.float64]].alloc(n * 2)
    var even = UnsafePointer[Scalar[DType.float64]].alloc(n)
    var odd = UnsafePointer[Scalar[DType.float64]].alloc(n)
    var T = UnsafePointer[Scalar[DType.float64]].alloc(n)
    var twiddle_factors = UnsafePointer[Scalar[DType.float64]].alloc(n)

    # Non-recursive Cooley-Tukey FFT recombination of the subsolutions
    for iteration in range(start_depth - 1, -1, -1):
        var subarray_size = n // (2**iteration)
        var num_subarrays = 2**iteration

        for k in range(subarray_size // 2):
            var p = (-2 * pi / subarray_size) * k
            twiddle_factors.store[width=2](
                2 * k, SIMD[DType.float64, 2](math.cos(p), math.sin(p))
            )

        for subarray in range(num_subarrays):
            var start = subarray * subarray_size
            var mid = start + subarray_size // 2

            for k in range(subarray_size // 2):
                var k_times_2 = 2 * k
                var k_times_2_plus_1 = k_times_2 + 1
                even.store[width=2](
                    k_times_2, res_data.load[width=2](2 * (start + k))
                )
                odd.store[width=2](
                    k_times_2, res_data.load[width=2](2 * (mid + k))
                )
                var twiddle_factor = twiddle_factors.load[width=2](k_times_2)
                T.store(
                    k_times_2,
                    twiddle_factor[0] * odd[k_times_2]
                    - twiddle_factor[1] * odd[k_times_2_plus_1],
                )
                T.store(
                    k_times_2_plus_1,
                    twiddle_factor[0] * odd[k_times_2_plus_1]
                    + twiddle_factor[1] * odd[k_times_2],
                )
                temp.store[width=2](
                    2 * (start + k),
                    even.load[width=2](k_times_2) + T.load[width=2](k_times_2),
                )
                temp.store[width=2](
                    2 * (mid + k),
                    even.load[width=2](k_times_2) - T.load[width=2](k_times_2),
                )

        memcpy(res_data, temp, n * 2)

    # Free the temporary buffers
    even.free()
    odd.free()
    T.free()
    twiddle_factors.free()
    temp.free()


@always_inline
fn cooley_tukey_with_bit_reversal(
    workload: Int,
    data: UnsafePointer[Scalar[DType.float64]],
    reordered_arr_data: UnsafePointer[Scalar[DType.uint32]],
    dims: List[Int] = List[Int](),
):
    """
    Iterative fast Fourier transform using the Cooley-Tukey algorithm with bit-reversal permutation.
    """
    # permute x according to the bit-reversal permutation
    for i in range(workload):
        var j = int(reordered_arr_data.load(i))
        if i < j:
            var tmp = data.load[width=2](2 * i)
            data.store[width=2](2 * i, data.load[width=2](2 * j))
            data.store[width=2](2 * j, tmp)

    # Cooley-Tukey FFT
    var m = 2
    while m <= workload:
        var u = SIMD[DType.float64, 2](1.0, 0.0)
        var angle = -2 * pi / m
        var w_real = math.cos(angle)
        var w_imag = math.sin(angle)

        for k in range(0, m // 2):
            for j in range(k, workload, m):
                var j_2 = 2 * j
                var j_2_plus_m = j_2 + m

                var z = data.load[width=2](j_2)
                var d = data.load[width=2](j_2_plus_m)
                var t = SIMD[DType.float64, 2](
                    u[0] * d[0] - u[1] * d[1], u[0] * d[1] + u[1] * d[0]
                )
                data.store[width=2](j_2_plus_m, z - t)
                data.store[width=2](j_2, z + t)

            # Update u for the next iteration
            u = SIMD[DType.float64, 2](
                u[0] * w_real - u[1] * w_imag, u[0] * w_imag + u[1] * w_real
            )

        m *= 2


fn copy_complex_and_cast[
    dst_type: DType, src_type: DType
](
    dst: UnsafePointer[Scalar[dst_type]],
    src: UnsafePointer[Scalar[src_type]],
    size: Int,
    conjugate_and_divide: Bool = False,
    divisor: SIMD[dst_type, 1] = 1,
):
    """
    Copy complex data from one buffer to another and cast the data to a different type. Optionally conjugate and divide by a scalar (usefule for inverse FFT).
    """
    if conjugate_and_divide:

        @parameter
        fn do_copy_with_div[simd_width: Int](i: Int):
            dst.store[width = 2 * simd_width](
                2 * i,
                src.load[width = 2 * simd_width](2 * i).cast[dst_type]()
                / divisor,
            )

        vectorize[do_copy_with_div, nelts[dtype]()](size)

        for i in range(size):
            dst.store(2 * i + 1, -dst.load(2 * i + 1))
    else:

        @parameter
        fn do_copy[simd_width: Int](i: Int):
            dst.store[width = 2 * simd_width](
                2 * i, src.load[width = 2 * simd_width](2 * i).cast[dst_type]()
            )

        vectorize[do_copy, nelts[dtype]()](size)


fn get_workload(n: Int, divisions: Int, num_workers: Int) raises -> Int:
    """
    Calculate the workload size for each worker.
    """
    var workload = (n // num_workers) if (divisions == 1) else (n // divisions)
    if (workload & (workload - 1)) != 0:
        raise "Workload size must be a power of two"
    return workload


fn list_swap(arg: List[Int], i: Int, j: Int) raises -> List[Int]:
    if i < 0 or j < 0 or i >= arg.size or j >= arg.size:
        raise "Invalid index"
    var arr = arg
    var tmp = arr[i]
    arr[i] = arr[j]
    arr[j] = tmp
    return arr


fn determine_num_workers(size: Int) raises -> Int:
    """
    Determine the number of workers to use for parallelization.
    """
    if size < 2**14:
        return 1
    elif size < 2**16:
        return 2
    elif size < 2**18:
        return 4
    else:
        return num_physical_cores()


fn cooley_tukey_parallel(
    input: Array,
    dims: List[Int],
    norm: String,
    out: Optional[Array] = None,
    conj_input: Bool = False,
    conj_output: Bool = False,
    input_divisor: Float64 = 1.0,
    output_divisor: SIMD[dtype, 1] = 1.0,
) raises -> Array:
    """Compute the n-dimensional FFT.

    Args:
        input: The input array.
        dims: The dimensions along which to compute the FFT.
        norm: The normalization mode.
        out: The output array (optional).
        conj_input: Whether to conjugate the input data.
        conj_output: Whether to conjugate the output data.
        input_divisor: The divisor for the input data.
        output_divisor: The divisor for the output data.

    Returns:
        The n-dimensional FFT/IFFT of the input array.
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
            if (shape[fft_dims[i]] & (shape[fft_dims[i]] - 1)) != 0:
                raise "Dimension must be a power of two"

    var res_data = UnsafePointer[Scalar[DType.float64]].alloc(size * 2)
    var data = UnsafePointer[Scalar[DType.float64]].alloc(size * 2)
    copy_complex_and_cast(data, x.data(), size, conj_input, input_divisor)

    for dim in fft_dims:
        var divisions = size // shape[dim[]]
        # var parallelize_threshold = 2**14
        var num_workers = determine_num_workers(size)
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

    copy_complex_and_cast(
        output.data(), data, size, conj_output, output_divisor
    )
    data.free()

    return output


fn cooley_tukey_parallel_inplace(
    input: Array,
    inout out: Array,
    dims: List[Int],
    norm: String,
    conj_input: Bool = False,
    conj_output: Bool = False,
    input_divisor: Float64 = 1.0,
    output_divisor: SIMD[dtype, 1] = 1.0,
) raises:
    _ = cooley_tukey_parallel(
        input,
        dims,
        norm,
        out,
        conj_input,
        conj_output,
        input_divisor,
        output_divisor,
    )


trait DifferentiableFftOp:
    @staticmethod
    fn fwd(
        arg0: Array,
        dims: List[Int],
        norm: String,
    ) raises -> Array:
        ...

    @staticmethod
    fn jvp(primals: List[Array], tangents: List[Array]) raises -> Array:
        ...

    @staticmethod
    fn vjp(primals: List[Array], grad: Array, out: Array) raises -> List[Array]:
        ...

    @staticmethod
    fn __call__(inout curr: Array, args: List[Array]) raises:
        ...


fn fft_op_array(
    arg0: Array,
    name: String,
    fwd: fn (inout Array, List[Array]) raises -> None,
    jvp: fn (List[Array], List[Array]) raises -> Array,
    vjp: fn (List[Array], Array, Array) raises -> List[Array],
    dims: List[Int],
    norm: String,
) raises -> Array:
    var arr_shape = setup_array_shape(
        arg0.array_shape(),
        "copy_shape",
        copy_shape,
    )
    # print("inside fft_op_array 2222")
    # var p = encode_fft_params(dims, norm)
    # print("params ehre: ", end="")
    # for i in range(len(p)):
    #     print(p[i], end=", ")
    # print()

    return op_array(
        arr_shape,
        arg0,
        NA,
        name,
        fwd,
        jvp,
        vjp,
        meta_data=encode_fft_params(dims, norm),
        is_complex_p=True,
    )


fn encode_fft_params(
    dims: List[Int],
    norm: String,
) raises -> List[Int]:
    var num_dims = len(dims)
    var params = List[Int](num_dims)
    params.extend(dims)
    if norm == "backward":
        params.append(0)
    elif norm == "forward":
        params.append(1)
    elif norm == "ortho":
        params.append(2)
    else:
        raise "encode: Invalid norm"
    return params


fn get_dims_from_encoded_params(
    params: List[Int],
) raises -> List[Int]:
    var num_dims = params[0]
    var dims = List[Int]()
    for i in range(1, num_dims + 1):
        dims.append(params[i])
    return dims


fn get_norm_from_encoded_params(
    params: List[Int],
) raises -> String:
    var num_dims = params[0]
    if params[num_dims + 1] == 0:
        return "backward"
    elif params[num_dims + 1] == 1:
        return "forward"
    elif params[num_dims + 1] == 2:
        return "ortho"
    else:
        raise "get_norm: Invalid norm"
