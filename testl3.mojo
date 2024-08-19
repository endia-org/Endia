import math
import endia as nd
import time
from python import Python, PythonObject
from collections import Optional
from algorithm import parallelize

alias pi = Float32(3.14159265358979323846264)


fn reverse_bits_simd(
    x: SIMD[DType.uint32, nd.nelts[DType.uint32]()]
) -> SIMD[DType.uint32, nd.nelts[DType.uint32]()]:
    var y = x
    y = ((y >> 1) & 0x55555555) | ((y & 0x55555555) << 1)
    y = ((y >> 2) & 0x33333333) | ((y & 0x33333333) << 2)
    y = ((y >> 4) & 0x0F0F0F0F) | ((y & 0x0F0F0F0F) << 4)
    y = ((y >> 8) & 0x00FF00FF) | ((y & 0x00FF00FF) << 8)
    return (y >> 16) | (y << 16)


fn bit_reversal(
    n: Int, reordered_arr_data: UnsafePointer[Scalar[DType.uint32]]
):
    """
    Generate a bit reversal permutation for integers from 0 to n-1.
    Works for any positive integer n.
    """
    var log2_n = int(math.ceil(math.log2(Float32(n))))
    var n_pow2 = 1 << log2_n
    alias u = SIMD[DType.uint32, nd.nelts[DType.uint32]()](
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    )

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


def fft_sequential(x: nd.Array) -> nd.Array:
    if x.ndim() != 1:
        raise "Input must be 1-dimensional"

    if not x.is_complex():
        x = nd.complex(x, nd.zeros_like(x))

    n = x.size()  # n is the number of complex elements in x

    var data = x.data()  # data has actual capacity 2*n since each complex element has 2 scalars

    # Bit-reversal permutation
    var reordered_arr_data = UnsafePointer[Scalar[DType.uint32]].alloc(n)
    bit_reversal(n, reordered_arr_data)

    # permute x according to the bit-reversal permutation
    for i in range(n):
        var j = int(reordered_arr_data.load(i))
        if i < j:
            var tmp = data.load[width=2](2 * i)
            data.store[width=2](2 * i, data.load[width=2](2 * j))
            data.store[width=2](2 * j, tmp)

    reordered_arr_data.free()

    var m = 2
    while m <= n:
        var u = SIMD[DType.float32, 2](1.0, 0.0)
        # Calculate w_real and w_imag for the complex exponential
        var angle = -2 * pi / m
        var w_real = math.cos(angle)
        var w_imag = math.sin(angle)

        for k in range(0, m // 2):
            for j in range(k, n, m):
                j_2 = 2 * j
                z = data.load[width=2](j_2)
                d = data.load[width=2](2 * j + m)
                t = SIMD[DType.float32, 2](
                    u[0] * d[0] - u[1] * d[1], u[0] * d[1] + u[1] * d[0]
                )
                data.store[width=2](j_2 + m, z - t)
                data.store[width=2](j_2, z + t)

            # Update u for the next iteration
            u = SIMD[DType.float32, 2](
                u[0] * w_real - u[1] * w_imag, u[0] * w_imag + u[1] * w_real
            )

        m *= 2

    return x


def fft(x: nd.Array) -> nd.Array:
    if x.size() < 2**16:
        return fft_sequential(x)

    if not x.is_complex():
        x = nd.complex(x, nd.zeros_like(x))

    n = x.shape()[0]
    if n <= 1:
        return x

    # First level of recursion (even/odd split)
    var even_x = x[::2]
    var odd_x = x[1::2]

    # Second level of recursion (split even_x and odd_x into their even/odd)
    var even_even = nd.contiguous(even_x[::2])
    var even_odd = nd.contiguous(even_x[1::2])
    var odd_even = nd.contiguous(odd_x[::2])
    var odd_odd = nd.contiguous(odd_x[1::2])

    var ins = List[nd.Array](even_even, even_odd, odd_even, odd_odd)
    var sols = ins

    # Bit-reversal permutation
    var reordered_arr_data = UnsafePointer[Scalar[DType.uint32]].alloc(n // 4)
    bit_reversal(n // 4, reordered_arr_data)

    @parameter
    fn do_work(i: Int) capturing:
        var y = ins[i]
        var n = y.node[].shape[].size
        var data = y.data()  # assume its contiguous memory !

        # permute x according to the bit-reversal permutation
        for i in range(n):
            var j = int(reordered_arr_data.load(i))
            if i < j:
                var tmp = data.load[width=2](2 * i)
                data.store[width=2](2 * i, data.load[width=2](2 * j))
                data.store[width=2](2 * j, tmp)

        var m = 2
        while m <= n:
            var u = SIMD[DType.float32, 2](1.0, 0.0)
            # Calculate w_real and w_imag for the complex exponential
            var angle = -2 * pi / m
            var w_real = math.cos(angle)
            var w_imag = math.sin(angle)

            for k in range(0, m // 2):
                for j in range(k, n, m):
                    var j_2 = 2 * j
                    var j_2_plus_m = j_2 + m

                    var z = data.load[width=2](j_2)
                    var d = data.load[width=2](j_2_plus_m)
                    var t = SIMD[DType.float32, 2](
                        u[0] * d[0] - u[1] * d[1], u[0] * d[1] + u[1] * d[0]
                    )
                    data.store[width=2](j_2_plus_m, z - t)
                    data.store[width=2](j_2, z + t)

                # Update u for the next iteration
                u = SIMD[DType.float32, 2](
                    u[0] * w_real - u[1] * w_imag, u[0] * w_imag + u[1] * w_real
                )

            m *= 2

        sols[i] = y

    parallelize[do_work](4, 4)

    _ = sols
    _ = ins
    reordered_arr_data.free()

    var even_even_res = sols[0]
    var even_odd_res = sols[1]
    var odd_even_res = sols[2]
    var odd_odd_res = sols[3]

    var n2 = n // 2
    var n4 = n // 4

    var k = nd.arange(0, n4)
    var twiddle_factors_n4 = nd.complex(
        nd.cos(-2 * pi * k / n2), nd.sin(-2 * pi * k / n2)
    )

    var T0 = twiddle_factors_n4 * odd_even_res
    var T1 = twiddle_factors_n4 * odd_odd_res
    #
    var even = nd.Array(List(n2), is_complex=True)
    even[:n4] = even_even_res + T0
    even[n4:] = even_even_res - T0
    var odd = nd.Array(List(n2), is_complex=True)
    odd[:n4] = even_odd_res + T1
    odd[n4:] = even_odd_res - T1

    k = nd.arange(0, n2)
    var twiddle_factors = nd.complex(
        nd.cos(-2 * pi * k / n), nd.sin(-2 * pi * k / n)
    )
    var T = twiddle_factors * odd

    var result = nd.Array(List(n), is_complex=True)
    result[:n2] = even + T
    result[n2:] = even - T

    return result


from time import now


def main():
    # power of two
    var n = 2**20
    print(n)
    var torch = Python.import_module("torch")

    var x = nd.complex(nd.arange(0, n), nd.arange(0, n))
    # print("Input:", x)
    start = now()
    var y = fft(x)
    end = now()
    print(y)
    print("Time taken:", (end - start) / 1000000000)
    print()

    var x_torch = torch.complex(
        torch.arange(0, n).float(), torch.arange(0, n).float()
    )
    # print("Input Torch:", x_torch)
    start = now()
    var y_torch = torch.fft.fft(x_torch)
    print("Time taken Torch:", (now() - start) / 1000000000)
    # print("Output Torch:")
    # print(y_torch)

    real_torch = y_torch.real
    imag_torch = y_torch.imag
    var diff = Float32(0)
    var epsilon = Float32(1e-6)  # Small value to avoid division by zero
    for i in range(n):
        real = y.data().load(2 * i)
        imag = y.data().load(2 * i + 1)
        var real_torch_val = real_torch[i].to_float64().cast[DType.float32]()
        var imag_torch_val = imag_torch[i].to_float64().cast[DType.float32]()
        var magnitude = max(
            math.sqrt(real_torch_val**2 + imag_torch_val**2), epsilon
        )
        diff += (
            abs(real - real_torch_val) + abs(imag - imag_torch_val)
        ) / magnitude

    diff /= n
    print("Mean relative difference:", diff)
