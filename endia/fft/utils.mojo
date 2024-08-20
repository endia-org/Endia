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


#####---------------------------------------------------------####
#                    1D FFT Building Blocks
#####---------------------------------------------------------####

import math
import endia as nd
import time
from python import Python, PythonObject
from collections import Optional
from algorithm import parallelize
from memory import memcpy
from time import now
from sys import num_physical_cores

alias pi = Float64(3.141592653589793)  # Maximum useful precision for Float64


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


def fft_c(x: nd.Array) -> nd.Array:
    # q: check if x.size() is a power of two
    if (x.size() & (x.size() - 1)) != 0:
        raise "Input size must be a power of two"

    if not x.is_complex():
        x = nd.complex(x, nd.zeros_like(x))

    n = x.shape()[0]
    if n <= 1:
        return x

    x = nd.contiguous(x.reshape(x.size()))

    var parallelize_threshold = 2**14
    var num_workers = num_physical_cores() if x.size() >= parallelize_threshold else 1
    var workload = n // num_workers
    var h = int(math.log2(Float32(n // workload)))

    # Bit-reversal permutation
    var reordered_arr_data = UnsafePointer[Scalar[DType.uint32]].alloc(workload)
    bit_reversal(n // num_workers, reordered_arr_data)

    var data = x.data()
    var res_data = UnsafePointer[Scalar[DType.float64]].alloc(n * 2)
    for i in range(2 * n):
        res_data.store(i, data.load(i).cast[DType.float64]())

    for iteration in range(h):
        var subarray_size = n // (2**iteration)
        var num_subarrays = 2**iteration
        var temp = UnsafePointer[Scalar[DType.float64]].alloc(subarray_size * 2)

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

        temp.free()

    @parameter
    fn do_work(i: Int) capturing:
        var n = workload

        # tranfer data to float64 for higher precision
        var data = UnsafePointer[Scalar[DType.float64]].alloc(2 * n)

        for j in range(2 * n):
            data.store(j, res_data.load(2 * i * workload + j))

        # permute x according to the bit-reversal permutation
        for i in range(n):
            var j = int(reordered_arr_data.load(i))
            if i < j:
                var tmp = data.load[width=2](2 * i)
                data.store[width=2](2 * i, data.load[width=2](2 * j))
                data.store[width=2](2 * j, tmp)

        var m = 2
        while m <= n:
            var u = SIMD[DType.float64, 2](1.0, 0.0)
            var angle = -2 * pi / m
            var w_real = math.cos(angle)
            var w_imag = math.sin(angle)

            for k in range(0, m // 2):
                for j in range(k, n, m):
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

        memcpy(res_data.offset(2 * i * workload), data, 2 * n)
        data.free()

    parallelize[do_work](num_workers, num_workers)

    _ = workload
    reordered_arr_data.free()

    var temp = UnsafePointer[Scalar[DType.float64]].alloc(n * 2)
    var even = UnsafePointer[Scalar[DType.float64]].alloc(n)
    var odd = UnsafePointer[Scalar[DType.float64]].alloc(n)
    var T = UnsafePointer[Scalar[DType.float64]].alloc(n)
    var twiddle_factors = UnsafePointer[Scalar[DType.float64]].alloc(n)

    for iteration in range(h - 1, -1, -1):
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

    even.free()
    odd.free()
    T.free()
    twiddle_factors.free()
    temp.free()

    var result = nd.Array(List(n), is_complex=True)
    var data_orig = result.data()

    for i in range(2 * n):
        data_orig.store(i, res_data.load(i).cast[DType.float32]())

    res_data.free()

    return result


def fft_benchmark():
    var torch = Python.import_module("torch")

    for n in range(4, 23):
        size = 2**n
        print("Size: 2**", end="")
        print(n, "=", size)
        x = nd.complex(nd.arange(0, size), nd.arange(0, size))
        x_torch = torch.complex(
            torch.arange(0, size).float(), torch.arange(0, size).float()
        )

        num_iterations = 20
        warmup = 5
        total = Float32(0)
        total_torch = Float32(0)

        for iteration in range(num_iterations + warmup):
            if iteration < warmup:
                total = 0
                total_torch = 0

            start = now()
            _ = fft_c(x)
            total += now() - start

            start = now()
            _ = torch.fft.fft(x_torch)
            total_torch += now() - start

        my_time = total / (1000000000 * num_iterations)
        torch_time = total_torch / (1000000000 * num_iterations)
        print("Time taken:", my_time)
        print("Time taken Torch:", torch_time)
        print("Difference:", (torch_time - my_time) / torch_time * 100, "%")
        print()


def fft_test():
    var n = 2**20  # power of two
    print("Input Size: ", n)
    var torch = Python.import_module("torch")

    var x = nd.complex(nd.randn(n), nd.randn(n))
    var x_torch = nd.utils.to_torch(x)

    var y = fft_c(x)
    var y_torch = torch.fft.fft(x_torch)
    real_torch = y_torch.real
    imag_torch = y_torch.imag

    var diff = Float32(0)
    var epsilon = Float32(1e-8)  # Small value to avoid division by zero

    var data = y.data()
    for i in range(n):
        real = data.load(2 * i)
        imag = data.load(2 * i + 1)
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


#####---------------------------------------------------------####
#              BENCHMARK RESULLTS so far (on Apple M3)
#####---------------------------------------------------------####
#
# Size: 2**4 = 16
# Time taken: 2.0000000233721948e-07
# Time taken Torch: 5.5000000429572538e-06
# Gain: 96.363632202148438 %

# Size: 2**5 = 32
# Time taken: 2.0000000233721948e-07
# Time taken Torch: 5.8000000535685103e-06
# Gain: 96.551719665527344 %

# Size: 2**6 = 64
# Time taken: 4.9999999873762135e-07
# Time taken Torch: 5.7000002016138751e-06
# Gain: 91.228065490722656 %

# Size: 2**7 = 128
# Time taken: 9.9999999747524271e-07
# Time taken Torch: 6.0500001382024493e-06
# Gain: 83.471076965332031 %

# Size: 2**8 = 256
# Time taken: 1.6999999843392288e-06
# Time taken Torch: 6.6500001594249625e-06
# Gain: 74.43609619140625 %

# Size: 2**9 = 512
# Time taken: 3.5999998999614036e-06
# Time taken Torch: 7.7499998951680027e-06
# Gain: 53.548385620117188 %

# Size: 2**10 = 1024
# Time taken: 7.2500001806474756e-06
# Time taken Torch: 1.0149999980058055e-05
# Gain: 28.571426391601562 %

# Size: 2**11 = 2048
# Time taken: 1.5100000382517464e-05
# Time taken Torch: 1.5149999853747431e-05
# Gain: 0.33002951741218567 %

# Size: 2**12 = 4096
# Time taken: 3.3600001188460737e-05
# Time taken Torch: 3.1150000722846016e-05
# Gain: -7.8651695251464844 %

# Size: 2**13 = 8192
# Time taken: 7.1900001785252243e-05
# Time taken Torch: 7.2449998697265983e-05
# Gain: 0.75913995504379272 %

# Size: 2**14 = 16384
# Time taken: 0.00027930000214837492
# Time taken Torch: 0.00019720000273082405
# Gain: -41.632858276367188 %

# Size: 2**15 = 32768
# Time taken: 0.000534099992364645
# Time taken Torch: 0.00036584999179467559
# Gain: -45.988796234130859 %

# Size: 2**16 = 65536
# Time taken: 0.0012974500423297286
# Time taken Torch: 0.00075030000880360603
# Gain: -72.924163818359375 %

# Size: 2**17 = 131072
# Time taken: 0.0034556998871266842
# Time taken Torch: 0.0021065499167889357
# Gain: -64.045478820800781 %

# Size: 2**18 = 262144
# Time taken: 0.0071354503743350506
# Time taken Torch: 0.0040787500329315662
# Gain: -74.942085266113281 %

# Size: 2**19 = 524288
# Time taken: 0.014604948461055756
# Time taken Torch: 0.0083753997460007668
# Gain: -74.379119873046875 %

# Size: 2**20 = 1048576
# Time taken: 0.03164985403418541
# Time taken Torch: 0.020133750513195992
# Gain: -57.198005676269531 %

# Size: 2**21 = 2097152
# Time taken: 0.063971899449825287
# Time taken Torch: 0.035383101552724838
# Gain: -80.797889709472656 %

# Size: 2**22 = 4194304
# Time taken: 0.16245074570178986
# Time taken Torch: 0.074759058654308319
# Gain: -117.29907989501953 %
