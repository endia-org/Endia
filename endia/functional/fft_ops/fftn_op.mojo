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


from endia.utils.aliases import dtype, nelts, NA
from endia.functional.unary_ops._utils import DifferentiableUnaryOp
from .utils import DifferentiableFftOp, fft_op_array
from .fft_cooley_tukey import (
    fft_cooley_tukey_parallel_inplace,
    get_dims_from_encoded_params,
    get_norm_from_encoded_params,
)


struct FFTN(DifferentiableFftOp):
    @staticmethod
    fn fwd(
        arg0: Array,
        dims: List[Int],
        norm: String,
    ) raises -> Array:
        """Sets up the Array object for the FFT operation."""
        return fft_op_array(
            arg0, "fftn", FFTN.__call__, FFTN.jvp, FFTN.vjp, dims, norm
        )

    @staticmethod
    fn jvp(primals: List[Array], tangents: List[Array]) raises -> Array:
        """Computes the Jacobian-vector product for the FFT function."""
        return default_jvp(primals, tangents)

    @staticmethod
    fn vjp(primals: List[Array], grad: Array, out: Array) raises -> List[Array]:
        """Computes the vector-Jacobian product for the FFT function."""
        var params = out.meta_data()
        var dims = get_dims_from_encoded_params(params)
        var norm = get_norm_from_encoded_params(params)
        var res = conj(fftn(conj(grad), dims, norm))
        return res

    @staticmethod
    fn __call__(inout curr: Array, args: List[Array]) raises:
        """Executes the FFT operation inplace."""
        setup_shape_and_data(curr)

        var params = curr.meta_data()
        var dims = get_dims_from_encoded_params(params)
        var norm = get_norm_from_encoded_params(params)

        fft_cooley_tukey_parallel_inplace(args[0], curr, dims, norm)


def fftn(
    x: Array,
    dims: List[Int] = List[Int](),
    norm: String = "backward",
) -> Array:
    """
    Compute the n-dimensional FFT.

    Args:
        x: The input array.
        dims: The dimensions along which to compute the FFT.
        norm: The normalization mode.

    Returns:
        The n-dimensional FFT of the input array.
    """
    return FFTN.fwd(x, dims, norm)


"""
General notes on how to implement FFTN operation using Endia Arrays.

Initialize the input Array as a real valued or complex valued Array. In both cases, the output Array should always be a complex valued Array.
Here is how to initalize a real valued Endia Arra:
1. version: Via explicit String initalization:
    var x = Array("[1.0, 2.0, 3.0, 4.0]")
2. version: Via initlaization function, like array, zeros, ones, randn, randu, etc.
    var x = endia.randn(shape=List(2,3,4))
3. version: Via empty array initalization and manual data assignment:
    var x = endia.Array(shape=List(2,3,4))
    for i in range(x.size()):
        x.store(i, 42)

Here is how to initalize a complex valued Endia Array:
1. version: Via explicit String initalization:
    var x = Array("[1.0 + 1.0i, 2.0 + 2.0i, 3.0 + 3.0i, 4.0 + 4.0i]")
2. version: Via initlaization function, like array, zeros, ones, randn, randu, etc. and  the endia.complex function:
    var x = endia.complex(endia.randn(shape=List(2,3,4)), endia.randn(shape=List(2,3,4)))
3. version: Via empty array initalization and manual data assignment:
    var x = endia.Array(shape=List(2,3,4), is_complex=True)
    for i in range(x.size()):
        x.store(2*i, 42)
        x.store(2*i+1, 42)
    
    Note: Even though there is already a ComplexSIMD type in Mojo, Endia will store a single complex number as two consecutive Float32 numbers in the Array object. The first Float32 number will be the real part and the second Float32 number will be the imaginary part of the complex number.
    That is the reason why the data assignment is done in the above way.

How to work on the Array data directly? And why might this be benificial? 
The Endia Array object is very similar to a Numpy object, i.e you can call getters and setter (slicing operations) on it, however, all these operations come at the cost of potentially initalizing new intermediate Array objects. So in order to avoid this overhead, you can work on the Array data buffer (a UnsafePointer[Scalar[DType.float32]] object) directly. Here is how you can do it:
    var data_ptr = x.data()
    for i in range(x.size()):
        data_ptr[i] = 42

Important Note: The data buffer will be automtically freed as soon as the Mojo compiler decides that the Array object is no longer needed. This is generally convenient, but it can also lead to unexpected behavior if you are not careful. For example, if you store the data buffer in a variable and there are no further direct usages of the Array object, the data buffer will be freed and the variable will point to an invalid memory location. 
To avoid this simply do an empty assignment to the Array object, like this:
    x = Array(...) # create an Array
    data_ptr = x.data()
    ... do something with the data_ptr ...
    _ = x # create an empty assignment to the Array object, so that the data buffer is not freed after the x.data() call.

Suggested interface for implemting a FFT:
def fftn(
    x: Array,
    dims: List[Int] = List[Int](),
    norm: String = "backward",
) -> Array:
    ...

Explanation of the interface:
1. The input Array x can be a real valued or complex valued Array.
2. The dims parameter is a list of integers that specifies the dimensions along which to compute the FFT. 
3. The norm parameter is a string that specifies the normalization mode. The default value is "backward".
4. The return value is the n-dimensional FFT of the input array. The output Array should always be a complex valued Array.


A Note on the data type:
By default Endia uses dtype=DType.float32 as the data type for the Array object. In some opcoming release, this will become a more dynamic parameter.
If you want to change this however, you can alter the value in the endia.utils.aliases module, like this:
    alias dtype = DType.float64 
That's it. Since the FFT is an operation where numerical precision is important, it might be a good idea to either change this parameter here globally or to cast the values to float64 inside the FFT operation (That is what the builtin FFT operaiton of Endia is currently doing).

Testing against established implementations like PyTorch:
import math
import endia as nd
import time
from python import Python
from endia import fftn


def your_custom_fftn_test():
    var depth = 2**2
    var width = 2**4
    var height = 2**6

    # print("\nDepth:", depth, " - Width:", width, " - Height:", height)

    var torch = Python.import_module("torch")

    var shape = List(2, 2, depth, width, height)
    var x = nd.complex(nd.randn(shape), nd.randn(shape))
    var x_torch = nd.utils.to_torch(x)

    var y = your_custom_fftn(x)            # use you custom FFTN operation here, add potential parameters
    var y_torch = torch.fft.fftn(x_torch)  # use the PyTorch FFTN operation here, add potential parameters

    var msg = "fftn"
    if not nd.utils.is_close(y, y_torch, rtol=1e-5): # Set the rtol parameter to a value that is appropriate for the FFT operation
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)

Explanation of the test:
1. The test function generates a random complex valued Array with some shape. Highly recommended for the start: Make this a one dimensional Array first and test hgiher dimensional Arrays later.
The main workhorse behind the FFT is a always an algorithm that works in a single dimension. Making a FFT work on higher dimensional Arrays is a matter of doing some rehsaping tricks and calling the 1D FFT multiple times.
2. We then create a copy of the Array object and convert it to a PyTorch tensor.
3. We then call the custom FFTN operation and the PyTorch FFTN operation on the same input data. Potential parameters can be added to the custom FFTN operation like the dims and norm parameter.
4. We then compare the results of the custom FFTN operation and the PyTorch FFTN operation. The is_close functiion computes a relative difference between the two results and checks if it is smaller than a certain threshold. The threshold is defined by the rtol parameter. The smaller the rtol parameter, the more accurate the comparison will be.


Benchmarking against established implementations like PyTorch:
import endia as nd
import time
import os


fn benchmark_your_custom_fft(
    shapes: List[Int],
    base_num_iterations: Int = 100,
    warm_up_iterations: Int = 10,
) raises:
    var avg_times = List[Float64]()

    for shape in shapes:
        var array = endia.complex(nd.randn(shape[]), nd.randn(shape[]))

        # Adjust the number of iterations based on the input size
        var num_iterations = max(base_num_iterations, 1000 // shape[])

        # Warm-up phase
        for _ in range(warm_up_iterations):
            _ = your_custom_fft(array)

        var start_time = time.perf_counter()
        for _ in range(num_iterations):
            _ = your_custom_fft(array)
        var end_time = time.perf_counter()

        var avg_time = (end_time - start_time) / num_iterations
        avg_times.append(avg_time)

    with open("your_custom_fft_benchmarks.csv", "w") as txtfile:
        for i in range(len(shapes)):
            txtfile.write(
                String('"({},)",{}\n').format(shapes[i], avg_times[i])
            )


def main():
    shapes_to_benchmark = List(2**1, 2**2, 2**3, 2**4, 2**5, 2**6,) # change this to ndimensional shapes if you wish
    benchmark_your_custom_fft(shapes_to_benchmark)

Explanation of the benchmark:
The above code will measure the average time it takes to compute the FFT of a random complex valued Array for different shapes. The benchmark will be run multiple times and the average time will be written to a file called your_custom_fft_benchmarks.csv. The file will contain the shape of the Array and the average time it took to compute the FFT for that shape.

You can create a similar output for the PyTorch FFT operation and compare the results in a plot. This will give you a good indication of how well your custom FFT operation performs.

Here is hwo you can set this up in Python with PyTorch:
import torch
import time
import csv


def benchmark_pytorch_fft(
    shapes, base_num_iterations=100, warm_up_iterations=10
):
    results = []

    for shape in shapes:
        tensor = torch.randn(
            shape, dtype=torch.complex64
        )  # Complex input for FFT

        if len(shape) == 1:
            fft_func = torch.fft.fft
        elif len(shape) == 2:
            fft_func = torch.fft.fft2
        elif len(shape) == 3:
            fft_func = torch.fft.fftn
        else:
            raise ValueError("Unsupported tensor dimension")

        # Adjust the number of iterations based on the input size
        num_iterations = max(base_num_iterations, 1000 // sum(shape))

        # Warm-up phase
        for _ in range(warm_up_iterations):
            fft_func(tensor)

        start_time = time.perf_counter()
        for _ in range(num_iterations):
            fft_func(tensor)
        end_time = time.perf_counter()

        avg_time = (end_time - start_time) / num_iterations
        results.append((shape, avg_time))

    with open("pytorch_fft_benchmarks.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Shape", "Average Time (s)"])
        writer.writerows(results)


shapes_to_benchmark = [(2**1,), (2**2,), (2**3,), (2**4,), (2**5,), (2**6,), ...] # Add more shapes here

benchmark_pytorch_fft(shapes_to_benchmark)


Plot the results with Python and Matplotlib:
import matplotlib.pyplot as plt
import csv


def plot_benchmark_results(
    filename="pytorch_fft_benchmarks.csv",
    endia_filename="endia_fft_benchmarks.csv",
    numpy_filename="numpy_fft_benchmarks.csv",
):
    shapes = []
    times = []
    endia_shapes = []
    endia_times = []
    numpy_shapes = []
    numpy_times = []  # New list to store NumPy data

    with open(filename, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        for row in reader:
            shapes.append(eval(row[0]))
            times.append(float(row[1]))

    with open(endia_filename, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        for row in reader:
            endia_shapes.append(eval(row[0]))
            endia_times.append(float(row[1]))

    with open(numpy_filename, "r") as csvfile:  # Read NumPy data
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        for row in reader:
            numpy_shapes.append(eval(row[0]))
            numpy_times.append(float(row[1]))

    plt.figure(figsize=(10, 6))
    plt.xscale("log")
    plt.yscale("log")

    for dim in [1, 2, 3]:
        dim_shapes = [s for s in shapes if len(s) == dim]
        dim_times = [t for i, t in enumerate(times) if len(shapes[i]) == dim]
        dim_sizes = [sum(s) for s in dim_shapes]

        endia_dim_shapes = [s for s in endia_shapes if len(s) == dim]
        endia_dim_times = [
            t for i, t in enumerate(endia_times) if len(endia_shapes[i]) == dim
        ]
        endia_dim_sizes = [sum(s) for s in endia_dim_shapes]

        numpy_dim_shapes = [s for s in numpy_shapes if len(s) == dim]
        numpy_dim_times = [
            t for i, t in enumerate(numpy_times) if len(numpy_shapes[i]) == dim
        ]
        numpy_dim_sizes = [sum(s) for s in numpy_dim_shapes]

        if not dim_sizes or not dim_times:
            print(
                f"Skipping plotting for {dim}D PyTorch FFT due to insufficient"
                " data."
            )
            continue

        if not endia_dim_sizes or not endia_dim_times:
            print(
                f"Skipping plotting for {dim}D Endia FFT due to insufficient"
                " data."
            )
            continue

        if not numpy_dim_sizes or not numpy_dim_times:
            print(
                f"Skipping plotting for {dim}D NumPy FFT due to insufficient"
                " data."
            )
            continue

        sorted_data = sorted(zip(dim_sizes, dim_times))
        dim_sizes, dim_times = zip(*sorted_data)

        sorted_endia_data = sorted(zip(endia_dim_sizes, endia_dim_times))
        endia_dim_sizes, endia_dim_times = zip(*sorted_endia_data)

        sorted_numpy_data = sorted(zip(numpy_dim_sizes, numpy_dim_times))
        numpy_dim_sizes, numpy_dim_times = zip(*sorted_numpy_data)

        plt.plot(endia_dim_sizes, endia_dim_times, label=f"{dim}D Endia FFT")
        plt.plot(
            dim_sizes, dim_times, label=f"{dim}D PyTorch FFT", linestyle="--"
        )
        plt.plot(
            numpy_dim_sizes,
            numpy_dim_times,
            label=f"{dim}D NumPy FFT",
            linestyle=":",
        )  # Add NumPy plot

    powers_of_two = [
        2**i for i in range(1, 23)
    ]  # Assuming input sizes from 2^1 to 2^22
    plt.xticks(powers_of_two, [f"$2^{{{i}}}$" for i in range(1, 23)])
    plt.xlabel("Input Size (log scale)")
    plt.ylabel("Average Time (s)")
    plt.title("PyTorch vs Endia FFT Benchmark Results (Log-Log Scale)")
    plt.legend()
    plt.grid(True)
    plt.show()


plot_benchmark_results()


"""
