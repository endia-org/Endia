# Implementing, Testing and Benchmarking a Custom FFT Operation with [Endia Arrays](https://endia.vercel.app/docs/array)

We'll explore how to create your own Fast Fourier Transform (FFT) implementation using Endia Arrays. We'll also cover methods for testing and benchmarking your implementation against established solutions like PyTorch, NumPy and Endia itself.

> **🧪 Experimental:** The `close_to` function, which calculates the relative difference between an Endia Array and a PyTorch Tensor, has been updated and fixed in the current nightly version of Endia. Additionally, a lot of new Array methods and also the entire FFT module itself is currently only available in Endia's nightly branch. To access these features, you need the [Mojo nightly build](https://docs.modular.com/max/install).

## Working with Endia Arrays

Endia Arrays provide a convenient way to set up data for your FFT implementation in Mojo and facilitate testing against other frameworks. Let's start by examining how to initialize these arrays.

### Array Initialization 🚀

You can create both real-valued and complex-valued Endia Arrays. For FFT operations, while the input can be either real or complex, the output should always be complex-valued.

#### Real-valued Endia Array Initialization:

```python
# 1. Via explicit String initialization:
var x = Array("[1.0, 2.0, 3.0, 4.0]")

# 2. Via initialization function:
var x = endia.randn(shape=List(2,3,4))

# 3. Via empty array initialization and manual data assignment on the contiguous memory buffer.
var x = endia.Array(shape=List(2,3,4))
for i in range(x.size()):
    x.store(i, 42)
```

#### Complex-valued Endia Array Initialization:

```python
# 1. Via explicit String initialization:
var x = Array("[1.0 + 1.0i, 2.0 + 2.0i, 3.0 + 3.0i, 4.0 + 4.0i]")

# 2. Via initialization function and the endia.complex function:
var x = endia.complex(endia.randn(shape=List(2,3,4)), endia.randn(shape=List(2,3,4)))

# 3. Via empty array initialization and manual data assignment on the contiguous memory buffer.
var x = endia.Array(shape=List(2,3,4), is_complex=True)
for i in range(x.size()):
    x.store(2*i, 42)
    x.store(2*i+1, 42)
```

**Note:** Endia stores complex numbers as two consecutive floating point numbers in a contiguous memory buffer, with the real part first, followed by the imaginary part. If we have more than one entry in our Array, these real and imaginary parts are therefore alternating.

### Accessing the Data Pointer of an Array 🛠️

For optimal performance, you can work directly with the Array's data buffer. This approach avoids the overhead of creating intermediate Array objects and Array slices.

```python
var data_ptr = x.data() # let's assume x is a complex-valued Array
for i in range(x.size()):
    data_ptr[2*i] = 42
    data_ptr[2*i+1] = 42

# To prevent premature freeing of the data buffer:
_ = x
```

**Important:** Be cautious when storing the data buffer in a variable, as it may be freed when the Array object is no longer needed. To prevent this, use an empty assignment to the Array object as shown above.

### Arrays as special views on Memory 📦

[See the full list of viewing operations here](https://endia.vercel.app/docs/view_ops)

You can access the shape of an Array, its stride, and the offset of the first element in memory. The stride is the number of elements to move in memory to reach the next element along each dimension.

```python
var x = endia.randn(shape=List(2,3,4))

var shape = x.shape()
var stride = x.stride()
var offset = x.storage_offset()
```

You can also create slices on an Array to work with specific parts of the data:

```python
var x = endia.randn(shape=List(2,3,4))

var x_slice = x[0:1, 1:3, :] # Slicing along the first and second dimensions
```

Additionally, you can assign Arrays to slices of an Array:

```python
var x = endia.randn(shape=List(2,3,4))

var y = endia.randn(shape=List(1,2,4))
x[1:2, 1:3, :] = y # Assigning y to a slice of x
```

If you created a slice and want to work on the data as if the slice was the original Array, you can use the `contiguous` function:

```python
var x = endia.randn(shape=List(2,3,4))

var x_slice = x[0:1, 1:3, :]
var x_slice_contiguous = endia.contiguous(x_slice)
```

**Note:** Slicing oeprations create views on the original Array, so modifying the slice will also modify the original Array.

**Important:** Even though Slcing only creates views and does not allocate new memory, it is still a costly operation. Therefore, it is recommended to avoid slicing in performance-critical code.

## Implementing Your Custom FFT 🌟

Here's a suggested interface for your FFT implementation:

```python
def fftn(
    x: Array,
    dims: List[Int] = List[Int](),
    norm: String = "backward",
) -> Array:
    # Your implementation here
    ...
```

This interface allows for flexibility in input type, dimensions for computation, and normalization mode.

Certainly! I'll revise that section to make the second part fit in better. Here's the updated version:

## Testing Your Implementation 🧪

To ensure the accuracy of your custom FFT, it's crucial to test it against established implementations. Here's how you can compare your results with PyTorch directly inside Mojo:

```python
import math
import endia as nd
import time
from python import Python
from endia import fftn

def your_custom_fftn_test():
    var depth = 2**2
    var width = 2**4
    var height = 2**6

    var torch = Python.import_module("torch")

    var shape = List(2, 2, depth, width, height)
    var x = nd.complex(nd.randn(shape), nd.randn(shape))
    var x_torch = nd.utils.to_torch(x)

    var y = your_custom_fftn(x)            # use your custom FFTN operation here
    var y_torch = torch.fft.fftn(x_torch)  # use the PyTorch FFTN operation here

    var msg = "fftn"
    if not nd.utils.is_close(y, y_torch, rtol=1e-5):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)
```

**Note:** The above setup requires the installation of a CPU version of PyTorch. If you prefer not to use PyTorch, you can alternatively compare your custom solutions directly against Endia's built-in FFT implementations. This approach allows you to validate your custom FFT without external dependencies.

This test generates random complex-valued input, applies both your custom FFT and the reference FFT (either PyTorch's or Endia's), and compares the results. It's recommended to start with one-dimensional arrays and progressively move to higher dimensions as you refine your implementation. This incremental approach helps isolate and address any dimension-specific challenges in your FFT algorithm.

## Benchmarking Your FFT ⏱️

Benchmarking helps you assess the performance of your custom FFT against established libraries. Here's how to set up a benchmark in Mojo using your custom FFT implementation:

```python
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
    shapes_to_benchmark = List(2**1, 2**2, 2**3, 2**4, 2**5, 2**6,)
    benchmark_your_custom_fft(shapes_to_benchmark)
```

This benchmark measures the average computation time for various input shapes and saves the results to a CSV file.

### Benchmarking PyTorch's FFT 🔥

For comparison, you'll want to benchmark PyTorch's FFT implementation:

```python
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

shapes_to_benchmark = [(2**1,), (2**2,), (2**3,), (2**4,), (2**5,), (2**6,)]
benchmark_pytorch_fft(shapes_to_benchmark)
```

### Visualizing Benchmark Results 📈

Finally, use this script to create a comparative plot of your custom FFT against PyTorch and NumPy:

```python
import matplotlib.pyplot as plt
import csv

def plot_benchmark_results(
    filename="pytorch_fft_benchmarks.csv",
    endia_filename="your_custom_fft_benchmarks.csv",
    numpy_filename="numpy_fft_benchmarks.csv",
):
    shapes = []
    times = []
    endia_shapes = []
    endia_times = []
    numpy_shapes = []
    numpy_times = []

    # Read data from CSV files
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

    with open(numpy_filename, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        for row in reader:
            numpy_shapes.append(eval(row[0]))
            numpy_times.append(float(row[1]))

    # Set up the plot
    plt.figure(figsize=(10, 6))
    plt.xscale("log")
    plt.yscale("log")

    # Plot data for each dimension
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

        # Plot PyTorch data
        if dim_sizes and dim_times:
            sorted_data = sorted(zip(dim_sizes, dim_times))
            dim_sizes, dim_times = zip(*sorted_data)
            plt.plot(dim_sizes, dim_times, label=f"{dim}D PyTorch FFT", linestyle="--")

        # Plot Endia data
        if endia_dim_sizes and endia_dim_times:
            sorted_endia_data = sorted(zip(endia_dim_sizes, endia_dim_times))
            endia_dim_sizes, endia_dim_times = zip(*sorted_endia_data)
            plt.plot(endia_dim_sizes, endia_dim_times, label=f"{dim}D Endia FFT")

        # Plot NumPy data
        if numpy_dim_sizes and numpy_dim_times:
            sorted_numpy_data = sorted(zip(numpy_dim_sizes, numpy_dim_times))
            numpy_dim_sizes, numpy_dim_times = zip(*sorted_numpy_data)
            plt.plot(numpy_dim_sizes, numpy_dim_times, label=f"{dim}D NumPy FFT", linestyle=":")

    # Customize the plot
    powers_of_two = [2**i for i in range(1, 23)]
    plt.xticks(powers_of_two, [f"$2^{{{i}}}$" for i in range(1, 23)])
    plt.xlabel("Input Size (log scale)")
    plt.ylabel("Average Time (s)")
    plt.title("PyTorch vs Endia vs NumPy FFT Benchmark Results (Log-Log Scale)")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_benchmark_results()
```