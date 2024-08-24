import endia as nd
import time

# import csv
from python import Python, PythonObject
import os


fn benchmark_pytorch_fft(
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
            _ = nd.fft(array)

        var start_time = time.perf_counter()
        for _ in range(num_iterations):
            _ = nd.fft(array)
        var end_time = time.perf_counter()

        var avg_time = (end_time - start_time) / num_iterations
        avg_times.append(avg_time)

    with open("endia_fft_benchmarks.csv", "w") as txtfile:
        for i in range(len(shapes)):
            txtfile.write(
                String('"({},)",{}\n').format(shapes[i], avg_times[i])
            )


def main():
    shapes_to_benchmark = List(
        2**1,
        2**2,
        2**3,
        2**4,
        2**5,
        2**6,
        2**7,
        2**8,
        2**9,
        2**10,
        2**11,
        2**12,
        2**13,
        2**14,
        2**15,
        2**16,
        2**17,
        2**18,
        2**19,
        2**20,
        2**21,
        2**22,
    )

    benchmark_pytorch_fft(shapes_to_benchmark)
