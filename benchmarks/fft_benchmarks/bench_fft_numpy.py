import numpy as np
import time
import csv


def benchmark_numpy_fft(shapes, base_num_iterations=100, warm_up_iterations=10):
    results = []

    for shape in shapes:
        array = np.random.randn(*shape) + 1j * np.random.randn(
            *shape
        )  # Complex input for FFT

        if len(shape) == 1:
            fft_func = np.fft.fft
        elif len(shape) == 2:
            fft_func = np.fft.fft2
        elif len(shape) == 3:
            fft_func = np.fft.fftn
        else:
            raise ValueError("Unsupported array dimension")

        # Adjust the number of iterations based on the input size
        num_iterations = max(
            base_num_iterations, 1000 // np.prod(shape)
        )  # Use np.prod

        # Warm-up phase
        for _ in range(warm_up_iterations):
            fft_func(array)

        start_time = time.perf_counter()
        for _ in range(num_iterations):
            fft_func(array)
        end_time = time.perf_counter()

        avg_time = (end_time - start_time) / num_iterations
        results.append((shape, avg_time))

    with open("numpy_fft_benchmarks.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Shape", "Average Time (s)"])
        writer.writerows(results)


shapes_to_benchmark = [
    (2**1,),
    (2**2,),
    (2**3,),
    (2**4,),
    (2**5,),
    (2**6,),
    (2**7,),
    (2**8,),
    (2**9,),
    (2**10,),
    (2**11,),
    (2**12,),
    (2**13,),
    (2**14,),
    (2**15,),
    (2**16,),
    (2**17,),
    (2**18,),
    (2**19,),
    (2**20,),
    (2**21,),
    (2**22,),
]

benchmark_numpy_fft(shapes_to_benchmark)
