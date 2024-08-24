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
