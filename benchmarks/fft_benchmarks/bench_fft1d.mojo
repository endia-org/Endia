import endia as nd


def fft1d_benchmark():
    var torch = Python.import_module("torch")

    for n in range(4, 23):
        size = 2**n
        print("Size: 2**", end="")
        print(n, "=", size)
        x = nd.complex(
            nd.unsqueeze(nd.arange(0, size), List(0)),
            nd.unsqueeze(nd.arange(0, size), List(0)),
        )
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
            _ = nd.fft.fft1d(x)
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
