import torch
import math


def fft(x: torch.Tensor) -> torch.Tensor:
    # Convert to complex if the input is real
    if not torch.is_complex(x):
        x = torch.complex(x, torch.zeros_like(x))

    n = x.shape[0]
    if n <= 1:
        return x

    print(" ")
    print("  ")
    print("x:", x)

    first_split = x[0::2]
    second_split = x[1::2]

    print("first_split:", first_split)
    print("second_split:", second_split)

    # Split even and odd indices
    even = fft(x[0::2])
    odd = fft(x[1::2])

    # print("even:", even)
    # print("odd:", odd)

    # Compute twiddle factors
    k = torch.arange(0, n // 2, dtype=torch.float32)
    real = torch.cos(k / n)
    imag = torch.sin(k / n)
    twiddle_factors = torch.complex(real, imag)

    # Combine results
    combined = torch.zeros(n, dtype=torch.complex64)
    combined[: n // 2] = even + twiddle_factors * odd
    combined[n // 2 :] = even - twiddle_factors * odd

    print("combined:", combined)

    return combined


# Example usage
def main():
    n = 16  # Must be a power of 2
    t = torch.linspace(0, 1, n)
    signal = torch.sin(
        t
    )  # torch.sin(2 * math.pi * 10 * t) #+ 0.5 * torch.sin(2 * math.pi * 20 * t)

    # Compute FFT
    spectrum = fft(signal)

    print("\nInput signal:", signal)
    print("FFT result:", spectrum)


if __name__ == "__main__":
    main()
