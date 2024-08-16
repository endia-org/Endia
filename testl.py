# import torch
# import math

# def fft1d(x: torch.Tensor) -> torch.Tensor:
#     if not torch.is_complex(x):
#         x = torch.complex(x, torch.zeros_like(x))

#     n = x.shape[0]
#     if n <= 1:
#         return x

#     even = fft1d(x[0::2])
#     odd = fft1d(x[1::2])

#     k = torch.arange(0, n // 2, dtype=torch.float32, device=x.device)
#     real = torch.cos(-2 * torch.pi * k / n)
#     imag = torch.sin(-2 * torch.pi * k / n)
#     twiddle_factors = torch.complex(real, imag)

#     combined = torch.empty(n, dtype=torch.complex64, device=x.device)
#     combined[: n // 2] = even + twiddle_factors * odd
#     combined[n // 2 :] = even - twiddle_factors * odd

#     return combined

# def fft2d(x: torch.Tensor) -> torch.Tensor:
#     rows, cols = x.shape
    
#     if not torch.is_complex(x):
#         x = torch.complex(x, torch.zeros_like(x))
    
#     # Apply 1D FFT to each row
#     x = torch.stack([fft1d(row) for row in x])
    
#     # Apply 1D FFT to each column
#     x = torch.stack([fft1d(col) for col in x.t()]).t()
    
#     # Removed normalization
#     return x

# def fft3d(x: torch.Tensor) -> torch.Tensor:
#     # Apply 2D FFT to each channel
#     x = torch.stack([fft2d(channel) for channel in x])
    
#     # Apply 1D FFT to each channel
#     x = torch.stack([fft2d(channel) for channel in x])
    
#     # Removed normalization
#     return x

# import numpy as np


# # Assuming fft2d and fft1d implementations are defined above

# def main():

#     n = 4  # Example value for n
#     t = torch.zeros(n ** 3)
#     for i in range(t.size(0)):
#         t[i] = i / (t.size(0) - 1)
#     x = t.reshape(n, n, n)
#     y = t.reshape(n, n, n)
#     z = t.reshape(n, n, n)
    
#     # Create a 2D signal
#     signal = torch.sin(2 * torch.pi * 10 * x) + 0.5 * torch.sin(2 * torch.pi * 20 * y) + torch.sin(2 * torch.pi * 30 * z)



#     # Convert signal to NumPy array and compute 2D FFT using NumPy's built-in function
#     signal_np = signal.numpy()
#     spectrum_np = np.fft.fft(signal_np)

#     # Convert NumPy's result back to PyTorch tensor for easy comparison
#     spectrum_np_torch = torch.from_numpy(spectrum_np)

#     # print("\nInput signal shape:", signal.shape)
#     # print("Input signal:")
#     # print(signal)
#     # print("\nOur FFT result:")
#     # print(spectrum_our)
#     print("\nNumPy FFT result:")
#     print(spectrum_np_torch)
#     print("\nDifference:")
#     # print(torch.abs(spectrum_our - spectrum_np_torch).max())

# if __name__ == "__main__":
#     main()




import torch
import torch.fft

# Constant for Pi
pi = 3.14159265358979323846264

# Function for 1D FFT using recursion
def fft1d(x: torch.Tensor) -> torch.Tensor:
    # Convert to complex if the input is real
    if not torch.is_complex(x):
        x = torch.complex(x, torch.zeros_like(x))

    n = x.shape[0]
    if n <= 1:
        return x
    
    print("\n       ",x)

    # Split even and odd indices
    even = fft1d(x[0::2])
    odd = fft1d(x[1::2])

    # Compute twiddle factors (use real dtype for arange)
    k = torch.arange(0, n // 2, dtype=torch.float32, device=x.device)
    twiddle_factors = torch.exp(-2j * pi * k / n)

    # Combine results
    combined = torch.zeros_like(x)
    combined[:n // 2] = even + twiddle_factors * odd
    combined[n // 2:] = even - twiddle_factors * odd

    return combined

# Function for 2D FFT
def fft2d(x: torch.Tensor) -> torch.Tensor:
    # Apply 1D FFT to rows
    rows, cols = x.shape

    print("\n   ",x)

    if not torch.is_complex(x):
        x = torch.complex(x, torch.zeros_like(x))

    for i in range(rows):
        x[i, :] = fft1d(x[i, :])

    for j in range(cols):
        x[:, j] = fft1d(x[:, j])

    return x

# Function for 3D FFT
def fft3d(x: torch.Tensor) -> torch.Tensor:
    # Apply 1D FFT to each dimension
    rows, cols, depth = x.shape

    print("\n",x)

    if not torch.is_complex(x):
        x = torch.complex(x, torch.zeros_like(x))

    for i in range(rows):
        x[i, :, :] = fft2d(x[i, :, :])

    for j in range(cols):
        x[:, j, :] = fft2d(x[:, j, :])

    for k in range(depth):
        x[:, :, k] = fft2d(x[:, :, k])

    return x

# Main function
def main():
    n = 2
    t = torch.linspace(0, 1, n**3)
    x = t.view(n, n, n)
    y = t.view(n, n, n)
    z = t.view(n, n, n)

    signal = torch.sin(2 * pi * 10 * x) + 0.5 * torch.sin(2 * pi * 20 * y) + torch.sin(2 * pi * 30 * z)

    spectrum = fft3d(signal)

    print("\nInput signal:", signal)
    print("3D FFT result:", spectrum)

    # a = torch.arange(0, 16).reshape(4, 4)
    # print(a)

    # b = a[0,:]
    # print(b)

if __name__ == "__main__":
    main()
