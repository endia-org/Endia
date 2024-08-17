import numpy as np
import time


import numpy as np

def fft1d(x):
    n = x.shape[0]
    x = x.astype(complex)

    # # Bit-reversal permutation
    # j = 0
    # for i in range(1, n):
    #     bit = n >> 1
    #     while j >= bit:
    #         j -= bit
    #         bit >>= 1
    #     j += bit
    #     if i < j:
    #         x[i], x[j] = x[j], x[i]

    # # Cooley-Tukey FFT
    # length = 2
    # while length <= n:
    #     half_length = length >> 1
    #     factor = np.exp(-2j * np.pi * np.arange(half_length) / length)
    #     for i in range(0, n, length):
    #         for k in range(half_length):
    #             t = factor[k] * x[i + k + half_length]
    #             x[i + k + half_length] = x[i + k] - t
    #             x[i + k] += t
    #     length <<= 1

    return x


def fft2d(x):
    n, m = x.shape
    x = x.astype(complex)

    # Step 2: Apply 1D FFT along the rows (axis 1)
    x = np.apply_along_axis(fft1d, axis=1, arr=x)

    # Step 3: Transpose the matrix to apply FFT along columns (which are now rows)
    x = x.T

    # Step 4: Apply 1D FFT along the new rows (originally columns)
    x = np.apply_along_axis(fft1d, axis=1, arr=x)

    # Step 5: Transpose back to original order
    x = x.T

    return x


def fft3d(x):
    n, m, l = x.shape
    x = x.astype(complex)

    # Step 1: Apply 2D FFT to each "plane" along the first axis
    x = x.reshape(n, m, l)
    for i in range(n):
        x[i, :, :] = fft2d(x[i, :, :])

    # Step 2: Reshape for applying FFT along the first axis
    x = x.transpose(1, 2, 0).reshape(m * l, n)

    # Step 3: Apply 1D FFT along the first axis (originally third axis)
    x = np.apply_along_axis(fft1d, axis=1, arr=x)

    # Step 4: Reshape back to the original shape
    x = x.reshape(m, l, n).transpose(2, 0, 1)

    return x



def create_signal1d(n):
    # t = np.arange(0, n) / n
    # signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
    # return signal
    return np.sin(np.arange(0, n) / n)

def create_signal2d(n):
    # t_x = np.arange(0, n).reshape(n, 1) / n
    # t_y = np.arange(0, n).reshape(1, n) / n
    # signal = np.sin(2 * np.pi * 10 * t_x) + 0.5 * np.sin(2 * np.pi * 20 * t_y)
    # return signal
    return np.sin(np.arange(0, n*n).reshape(n, n) / n**2)

def create_signal3d(n):
    # t_x = np.arange(0, n).reshape(n, 1, 1) / n
    # t_y = np.arange(0, n).reshape(1, n, 1) / n
    # t_z = np.arange(0, n).reshape(1, 1, n) / n
    # signal = np.sin(2 * np.pi * 10 * t_x) + 0.5 * np.sin(2 * np.pi * 20 * t_y) + np.sin(2 * np.pi * 30 * t_z)
    # return signal
    return np.sin(np.arange(0, n*n*n).reshape(n, n, n) / n**3)

def main():
    # Create test signals
    signal_1d = create_signal1d(1024)
    signal_2d = create_signal2d(128)  # Use smaller size for 2D and 3D to avoid memory issues
    signal_3d = create_signal3d(32)  # Further reduce size for 3D

    # Test 1D FFT
    start_time = time.time()
    fft1d_result = fft1d(signal_1d.copy())  # Use copy to avoid in-place modifications
    fft1d_time = time.time() - start_time
    start_time = time.time()
    fft1d_result_np = np.fft.fft(signal_1d)  # Compare against NumPy's FFT
    fft1d_time_np = time.time() - start_time
    print("1D FFT Comparison:", np.allclose(fft1d_result, fft1d_result_np, atol=1e-10))
    print("1D FFT time taken:", fft1d_time)
    print("1D FFT time taken (NumPy):", fft1d_time_np)
    # print(fft1d_result_np)

    # Test 2D FFT
    start_time = time.time()
    fft2d_result = fft2d(signal_2d.copy())  # Use copy to avoid in-place modifications
    fft2d_time = time.time() - start_time
    start_time = time.time()
    fft2d_result_np = np.fft.fft2(signal_2d)  # Compare against NumPy's 2D FFT
    fft2d_time_np = time.time() - start_time
    print("2D FFT Comparison:", np.allclose(fft2d_result, fft2d_result_np, atol=1e-10))
    print("2D FFT time taken:", fft2d_time)
    print("2D FFT time taken (NumPy):", fft2d_time_np)
    # print(fft2d_result_np)

    # Test 3D FFT
    start_time = time.time()
    fft3d_result = fft3d(signal_3d.copy())  # Use copy to avoid in-place modifications
    fft3d_time = time.time() - start_time
    start_time = time.time()
    fft3d_result_np = np.fft.fftn(signal_3d)  # Compare against NumPy's 3D FFT
    fft3d_time_np = time.time() - start_time
    print("3D FFT Comparison:", np.allclose(fft3d_result, fft3d_result_np, atol=1e-10))
    print("3D FFT time taken:", fft3d_time)
    print("3D FFT time taken (NumPy):", fft3d_time_np)
    # print(fft3d_result_np)

if __name__ == "__main__":
    main()
