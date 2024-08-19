
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# # Custom implementation of the 1D FFT (Cooley-Tukey algorithm)
# def fft(x):
#     x = np.asarray(x, dtype=np.complex128)  # Ensure x is a numpy array of complex numbers
#     n = len(x)
    
#     if n <= 1:
#         return x
    
#     # Recursive FFT implementation
#     even = fft(x[::2])
#     odd = fft(x[1::2])
    
#     # Calculate the twiddle factors
#     k = np.arange(n // 2)
#     twiddle_factors = np.exp(-2j * np.pi * k / n)
    
#     # Combine results
#     T = twiddle_factors * odd
#     result = np.concatenate([even + T, even - T])
    
#     return result

import numpy as np

def bit_reversal_permutation(x):
    n = len(x)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j >= bit:
            j -= bit
            bit >>= 1
        j += bit
        if i < j:
            x[i], x[j] = x[j], x[i]
    return x

def fft_sequential(x):
    n = len(x)
    log_n = int(np.log2(n))

    # Bit-reversal permutation
    x = bit_reversal_permutation(np.asarray(x, dtype=np.complex128))

    # Iterative FFT
    for s in range(1, log_n + 1):
        m = 2**s
        m2 = m // 2
        twiddle_factors = np.exp(-2j * np.pi * np.arange(m2) / m)
        
        for k in range(0, n, m):
            for j in range(m2):
                t = twiddle_factors[j] * x[k + j + m2]
                x[k + j + m2] = x[k + j] - t
                x[k + j] = x[k + j] + t
    
    return x

  
def fft(x):
    x = np.asarray(x, dtype=np.complex128)
    n = len(x)
    
    if n <= 1:
        return x
    
    # First level of recursion (even/odd split)
    even_x = x[::2]
    odd_x = x[1::2]
    
    # Second level of recursion (split even_x and odd_x into their even/odd)
    even_even = even_x[::2]
    even_odd = even_x[1::2]
    odd_even = odd_x[::2]
    odd_odd = odd_x[1::2]
    
    # Execute the four subproblems in parallel
    with ThreadPoolExecutor() as executor:
        future_even_even = executor.submit(fft_sequential, even_even)
        future_even_odd = executor.submit(fft_sequential, even_odd)
        future_odd_even = executor.submit(fft_sequential, odd_even)
        future_odd_odd = executor.submit(fft_sequential, odd_odd)
        
        # Collect the results
        even_even_res = future_even_even.result()
        even_odd_res = future_even_odd.result()
        odd_even_res = future_odd_even.result()
        odd_odd_res = future_odd_odd.result()
    
    # Combine results to form the final FFT
    n2 = n // 2
    n4 = n // 4
    
    k = np.arange(n4)
    twiddle_factors_n4 = np.exp(-2j * np.pi * k / n2)
    
    T0 = twiddle_factors_n4 * even_odd_res
    T1 = twiddle_factors_n4 * odd_odd_res
    
    even = np.concatenate([even_even_res + T0, even_even_res - T0])
    odd = np.concatenate([odd_even_res + T1, odd_even_res - T1])
    
    twiddle_factors_n = np.exp(-2j * np.pi * np.arange(n2) / n)
    T = twiddle_factors_n * odd
    
    return np.concatenate([even + T, even - T])



    
# Custom implementation of the 1D IFFT (inverse FFT)
def ifft(x):
    x_conj = np.conjugate(x)
    return np.conjugate(fft(x_conj)) / len(x)

# Updated Chirp-Z Transform implementation using custom FFT and IFFT
def czt(x, m=None, w=None, a=None):
    n = len(x)
    if m is None:
        m = n  # Default m to length of x
    if w is None:
        w = np.exp(-2j * np.pi / m)  # Default chirp multiplier
    if a is None:
        a = 1.0  # Default starting point, ensure it's a float
    
    # Generate the chirp sequence
    chirp = w ** (np.arange(1 - n, max(m, n)) ** 2 / 2.0)
    
    # Next power of 2 to zero-pad the sequence for convolution
    N2 = int(2 ** np.ceil(np.log2(m + n - 1)))
    
    # Prepare the sequences for convolution
    xp = np.append(x * (a ** -np.arange(n)) * chirp[n - 1: n + n - 1], np.zeros(N2 - n))
    ichirpp = np.append(1 / chirp[: m + n - 1], np.zeros(N2 - (m + n - 1)))
    
    # Perform convolution using custom FFT and IFFT
    r = ifft(np.array(fft(xp)) * np.array(fft(ichirpp)))
    
    # Final scaling
    result = r[n - 1: m + n - 1] * chirp[n - 1: m + n - 1]
    
    return result

import time 

import numpy as np
import pyfftw
import time

def main():
    # Test the implementation with a larger input size
    x = np.random.random(2**20)  # Generate a random input array of size 32000

    # # Calculate the CZT using custom FFT/IFFT
    # start = time.time()
    # czt_result_custom_fft = czt(x)
    # time_custom = time.time() - start

    # Calculate the FFT using FFTW
    start = time.time()
    fftw_result = pyfftw.interfaces.numpy_fft.fft(x)
    time_fftw = time.time() - start

    # # Print comparison of the results
    # print("CZT Result with Custom FFT (first 10 elements):", np.round(czt_result_custom_fft[:10], 4))
    # print("FFTW Result (first 10 elements):", np.round(fftw_result[:10], 4))

    # # Compute and print the difference between the results
    # difference = np.abs(czt_result_custom_fft - fftw_result)
    # print("Difference (first 10 elements):", np.round(difference[:10], 4))

    # # Check if the difference is within a small tolerance
    # tolerance = 1e-10
    # print("Maximum difference:", np.max(difference))
    # print("Average difference:", np.mean(difference))
    # print("All differences within tolerance:", np.all(difference < tolerance))

    # print("Time custom", time_custom)
    print("Time FFTW:", time_fftw)

if __name__ == "__main__":
    main()