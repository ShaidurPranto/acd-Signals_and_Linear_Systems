import numpy as np
import matplotlib.pyplot as plt
import time

def DFT(signal):
    n = len(signal)
    X = np.zeros(n, dtype=complex)
    for k in range(n):
        for t in range(n):
            X[k] += signal[t] * np.exp(-2j * np.pi * k * t / n)
    return X

def IDFT(freq_signal):
    n = len(freq_signal)
    x = np.zeros(n, dtype=complex)
    for t in range(n):
        for k in range(n):
            x[t] += freq_signal[k] * np.exp(2j * np.pi * k * t / n)
    return x / n

def FFT(signal):
    n = len(signal)
    if n <= 1:
        return signal
    even = FFT(signal[0::2])
    odd = FFT(signal[1::2])
    T = [np.exp(-2j * np.pi * k / n) * odd[k] for k in range(n // 2)]
    return np.concatenate([even + T, even - T])

def IFFT(freq_signal):
    n = len(freq_signal)
    time_signal = FFT(np.conj(freq_signal))
    return np.conj(time_signal) / n

def generate_random_signal(n):
    return np.random.rand(n)

def measure_runtimes(n_values, num_trials=10):
    dft_times, idft_times = [], []
    fft_times, ifft_times = [], []
    
    for n in n_values:
        signal = generate_random_signal(n)
        freq_signal = np.random.rand(n) + 1j * np.random.rand(n)
        
        # Measure DFT runtime
        dft_start = time.time()
        for _ in range(num_trials):
            DFT(signal)
        dft_end = time.time()
        dft_times.append((dft_end - dft_start) / num_trials)
        
        # Measure IDFT runtime
        idft_start = time.time()
        for _ in range(num_trials):
            IDFT(freq_signal)
        idft_end = time.time()
        idft_times.append((idft_end - idft_start) / num_trials)
        
        # Measure FFT runtime
        fft_start = time.time()
        for _ in range(num_trials):
            FFT(signal)
        fft_end = time.time()
        fft_times.append((fft_end - fft_start) / num_trials)
        
        # Measure IFFT runtime
        ifft_start = time.time()
        for _ in range(num_trials):
            IFFT(freq_signal)
        ifft_end = time.time()
        ifft_times.append((ifft_end - ifft_start) / num_trials)
    
    return dft_times, idft_times, fft_times, ifft_times

def main():
    n_values = [2**k for k in range(2, 10)] 
    dft_times, idft_times, fft_times, ifft_times = measure_runtimes(n_values)

    # Plot Runtime Comparisons
    plt.figure(figsize=(12, 8))
    plt.plot(n_values, dft_times, label="DFT Runtime", marker='o', color='red')
    plt.plot(n_values, idft_times, label="IDFT Runtime", marker='o', color='orange')
    plt.plot(n_values, fft_times, label="FFT Runtime", marker='x', color='blue')
    plt.plot(n_values, ifft_times, label="IFFT Runtime", marker='x', color='green')
    plt.xlabel("Signal Size (N)")
    plt.ylabel("Average Runtime (seconds)")
    plt.title("DFT/FFT and IDFT/IFFT Runtime Comparison")
    plt.legend()
    plt.grid()
    plt.show()

main()